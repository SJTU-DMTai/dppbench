import os
import sys
import json
import yaml
import time
import re
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from .llm_caller import LLMCaller
from .executor import TrainingExecutor
from .op_registry import get_operator_descriptions, get_operator_summary


SYSTEM_PROMPT = """You are a data preprocessing optimization agent. Your goal is to improve the model's performance (AUC) by modifying the preprocessing pipeline (pre_process.yaml).

## Constraints
- You can ONLY modify the pre_process.yaml file.
- You CANNOT modify the raw data or the model parameters.
- You should use operators from the available operator library.

## Available Actions
You must respond with exactly ONE of the following actions in each turn:

1. **THINK**: Analyze the current situation and plan next steps.
   Format:
   <action>THINK</action>
   <content>Your analysis and reasoning here</content>

2. **MODIFY_YAML**: Modify the pre_process.yaml file.
   Format:
   <action>MODIFY_YAML</action>
   <content>
   The complete new YAML content (will replace the entire file)
   </content>

3. **TRAIN**: Execute training with the current pre_process.yaml and observe results.
   Format:
   <action>TRAIN</action>
   <content>Reason for triggering training</content>

4. **FINISH**: End the optimization loop.
   Format:
   <action>FINISH</action>
   <content>Summary of what was achieved</content>
"""


def build_context_message(executor, op_summary, history):
    data_summary = executor.get_data_summary()
    current_yaml = executor.get_current_yaml()
    model_cfg = executor.get_model_config()

    task_type = data_summary.get("task_type", "unknown")

    if task_type == "tabular":
        context = f"""## Dataset Summary (Tabular)
- Train shape: {data_summary['train_shape']}
- Target column: {data_summary['target_col']}
- Target distribution: {data_summary['target_distribution']}
- Null ratio (mean): {data_summary['null_ratio']:.4f}
- Auxiliary tables: {json.dumps(data_summary.get('auxiliary_tables', {}), indent=2)}
- Sample columns (first 30): {data_summary['columns']}
"""
    else:
        context = f"""## Dataset Summary (Recommendation)
- Interaction shape: {data_summary['interaction_shape']}
- Column types: {json.dumps(data_summary.get('col_types', {}), indent=2)}
- Null ratio (mean): {data_summary['null_ratio']:.4f}
- Has user_df: {data_summary.get('has_user_df')} (shape: {data_summary.get('user_df_shape')})
- Has item_df: {data_summary.get('has_item_df')} (shape: {data_summary.get('item_df_shape')})
- Sample columns: {data_summary['columns']}
"""

    context += f"""
## Model Configuration
{yaml.dump(model_cfg, default_flow_style=False)}

## Current pre_process.yaml
```yaml
{current_yaml}
```

## Available Operators
{op_summary}

## Optimization History
"""
    if not history:
        context += "No previous runs yet. Start by running training to establish a baseline.\n"
    else:
        for i, entry in enumerate(history):
            context += f"\n### Run {i+1}\n"
            if entry.get("metrics"):
                context += f"- Metrics: {entry['metrics']}\n"
            if entry.get("train_shape"):
                context += f"- Features: {entry.get('n_features')}\n"
            if entry.get("error"):
                context += f"- Error: {entry['error'][:300]}\n"
            if entry.get("yaml_change"):
                context += f"- Change: {entry['yaml_change']}\n"
            if entry.get("duration_seconds"):
                context += f"- Duration: {entry['duration_seconds']}s\n"

    return context


def parse_action(response):
    action_match = re.search(r'<action>(.*?)</action>', response, re.DOTALL)
    content_match = re.search(r'<content>(.*?)</content>', response, re.DOTALL)

    if not action_match:
        return "THINK", response

    action = action_match.group(1).strip().upper()
    content = content_match.group(1).strip() if content_match else ""

    return action, content


class PreprocOptAgent:
    def __init__(self, task_dir, data_name, data_dir=None, max_turns=6, model=None, api_key=None, base_url=None, verbose=True, device="cpu", small_n=3000, fast_train=True):
        self.executor = TrainingExecutor(task_dir, data_name=data_name, data_dir=data_dir, device=device)
        self.llm = LLMCaller(model=model, api_key=api_key, base_url=base_url)
        self.max_turns = max_turns
        self.verbose = verbose
        self.device = device
        self.history = []
        self.messages = []
        self.best_metrics = None
        self.best_yaml = None
        self.small_n = int(small_n) if small_n else 0
        self.fast_train = bool(fast_train)

        # Apply search-time fast training overrides on the executor.
        if self.fast_train:
            self.executor.set_fast_mode(True)

        # Apply search-time row subsampling by patching _make_data_instance the
        # same way CtxPipeEvaluator does (so each load_data() call subsamples).
        if self.small_n > 0:
            from baselines.CtxPipe.evaluator import _subsample_data_inplace
            self._orig_make_data_instance = self.executor._make_data_instance
            small_n = self.small_n

            def _patched_make_data_instance():
                data = self._orig_make_data_instance()
                original_load_data = data.load_data

                def patched_load_data(*args, **kwargs):
                    ret = original_load_data(*args, **kwargs)
                    try:
                        _subsample_data_inplace(data, small_n=small_n, seed=42)
                    except Exception:
                        pass
                    return ret

                data.load_data = patched_load_data
                return data

            self.executor._make_data_instance = _patched_make_data_instance

    def _log(self, msg):
        if self.verbose:
            print(msg)

    def _init_messages(self):
        op_summary = get_operator_summary()
        context = build_context_message(self.executor, op_summary, self.history)
        self.messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": context + "\nPlease start by establishing a baseline: run TRAIN first, then analyze results and propose improvements."},
        ]

    def _append_observation(self, observation):
        self.messages.append({"role": "user", "content": f"<observation>\n{observation}\n</observation>"})

    def _step(self):
        response = self.llm.query(self.messages)
        self.messages.append({"role": "assistant", "content": response})
        self._log(f"\n{'='*60}\n[Agent Response]\n{response[:1000]}{'...' if len(response) > 1000 else ''}\n{'='*60}")

        action, content = parse_action(response)
        return action, content

    def _handle_think(self, content):
        self._log(f"[THINK] {content[:300]}")
        self._append_observation("Thought recorded. Please proceed with your next action (MODIFY_YAML or TRAIN).")

    def _handle_modify_yaml(self, content):
        yaml_content = content
        if "```yaml" in yaml_content:
            yaml_content = re.search(r'```yaml\s*(.*?)```', yaml_content, re.DOTALL)
            yaml_content = yaml_content.group(1) if yaml_content else content
        elif "```" in yaml_content:
            yaml_content = re.search(r'```\s*(.*?)```', yaml_content, re.DOTALL)
            yaml_content = yaml_content.group(1) if yaml_content else content

        try:
            parsed = yaml.safe_load(yaml_content)
            if not isinstance(parsed, dict) or "pipeline" not in parsed:
                self._append_observation("Error: YAML must contain a 'pipeline' key at the top level. Please fix and retry.")
                return
        except yaml.YAMLError as e:
            self._append_observation(f"Error: Invalid YAML syntax: {e}. Please fix and retry.")
            return

        self.executor.write_yaml(yaml_content)
        self._log(f"[MODIFY_YAML] Pipeline updated ({len(parsed['pipeline'])} steps)")
        self._append_observation(f"Working pre_process.yaml has been updated successfully with {len(parsed['pipeline'])} pipeline steps (original file is unchanged). You can now run TRAIN to see results.")

    def _handle_train(self, content):
        self._log(f"[TRAIN] Running training pipeline...")
        result = self.executor.run_training()

        entry = {
            "metrics": result.get("metrics"),
            "train_shape": result.get("train_shape"),
            "n_features": result.get("n_features"),
            "error": result.get("error"),
            "duration_seconds": result.get("duration_seconds"),
            "yaml_change": content[:200],
        }
        self.history.append(entry)

        if result["success"]:
            metrics = result["metrics"]
            self._log(f"[TRAIN] Success! Metrics: {metrics}")

            if self.best_metrics is None or metrics.get("auc", 0) > self.best_metrics.get("auc", 0):
                self.best_metrics = metrics
                self.best_yaml = self.executor.get_current_yaml()
                self._log(f"[TRAIN] New best AUC: {metrics.get('auc', 0):.4f}")

            obs = f"""Training completed successfully!
- Metrics: {json.dumps(metrics)}
- Number of features: {result.get('n_features')}
- Training duration: {result.get('duration_seconds')}s
- Top features: {json.dumps(result.get('top_features', [])[:10])}
- Best AUC so far: {self.best_metrics.get('auc', 0):.4f}

Analyze the results and decide whether to:
1. Try a different preprocessing strategy (MODIFY_YAML)
2. Finish if satisfied (FINISH)"""
        else:
            self._log(f"[TRAIN] Failed! Error: {result['error'][:200]}")
            obs = f"""Training FAILED with error:
{result['error']}

Please fix the pre_process.yaml and try again (MODIFY_YAML)."""

        self._append_observation(obs)

    def run(self):
        self._log("=" * 60)
        self._log("Preprocessing Optimization Agent Started")
        self._log("=" * 60)

        self._init_messages()

        for turn in range(self.max_turns):
            self._log(f"\n{'─'*40} Turn {turn+1}/{self.max_turns} {'─'*40}")

            action, content = self._step()

            if action == "THINK":
                self._handle_think(content)
            elif action == "MODIFY_YAML":
                self._handle_modify_yaml(content)
            elif action == "TRAIN":
                self._handle_train(content)
            elif action == "FINISH":
                self._log(f"\n[FINISH] {content}")
                break
            else:
                self._append_observation(f"Unknown action '{action}'. Use THINK, MODIFY_YAML, TRAIN, or FINISH.")

        output_path = None
        eval_error = None
        if self.best_yaml:
            output_path = self.executor.save_best_yaml(self.best_yaml)

            # ---- Final full-data evaluation ----
            # Restore full data and disable fast-mode overrides, then re-run
            # training on the best YAML so reported metrics match the rest of
            # the benchmark.
            if self.fast_train or self.small_n > 0:
                self._log("\n[FINAL_EVAL] Re-running best pipeline on full data...")
                self.executor.set_fast_mode(False)
                if self.small_n > 0 and hasattr(self, "_orig_make_data_instance"):
                    self.executor._make_data_instance = self._orig_make_data_instance
                    self.executor._data = None
                # Make sure the working YAML is the best one before re-training.
                self.executor.write_yaml(self.best_yaml)
                full_result = self.executor.run_training()
                if full_result.get("success"):
                    self.best_metrics = full_result["metrics"]
                    self._log(f"[FINAL_EVAL] Full-data metrics: {self.best_metrics}")
                else:
                    eval_error = full_result.get("error")
                    self._log(f"[FINAL_EVAL] Failed: {eval_error}")

            self._log(f"\n{'='*60}")
            self._log(f"Optimization complete. Best AUC: {self.best_metrics.get('auc', 0):.4f}")
            self._log(f"Best pipeline saved to: {output_path}")
            self._log(f"Original pre_process.yaml is unchanged.")
            self._log(f"Total LLM tokens used: {self.llm.total_tokens_used}")
            self._log(f"{'='*60}")

        return {
            "best_metrics": self.best_metrics,
            "best_yaml": self.best_yaml,
            "best_pipeline_yaml": self.best_yaml,
            "best_pipeline_path": output_path,
            "best_fitness": self.best_metrics.get("auc") if self.best_metrics else None,
            "eval_error": eval_error,
            "history": self.history,
            "total_turns": turn + 1,
            "total_tokens": self.llm.total_tokens_used,
            "work_dir": self.executor._work_dir,
        }
