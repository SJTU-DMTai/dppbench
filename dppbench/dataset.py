import importlib.util
import importlib.machinery
import inspect
import os
import re
import sys
import yaml
import pandas as pd


def _class_to_module(class_name):
    s = re.sub(r"(?<!^)(?=[A-Z])", "_", class_name).lower()
    return s


_OP_SUBDIRS = (
    "integration", "cleaning", "preprocessing",
    "feature_engineering", "transformation",
)


def _resolve_op_path(operator_dir, module_name):
    """Locate <module_name>.py under operator_dir or any of its known subdirs.

    Returns (full_dotted_suffix, file_path) or (None, None) if not found.
    """
    root = os.path.join(operator_dir, f"{module_name}.py")
    if os.path.exists(root):
        return module_name, root
    for sub in _OP_SUBDIRS:
        candidate = os.path.join(operator_dir, sub, f"{module_name}.py")
        if os.path.exists(candidate):
            return f"{sub}.{module_name}", candidate
    return None, None


def _ensure_base_op_loaded(pkg_name, operator_dir):
    base_full = f"{pkg_name}.base_op"
    if base_full in sys.modules:
        return
    base_file = os.path.join(operator_dir, "base_op.py")
    base_spec = importlib.util.spec_from_file_location(base_full, base_file)
    base_mod = importlib.util.module_from_spec(base_spec)
    sys.modules[base_full] = base_mod
    base_spec.loader.exec_module(base_mod)


def _ensure_subpkg_registered(pkg_name, sub_pkg, operator_dir):
    sub_full = f"{pkg_name}.{sub_pkg}"
    if sub_full in sys.modules:
        return
    sub_spec = importlib.machinery.ModuleSpec(sub_full, loader=None, is_package=True)
    sub_module = importlib.util.module_from_spec(sub_spec)
    sub_module.__path__ = [os.path.join(operator_dir, sub_pkg)]
    sys.modules[sub_full] = sub_module


class BaseData:
    def __init__(self, name):
        self.name = name

    def load_data(self):
        raise NotImplementedError("load_data method not implemented")

    @staticmethod
    def _relation_dag_spec(config):
        dag = config.get("dag")
        if not isinstance(dag, dict):
            return None
        train = dag.get("train")
        if (
            isinstance(dag.get("sources"), list)
            and isinstance(dag.get("ops"), list)
            and isinstance(train, dict)
            and "prev" in train
        ):
            return dag
        return None

    @staticmethod
    def _relation_source_entry(entry):
        if isinstance(entry, str):
            return entry, None
        if isinstance(entry, dict):
            source_id = entry.get("id")
            table_name = (
                entry.get("table")
                or entry.get("df")
                or entry.get("source")
                or entry.get("name")
            )
            return source_id, table_name
        raise ValueError(f"Invalid relation DAG source entry {entry!r}")

    def _relation_dag_sources(self, spec, available_sources):
        sources = spec.get("sources")
        if not isinstance(sources, list):
            raise ValueError("Relation DAG sources must be a list")

        default_names = [name for name in available_sources if name != "train"]
        seen = set()
        source_ids = []
        source_tables = {}
        for entry in sources:
            source_id, table_name = self._relation_source_entry(entry)
            if not isinstance(source_id, str) or not re.fullmatch(r"s\d+", source_id):
                raise ValueError(
                    f"Relation DAG source id must match s<number>: {source_id!r}"
                )
            if source_id in seen:
                raise ValueError(f"Relation DAG source {source_id!r} is duplicated")

            if table_name is None:
                source_idx = int(source_id[1:])
                if source_idx < 0 or source_idx >= len(default_names):
                    raise ValueError(f"Relation DAG source {source_id!r} is unavailable")
                table_name = default_names[source_idx]
            if not isinstance(table_name, str) or table_name not in available_sources:
                raise ValueError(
                    f"Relation DAG source {source_id!r} references unavailable "
                    f"table {table_name!r}; available: {list(available_sources)}"
                )

            seen.add(source_id)
            source_ids.append(source_id)
            source_tables[source_id] = available_sources[table_name]
        return source_ids, source_tables

    def _normalise_relation_ref(self, ref, op_ids, source_ids):
        if ref == "train":
            return "train"
        if not isinstance(ref, str):
            raise ValueError(f"Invalid DAG node reference {ref!r}")
        if ref in source_ids or ref in op_ids:
            return ref
        raise ValueError(f"Invalid DAG node reference {ref!r}")

    def _relation_dag_ops(self, spec, source_ids):
        ops = spec.get("ops") or []
        if not isinstance(ops, list):
            raise ValueError("Relation DAG 'ops' must be a list")

        normalised = []
        seen = set()
        for idx, node in enumerate(ops):
            if not isinstance(node, dict) or "op" not in node:
                raise ValueError(f"Relation DAG op #{idx} must contain 'op'")
            op_id = node.get("id")
            if op_id is None:
                raise ValueError(f"Relation DAG op #{idx} must contain 'id'")
            if not isinstance(op_id, str) or not op_id:
                raise ValueError(f"Relation DAG op #{idx} has invalid id {op_id!r}")
            if op_id == "train":
                raise ValueError("Relation DAG op id cannot be 'train'")
            if op_id in source_ids:
                raise ValueError(
                    f"Relation DAG op id {op_id!r} conflicts with a source id"
                )
            if op_id in seen:
                raise ValueError(f"Relation DAG op id {op_id!r} is duplicated")
            step = dict(node)
            step["id"] = op_id
            seen.add(op_id)
            normalised.append(step)
        return normalised

    def _relation_dag_edges_from_prev(self, spec, ops, source_ids):
        op_ids = [step["id"] for step in ops]
        edges = []
        for step in ops:
            prev = step.get("prev")
            if not isinstance(prev, list) or not prev:
                raise ValueError(
                    f"Relation DAG op {step['id']!r} must contain non-empty prev"
                )
            for raw_src in prev:
                src = self._normalise_relation_ref(raw_src, op_ids, source_ids)
                if src == "train":
                    raise ValueError("Relation DAG op prev cannot reference train")
                edges.append((src, step["id"]))

        train = spec.get("train")
        train_prev = train.get("prev") if isinstance(train, dict) else None
        if not isinstance(train_prev, list) or len(train_prev) != 1:
            raise ValueError("Relation DAG train.prev must contain exactly one node")
        src = self._normalise_relation_ref(train_prev[0], op_ids, source_ids)
        if src == "train":
            raise ValueError("Relation DAG train.prev cannot reference train")
        edges.append((src, "train"))
        return edges

    def _toposort_relation_dag(self, op_ids, source_ids, edges):
        op_refs = list(op_ids)
        op_set = set(op_refs)
        source_set = set(source_ids)
        deps = {ref: set() for ref in op_refs}
        children = {ref: set() for ref in op_refs}
        incoming = {ref: [] for ref in op_refs}
        incoming["train"] = []

        for src, dst in edges:
            if src not in source_set and src not in op_set:
                raise ValueError(f"Relation DAG references unknown source node {src!r}")
            if dst != "train" and dst not in op_set:
                raise ValueError(f"Relation DAG references unknown destination {dst!r}")
            if dst == "train":
                incoming["train"].append(src)
            else:
                incoming[dst].append(src)
                if src in op_set:
                    deps[dst].add(src)
                    children[src].add(dst)

        if len(incoming["train"]) != 1:
            raise ValueError("Relation DAG must have exactly one edge into train")
        for ref in op_refs:
            if not incoming[ref]:
                raise ValueError(f"Relation DAG {ref} has no input edge")

        ready = [ref for ref in op_refs if not deps[ref]]
        order = []
        while ready:
            ref = ready.pop(0)
            order.append(ref)
            for child in sorted(children[ref]):
                deps[child].discard(ref)
                if not deps[child]:
                    ready.append(child)
        if len(order) != len(op_refs):
            raise ValueError("Relation DAG contains a cycle")

        reachable = set()
        for ref in order:
            if any(src in source_set or src in reachable for src in incoming[ref]):
                reachable.add(ref)
        if len(reachable) != len(op_refs):
            missing = sorted(set(op_refs) - reachable)
            raise ValueError(f"Relation DAG has nodes unreachable from sources: {missing}")

        reverse = {ref: [] for ref in op_refs + ["train"]}
        for src, dst in edges:
            reverse[dst].append(src)
        ancestors = set()
        stack = list(reverse["train"])
        while stack:
            ref = stack.pop()
            if ref in ancestors:
                continue
            ancestors.add(ref)
            if ref in op_set:
                stack.extend(reverse[ref])
        unused = sorted(set(op_refs) - ancestors)
        if unused:
            raise ValueError(f"Relation DAG has nodes disconnected from train: {unused}")

        return order, incoming

    def _bind_relation_secondary_inputs(self, op_cls, params, secondary_dfs, op_name):
        if not secondary_dfs:
            return params
        signature = inspect.signature(op_cls.__init__)
        accepted = set(signature.parameters)
        params = dict(params)

        if (
            params.get("method") == "rec"
            and "user_df" in accepted
            and "item_df" in accepted
        ):
            user_col = params.get("user_col", "user_id")
            item_col = params.get("item_col", "item_id")
            for side_df in secondary_dfs:
                if user_col in side_df.columns and "user_df" not in params:
                    params["user_df"] = side_df
                elif item_col in side_df.columns and "item_df" not in params:
                    params["item_df"] = side_df
                elif "item_df" not in params:
                    params["item_df"] = side_df
                elif "user_df" not in params:
                    params["user_df"] = side_df
                else:
                    raise ValueError(
                        f"Relation DAG op {op_name} received too many rec side inputs"
                    )
            return params

        if "other_dfs" in accepted:
            if "other_dfs" in params:
                raise ValueError(f"Relation DAG op {op_name} already sets other_dfs")
            params["other_dfs"] = secondary_dfs
        elif "aux_df" in accepted:
            if len(secondary_dfs) != 1:
                raise ValueError(
                    f"Relation DAG op {op_name} accepts one auxiliary input, "
                    f"but received {len(secondary_dfs)}"
                )
            if "aux_df" in params:
                raise ValueError(f"Relation DAG op {op_name} already sets aux_df")
            params["aux_df"] = secondary_dfs[0]
        else:
            raise ValueError(
                f"Relation DAG op {op_name} has {len(secondary_dfs) + 1} inputs, "
                "but its constructor does not accept aux_df or other_dfs"
            )
        return params

    def _run_relation_dag(self, config, available_sources, transform_fn):
        spec = self._relation_dag_spec(config)
        source_ids, source_tables = self._relation_dag_sources(spec, available_sources)
        ops = self._relation_dag_ops(spec, source_ids)
        op_ids = [step["id"] for step in ops]
        edges = self._relation_dag_edges_from_prev(spec, ops, source_ids)
        order, incoming = self._toposort_relation_dag(op_ids, source_ids, edges)
        values = dict(source_tables)
        op_by_id = {step["id"]: step for step in ops}

        total = len(order)
        for step_no, ref in enumerate(order, start=1):
            step = op_by_id[ref]
            op_name = step["op"]
            input_dfs = [values[src] for src in incoming[ref]]
            raw_params = step.get("params", {}) or {}
            params = {k: self._resolve_param(v) for k, v in raw_params.items()}

            op_cls = self._load_op(op_name)
            params = self._bind_relation_secondary_inputs(
                op_cls, params, input_dfs[1:], op_name
            )
            op = op_cls(**params)
            df = input_dfs[0]
            new_df = transform_fn(df, op, op_name)
            values[ref] = new_df
            print(
                f"DAG step {step_no}/{total}: {op_name} {ref}, "
                f"shape: {df.shape} -> {new_df.shape}"
            )

        sink_ref = incoming["train"][0]
        return values[sink_ref]

    @staticmethod
    def _std_test_dir(task_data_dir):
        """Return the conventional std-test directory next to the task's
        data folder. Each task data class sets ``self.data_dir``; the
        std_test artifacts live one level up (alongside ``data/``) in
        ``std_test/``.
        """
        if not task_data_dir:
            return None
        return os.path.join(os.path.dirname(task_data_dir), "std_test")


class RecData(BaseData):
    OPERATOR_DIR = os.path.join(os.path.dirname(__file__), " operators")

    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    TEXT = "text"
    TIMESTAMP = "timestamp"
    NUMERIC_LIST = "numeric_list"
    CATEGORICAL_LIST = "categorical_list"

    VALID_COL_TYPES = {
        NUMERIC, CATEGORICAL, TEXT,
        TIMESTAMP, NUMERIC_LIST, CATEGORICAL_LIST,
    }

    def __init__(self, name):
        super().__init__(name)
        self.data_type = "rec data"
        self.interaction_df = None
        self.item_df = None
        self.user_df = None
        self.col_types = {}
        self._user_id_col = "user_id"
        self._item_id_col = "item_id"
        self._item_id_related_cols = []
        self._user_id_related_cols = []
        self._has_std_test = False
        self.std_test_negatives_df = None
        self.model_cfg = {}

    def set_model_config(self, cfg):
        self.model_cfg = cfg or {}

    def register_col_type(self, col_name, col_type):
        if col_type not in self.VALID_COL_TYPES:
            raise ValueError(
                f"Invalid col_type '{col_type}'. Must be one of {self.VALID_COL_TYPES}"
            )
        self.col_types[col_name] = col_type

    def register_col_types(self, type_map):
        for col_name, col_type in type_map.items():
            self.register_col_type(col_name, col_type)

    def _apply_label_rule(self, target_col, threshold=None,
                          positive_label=1, negative_label=0,
                          mode="ge"):
        """Freeze the binary label of the interaction table at data-load time.

        Modes:
          - "ge"   (default): df[target_col] >= threshold -> positive_label
          - "passthrough"   : assert values already subset of {pos,neg};
                              cast to int8.

        After this call, the target column is guaranteed to be int8 with
        values in {negative_label, positive_label}, and registered as
        CATEGORICAL so downstream embedding paths treat it as a label.
        """
        import numpy as np
        df = self.interaction_df
        if df is None or target_col not in df.columns:
            raise ValueError(
                f"_apply_label_rule: target_col={target_col!r} not found "
                f"in interaction_df"
            )
        if mode == "passthrough":
            vals = set(df[target_col].dropna().unique().tolist())
            assert vals.issubset({positive_label, negative_label}), (
                f"_apply_label_rule(passthrough) saw non-binary values: {vals}"
            )
            df[target_col] = df[target_col].astype("int8")
        elif mode == "ge":
            if threshold is None:
                raise ValueError("threshold required for mode='ge'")
            df[target_col] = np.where(
                df[target_col] >= threshold,
                positive_label, negative_label,
            ).astype("int8")
        else:
            raise ValueError(f"Unknown mode {mode!r}")
        self.col_types[target_col] = self.CATEGORICAL

    def _apply_configured_label_rule(self, target_col, default_threshold=None,
                                     default_mode="ge"):
        rule_cfg = self.model_cfg.get("feature", {}).get("label_rule", {}) or {}
        mode = rule_cfg.get("mode", default_mode)
        threshold = rule_cfg.get("threshold", default_threshold)
        positive_label = rule_cfg.get("positive_label", 1)
        negative_label = rule_cfg.get("negative_label", 0)
        if not rule_cfg:
            raise ValueError(
                f"Missing feature.label_rule in model.yaml for rec target "
                f"'{target_col}'."
            )
        self._apply_label_rule(
            target_col,
            threshold=threshold,
            positive_label=positive_label,
            negative_label=negative_label,
            mode=mode,
        )

    def _remap_ids(self):
        import numpy as np
        for id_col, related_cols in [
            (self._user_id_col, self._user_id_related_cols),
            (self._item_id_col, self._item_id_related_cols),
        ]:
            unique_vals = set()
            for df_attr in ("interaction_df", "user_df", "item_df", "std_test_negatives_df"):
                df = getattr(self, df_attr, None)
                if df is None:
                    continue
                if id_col in df.columns:
                    unique_vals.update(df[id_col].dropna().unique())
                for rel_col in related_cols:
                    if rel_col in df.columns:
                        for val_list in df[rel_col]:
                            if isinstance(val_list, (list, np.ndarray)):
                                unique_vals.update(val_list)

            if not unique_vals:
                continue

            mapping = {val: idx + 1 for idx, val in enumerate(sorted(unique_vals, key=str))}

            for df_attr in ("interaction_df", "user_df", "item_df", "std_test_negatives_df"):
                df = getattr(self, df_attr, None)
                if df is None:
                    continue
                if id_col in df.columns:
                    df[id_col] = df[id_col].map(mapping)
                for rel_col in related_cols:
                    if rel_col in df.columns:
                        df[rel_col] = df[rel_col].apply(
                            lambda x: [mapping.get(v, 0) for v in x] if isinstance(x, (list, np.ndarray)) else x
                        )

        already_remapped = set(self._item_id_related_cols + self._user_id_related_cols)
        for col, col_type in list(self.col_types.items()):
            if col_type != self.CATEGORICAL_LIST or col in already_remapped:
                continue
            has_list = False
            list_sample_is_str = False
            for df_attr in ("interaction_df", "user_df", "item_df"):
                df = getattr(self, df_attr, None)
                if df is None or col not in df.columns:
                    continue
                for val in df[col]:
                    if isinstance(val, (list, np.ndarray)) and len(val) > 0:
                        has_list = True
                        if not isinstance(val[0], (int, np.integer)):
                            list_sample_is_str = True
                        break
                if has_list:
                    break
            if not has_list or not list_sample_is_str:
                continue
            all_tokens = set()
            for df_attr in ("interaction_df", "user_df", "item_df"):
                df = getattr(self, df_attr, None)
                if df is None or col not in df.columns:
                    continue
                for val in df[col]:
                    if isinstance(val, (list, np.ndarray)):
                        all_tokens.update(val)
            tok2idx = {tok: idx + 1 for idx, tok in enumerate(sorted(all_tokens, key=str))}
            for df_attr in ("interaction_df", "user_df", "item_df"):
                df = getattr(self, df_attr, None)
                if df is None or col not in df.columns:
                    continue
                df[col] = df[col].apply(
                    lambda x: [tok2idx.get(v, 0) for v in x] if isinstance(x, (list, np.ndarray)) else x
                )

        id_cols = {self._user_id_col, self._item_id_col}
        for col, col_type in list(self.col_types.items()):
            if col_type != self.CATEGORICAL or col in id_cols:
                continue
            need_remap = False
            for df_attr in ("interaction_df", "user_df", "item_df", "std_test_negatives_df"):
                df = getattr(self, df_attr, None)
                if df is None or col not in df.columns:
                    continue
                non_null = df[col].dropna()
                if len(non_null) == 0:
                    continue
                sample = non_null.iloc[0]
                if not isinstance(sample, (int, np.integer)):
                    need_remap = True
                    break
                col_min = int(non_null.min())
                col_max = int(non_null.max())
                col_nunique = int(non_null.nunique())
                if col_max - col_min + 1 > col_nunique * 2 + 10:
                    need_remap = True
                    break
            if not need_remap:
                continue

            all_tokens = set()
            for df_attr in ("interaction_df", "user_df", "item_df", "std_test_negatives_df"):
                df = getattr(self, df_attr, None)
                if df is None or col not in df.columns:
                    continue
                all_tokens.update(df[col].dropna().unique())
            tok2idx = {tok: idx + 1 for idx, tok in enumerate(sorted(all_tokens, key=str))}
            for df_attr in ("interaction_df", "user_df", "item_df", "std_test_negatives_df"):
                df = getattr(self, df_attr, None)
                if df is None or col not in df.columns:
                    continue
                df[col] = df[col].map(tok2idx).fillna(0).astype("int64")

    def _resolve_param(self, value):
        if isinstance(value, str) and value.startswith("$"):
            attr = value[1:]
            if not hasattr(self, attr):
                raise ValueError(f"Unknown reference '{value}' in pipeline params")
            return getattr(self, attr)
        if isinstance(value, dict):
            return {k: self._resolve_param(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._resolve_param(v) for v in value]
        return value

    def _load_op(self, op_name):
        pkg_name = "_dppbench_ops"
        if pkg_name not in sys.modules:
            pkg_spec = importlib.machinery.ModuleSpec(
                pkg_name, loader=None, is_package=True
            )
            pkg_module = importlib.util.module_from_spec(pkg_spec)
            pkg_module.__path__ = [self.OPERATOR_DIR]
            sys.modules[pkg_name] = pkg_module

        module_name = _class_to_module(op_name)
        dotted_suffix, module_file = _resolve_op_path(self.OPERATOR_DIR, module_name)
        if module_file is None:
            raise ValueError(
                f"Operator '{op_name}' not found (expected module '{module_name}.py' "
                f"under any of {_OP_SUBDIRS})"
            )
        full_name = f"{pkg_name}.{dotted_suffix}"
        if full_name in sys.modules:
            return getattr(sys.modules[full_name], op_name)

        _ensure_base_op_loaded(pkg_name, self.OPERATOR_DIR)
        if "." in dotted_suffix:
            sub_pkg = dotted_suffix.rsplit(".", 1)[0]
            _ensure_subpkg_registered(pkg_name, sub_pkg, self.OPERATOR_DIR)

        spec = importlib.util.spec_from_file_location(
            full_name, module_file,
            submodule_search_locations=None,
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[full_name] = module
        spec.loader.exec_module(module)
        return getattr(module, op_name)

    def load_std_test_frozen(self):
        """Merge the frozen std-test interactions into ``self.interaction_df``.

        Looks for ``<task>/std_test/std_test.parquet`` and
        ``interaction_frozen.parquet``. When found:
          * If ``interaction_frozen.parquet`` exists, it replaces the
            current ``interaction_df`` (these are the rows with the
            std-test holdout already removed).
          * ``std_test.parquet`` contains held-out positives **plus** fixed
            sampled negatives, already materialized with full columns by
            ``scripts/build_std_test.py``. Those rows are appended with
            ``__split__`` set to ``"std_test"`` so train-only ops leave
            them untouched.

        Returns ``True`` if a std-test was loaded.
        """
        std_dir = self._std_test_dir(getattr(self, "data_dir", None))
        if not std_dir or not os.path.isdir(std_dir):
            return False
        std_test_path = os.path.join(std_dir, "std_test.parquet")
        if not os.path.exists(std_test_path):
            return False

        frozen_path = os.path.join(std_dir, "interaction_frozen.parquet")
        if os.path.exists(frozen_path):
            self.interaction_df = pd.read_parquet(frozen_path)

        std_df = pd.read_parquet(std_test_path).copy()
        std_df["__split__"] = "std_test"

        base = self.interaction_df.copy()
        if "__split__" not in base.columns:
            base["__split__"] = "train"
        for c in std_df.columns:
            if c not in base.columns:
                base[c] = pd.NA
        for c in base.columns:
            if c not in std_df.columns:
                std_df[c] = pd.NA
        self.interaction_df = pd.concat(
            [base, std_df[base.columns]], ignore_index=True
        )

        self._has_std_test = True
        feat_cfg = self.model_cfg.get("feature", {}) if self.model_cfg else {}
        target_col = feat_cfg.get("target_col")
        n_total = len(std_df)
        if target_col and target_col in std_df.columns:
            label_rule = feat_cfg.get("label_rule", {}) or {}
            pos_lbl = label_rule.get("positive_label", 1)
            n_pos = int((std_df[target_col] == pos_lbl).sum())
            n_neg = n_total - n_pos
            print(
                f"  [std_test] loaded {n_pos} held-out interactions + {n_neg} negatives"
            )
        else:
            print(f"  [std_test] loaded {n_total} held-out rows")
        return True

    def _filter_std_test_to_train_domain(self, std_test_df, train_df, op_name, op):
        filtered = std_test_df.copy()
        before = len(filtered)

        for col in (getattr(op, "user_col", None), getattr(op, "item_col", None)):
            if not col or col not in filtered.columns or col not in train_df.columns:
                continue
            allowed = set(train_df[col].dropna().unique())
            filtered = filtered[filtered[col].isin(allowed)]

        feat_cfg = self.model_cfg.get("feature", {}) if self.model_cfg else {}
        target_col = feat_cfg.get("target_col")
        label_rule = feat_cfg.get("label_rule", {}) or {}
        positive_label = label_rule.get("positive_label", 1)
        user_col = getattr(op, "user_col", None)
        if (
            target_col
            and user_col
            and target_col in filtered.columns
            and user_col in filtered.columns
        ):
            positives = filtered[filtered[target_col] == positive_label]
            positive_users = set(positives[user_col].dropna().unique())
            filtered = filtered[filtered[user_col].isin(positive_users)]

        dropped = before - len(filtered)
        if dropped > 0:
            print(
                f"  [std_test] op {op_name} removed {dropped} rows outside "
                "the filtered training domain"
            )
        return filtered.reset_index(drop=True)

    def _relation_dag_available_sources(self):
        sources = {}
        if self.interaction_df is not None:
            sources["interaction"] = self.interaction_df
        if self.user_df is not None:
            sources["user"] = self.user_df
        if self.item_df is not None:
            sources["item"] = self.item_df
        return sources

    def _run_pre_process_relation_dag(self, config):
        available_sources = self._relation_dag_available_sources()
        final_df = self._run_relation_dag(
            config, available_sources,
            lambda df, op, op_name: self._transform_rec_dag_df(df, op, op_name),
        )
        self.interaction_df = final_df
        self._remap_ids()

    def _transform_rec_dag_df(self, df, op, op_name):
        applies_to_std_test = getattr(op, "APPLIES_TO_STD_TEST", True)
        uses_train_history = getattr(op, "USES_TRAIN_HISTORY_FOR_STD_TEST", False)
        if "__split__" in df.columns and uses_train_history:
            new_df = op.transform(df)
        elif "__split__" in df.columns:
            parts = []
            std_test_pre = df[df["__split__"] == "std_test"]
            deferred_std_test = None
            transformed_train = None
            for split_name in ["train", "val", "test", "std_test"]:
                group = df[df["__split__"] == split_name]
                if len(group) == 0:
                    continue
                if split_name == "std_test" and not applies_to_std_test:
                    deferred_std_test = group.copy()
                    continue
                group = group.drop(columns="__split__")
                transformed = op.transform(group)
                if split_name == "train":
                    transformed_train = transformed.copy()
                transformed["__split__"] = split_name
                parts.append(transformed)
            if deferred_std_test is not None:
                if (
                    getattr(op, "FILTER_STD_TEST_TO_TRAIN_DOMAIN", False)
                    and transformed_train is not None
                ):
                    deferred_std_test = self._filter_std_test_to_train_domain(
                        deferred_std_test, transformed_train, op_name, op
                    )
                parts.append(deferred_std_test)
            new_df = pd.concat(parts, ignore_index=True)
            if (
                len(std_test_pre) > 0
                and (new_df["__split__"] == "std_test").sum() == 0
                and not getattr(op, "FILTER_STD_TEST_TO_TRAIN_DOMAIN", False)
            ):
                print(
                    f"  [std_test] op {op_name} would drop all "
                    f"{len(std_test_pre)} std_test rows; "
                    "preserving them untouched."
                )
                new_df = pd.concat([new_df, std_test_pre.copy()], ignore_index=True)
        else:
            new_df = op.transform(df)

        if hasattr(op, 'output_col_types') and op.output_col_types:
            self.register_col_types(op.output_col_types)
        if hasattr(op, 'drop_col_types') and op.drop_col_types:
            for col_name in op.drop_col_types:
                self.col_types.pop(col_name, None)
        return new_df

    def run_pre_process(self, yaml_path):
        if not self._has_std_test:
            self.load_std_test_frozen()
        with open(yaml_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        if self._relation_dag_spec(config) is None:
            raise ValueError(
                "Preprocessing YAML must use prev-only DAG schema: "
                "dag.sources, dag.ops[*].prev, and dag.train.prev"
            )
        self._run_pre_process_relation_dag(config)

    def split(self):
        """Return ``{"train": ..., "test": ...}``.

        ``test`` corresponds to the std-test holdout (each user's last
        interaction; produced by ``scripts/build_std_test.py``). Validation
        slicing is performed inside the downstream model's ``fit``/
        ``train_and_evaluate`` so baselines cannot tamper with the split via
        preprocessing operators.
        """
        df = self.interaction_df.reset_index(drop=True)

        if "__split__" in df.columns:
            test = df[df["__split__"] == "std_test"].drop(columns="__split__").reset_index(drop=True)
            train = df[df["__split__"] != "std_test"].drop(columns="__split__").reset_index(drop=True)
        else:
            test = df.iloc[0:0].copy()
            train = df

        # print(f"Data split: train_pool={len(train)}, test={len(test)}")
        return {"train": train, "test": test}


class TabularData(BaseData):
    OPERATOR_DIR = os.path.join(os.path.dirname(__file__), " operators")

    def __init__(self, name):
        super().__init__(name)
        self.data_type = "tabular data"
        self.train_df = None
        self.test_df = None
        self.target_col = None
        self.id_col = None
        self.auxiliary_dfs = {}
        self._has_std_test = False
        self._std_test_size = 0

    def _resolve_param(self, value):
        if isinstance(value, str) and value.startswith("$"):
            attr = value[1:]
            if attr in self.auxiliary_dfs:
                return self.auxiliary_dfs[attr]
            if not hasattr(self, attr):
                raise ValueError(f"Unknown reference '{value}' in pipeline params")
            return getattr(self, attr)
        if isinstance(value, dict):
            return {k: self._resolve_param(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._resolve_param(v) for v in value]
        return value

    def _load_op(self, op_name):
        pkg_name = "_dppbench_tab_ops"
        if pkg_name not in sys.modules:
            pkg_spec = importlib.machinery.ModuleSpec(
                pkg_name, loader=None, is_package=True
            )
            pkg_module = importlib.util.module_from_spec(pkg_spec)
            pkg_module.__path__ = [self.OPERATOR_DIR]
            sys.modules[pkg_name] = pkg_module

        module_name = _class_to_module(op_name)
        dotted_suffix, module_file = _resolve_op_path(self.OPERATOR_DIR, module_name)
        if module_file is None:
            raise ValueError(
                f"Operator '{op_name}' not found (expected module '{module_name}.py' "
                f"under any of {_OP_SUBDIRS})"
            )
        full_name = f"{pkg_name}.{dotted_suffix}"
        if full_name in sys.modules:
            return getattr(sys.modules[full_name], op_name)

        _ensure_base_op_loaded(pkg_name, self.OPERATOR_DIR)
        if "." in dotted_suffix:
            sub_pkg = dotted_suffix.rsplit(".", 1)[0]
            _ensure_subpkg_registered(pkg_name, sub_pkg, self.OPERATOR_DIR)

        spec = importlib.util.spec_from_file_location(
            full_name, module_file,
            submodule_search_locations=None,
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[full_name] = module
        spec.loader.exec_module(module)
        return getattr(module, op_name)

    def load_std_test_frozen(self):
        """Replace ``self.train_df`` with the frozen training subset and
        attach the held-out std-test rows (with their original labels)
        as additional rows tagged by ``__split__ == "std_test"``.

        Looks for files under ``<task>/std_test/``:
          * ``train_frozen.parquet`` — training rows after holdout. If
            present, replaces ``self.train_df``.
          * ``std_test.parquet``   — REQUIRED for std-test activation.
            Concatenated to ``self.train_df`` with ``__split__='std_test'``.

        Returns ``True`` iff std-test was activated.
        """
        std_dir = self._std_test_dir(getattr(self, "data_dir", None))
        if not std_dir or not os.path.isdir(std_dir):
            return False
        std_test_path = os.path.join(std_dir, "std_test.parquet")
        if not os.path.exists(std_test_path):
            return False

        frozen_path = os.path.join(std_dir, "train_frozen.parquet")
        if os.path.exists(frozen_path):
            self.train_df = pd.read_parquet(frozen_path)

        std_df = pd.read_parquet(std_test_path).copy()

        base = self.train_df.copy()
        if "__split__" not in base.columns:
            base["__split__"] = "train"
        std_df["__split__"] = "std_test"
        for c in std_df.columns:
            if c not in base.columns:
                base[c] = pd.NA
        for c in base.columns:
            if c not in std_df.columns:
                std_df[c] = pd.NA
        self.train_df = pd.concat(
            [base, std_df[base.columns]], ignore_index=True
        )
        self._has_std_test = True
        self._std_test_size = len(std_df)
        print(f"  [std_test] loaded {self._std_test_size} held-out rows")
        return True

    def _relation_dag_available_sources(self):
        sources = {}
        if self.train_df is not None:
            sources["main"] = self.train_df
            sources["train"] = self.train_df
        for name, df in self.auxiliary_dfs.items():
            if df is not None:
                sources[name] = df
        if self.test_df is not None:
            sources["test"] = self.test_df
        return sources

    def _run_pre_process_relation_dag(self, config):
        available_sources = self._relation_dag_available_sources()
        final_df = self._run_relation_dag(
            config, available_sources,
            lambda df, op, op_name: self._transform_tabular_dag_df(df, op, op_name),
        )
        self.train_df = final_df

    def _transform_tabular_dag_df(self, df, op, op_name):
        applies_to_std_test = getattr(op, "APPLIES_TO_STD_TEST", True)
        if (
            "__split__" in df.columns
            and (df["__split__"] == "std_test").any()
        ):
            if getattr(op, "FIT_ON_TRAIN_ONLY", False):
                parts = []
                for split_name in ["train", "val", "test", "std_test"]:
                    group = df[df["__split__"] == split_name]
                    if len(group) == 0:
                        continue
                    if split_name == "std_test" and not applies_to_std_test:
                        parts.append(group.copy())
                        continue
                    g_body = group.drop(columns="__split__")
                    trans = op.transform(g_body).copy()
                    trans["__split__"] = split_name
                    parts.append(trans)
                new_df = pd.concat(parts, ignore_index=True)
            elif applies_to_std_test:
                marker = df["__split__"].copy()
                body = df.drop(columns="__split__")
                new_body = op.transform(body)
                if len(new_body) == len(marker):
                    new_body = new_body.copy()
                    new_body["__split__"] = marker.values
                    new_df = new_body
                else:
                    print(
                        f"  [std_test] op {op_name} changed row count "
                        f"({len(body)} -> {len(new_body)}); falling back "
                        f"to splitwise transform to preserve std-test."
                    )
                    parts = []
                    for split_name in ["train", "val", "test", "std_test"]:
                        group = df[df["__split__"] == split_name]
                        if len(group) == 0:
                            continue
                        g_body = group.drop(columns="__split__")
                        trans = op.transform(g_body).copy()
                        trans["__split__"] = split_name
                        parts.append(trans)
                    new_df = pd.concat(parts, ignore_index=True)
            else:
                std_part = df[df["__split__"] == "std_test"].copy()
                other_part = df[df["__split__"] != "std_test"]
                other_body = other_part.drop(columns="__split__")
                new_other = op.transform(other_body).copy()
                if "__split__" not in new_other.columns:
                    new_other["__split__"] = "train"
                new_df = pd.concat([new_other, std_part], ignore_index=True)
        else:
            new_df = op.transform(df)
        return new_df

    def run_pre_process(self, yaml_path):
        if not self._has_std_test:
            self.load_std_test_frozen()
        with open(yaml_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        if self._relation_dag_spec(config) is None:
            raise ValueError(
                "Preprocessing YAML must use prev-only DAG schema: "
                "dag.sources, dag.ops[*].prev, and dag.train.prev"
            )
        self._run_pre_process_relation_dag(config)

    def split(self, val_ratio=0.2, seed=42):
        from sklearn.model_selection import train_test_split
        df = self.train_df

        std_test = None
        if "__split__" in df.columns and (df["__split__"] == "std_test").any():
            std_test = df[df["__split__"] == "std_test"].drop(columns="__split__").reset_index(drop=True)
            df = df[df["__split__"] != "std_test"].drop(columns="__split__").reset_index(drop=True)

        train, val = train_test_split(
            df, test_size=val_ratio, random_state=seed,
            stratify=df[self.target_col] if self.target_col else None,
        )
        train = train.reset_index(drop=True)
        val = val.reset_index(drop=True)
        print(
            f"Split: train={len(train)}, val={len(val)}, "
            f"test={len(self.test_df) if self.test_df is not None else 0}"
            + (f", std_test={len(std_test)}" if std_test is not None else "")
        )
        out = {"train": train, "val": val, "test": self.test_df}
        if std_test is not None:
            out["std_test"] = std_test
        return out
