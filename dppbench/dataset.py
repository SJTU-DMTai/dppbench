import importlib.util
import importlib.machinery
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

    def _run_pipeline_step(self, step, step_idx, total_steps):
        op_name = step["op"]
        target = step.get("target", "interaction")
        raw_params = step.get("params", {}) or {}
        params = {k: self._resolve_param(v) for k, v in raw_params.items()}
        op_cls = self._load_op(op_name)
        op = op_cls(**params)
        applies_to_std_test = getattr(op, "APPLIES_TO_STD_TEST", True)
        uses_train_history = getattr(op, "USES_TRAIN_HISTORY_FOR_STD_TEST", False)
        attr = f"{target}_df"
        if not hasattr(self, attr) or getattr(self, attr) is None:
            raise ValueError(f"Target '{target}' has no DataFrame on {self.name}")
        df = getattr(self, attr)

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

        setattr(self, attr, new_df)

        if hasattr(op, 'output_col_types') and op.output_col_types:
            self.register_col_types(op.output_col_types)

        if hasattr(op, 'drop_col_types') and op.drop_col_types:
            for col_name in op.drop_col_types:
                self.col_types.pop(col_name, None)

        print(f"Step {step_idx}/{total_steps}: {op_name} on {target}_df, shape: {df.shape} -> {new_df.shape}")

    def run_pre_process(self, yaml_path):
        if not self._has_std_test:
            self.load_std_test_frozen()
        with open(yaml_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        steps = config.get("pipeline", [])

        pre_remap_targets = {"item", "user"}
        pre_steps = [s for s in steps if s.get("target", "interaction") in pre_remap_targets]
        post_steps = [s for s in steps if s.get("target", "interaction") not in pre_remap_targets]

        total = len(steps)
        step_no = 0
        for step in pre_steps:
            step_no += 1
            self._run_pipeline_step(step, step_no, total)

        self._remap_ids()

        for step in post_steps:
            step_no += 1
            self._run_pipeline_step(step, step_no, total)

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

    def run_pre_process(self, yaml_path):
        if not self._has_std_test:
            self.load_std_test_frozen()
        with open(yaml_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        steps = config.get("pipeline", [])
        for i, step in enumerate(steps, start=1):
            op_name = step["op"]
            target = step.get("target", "train")
            raw_params = step.get("params", {}) or {}
            params = {k: self._resolve_param(v) for k, v in raw_params.items()}
            op_cls = self._load_op(op_name)
            op = op_cls(**params)
            applies_to_std_test = getattr(op, "APPLIES_TO_STD_TEST", True)

            if target == "both":
                targets = ["train", "test"]
            else:
                targets = [target]

            for t in targets:
                if t == "train":
                    df = self.train_df
                elif t == "test":
                    df = self.test_df
                else:
                    df = self.auxiliary_dfs.get(t)

                if df is None:
                    continue

                # When operating on the train_df with std-test rows packed
                # in via __split__, run the op only on the train slice for
                # train-only operators; otherwise transform the whole df so
                # std-test rows are aligned with train post-transform.
                if (
                    t == "train"
                    and "__split__" in df.columns
                    and (df["__split__"] == "std_test").any()
                ):
                    if getattr(op, "FIT_ON_TRAIN_ONLY", False):
                        # Stateful encoders/statistical transforms must learn
                        # their mapping from training rows only, then reuse it
                        # for held-out rows to avoid std-test leakage.
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
                        # Drop the marker, transform together, re-mark.
                        marker = df["__split__"].copy()
                        body = df.drop(columns="__split__")
                        new_body = op.transform(body)
                        # If the op preserves index/length we can re-attach
                        # the marker; else fall back to "all rows are train".
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
                                trans = op.transform(g_body)
                                trans = trans.copy()
                                trans["__split__"] = split_name
                                parts.append(trans)
                            new_df = pd.concat(parts, ignore_index=True)
                    else:
                        # Train-only op: transform train rows only, keep
                        # std-test rows (and any val/test markers) intact.
                        std_part = df[df["__split__"] == "std_test"].copy()
                        other_part = df[df["__split__"] != "std_test"]
                        other_body = other_part.drop(columns="__split__")
                        new_other = op.transform(other_body).copy()
                        if "__split__" not in new_other.columns:
                            new_other["__split__"] = "train"
                        new_df = pd.concat(
                            [new_other, std_part], ignore_index=True
                        )
                else:
                    new_df = op.transform(df)

                if t == "train":
                    self.train_df = new_df
                elif t == "test":
                    self.test_df = new_df
                else:
                    self.auxiliary_dfs[t] = new_df

            shape_before = self.train_df.shape if target in ("train", "both") else (self.test_df.shape if self.test_df is not None else None)
            print(f"Step {i}/{len(steps)}: {op_name} on {target}, train shape: {self.train_df.shape}")

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
