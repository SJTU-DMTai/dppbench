import os
import sys
import importlib
import importlib.util
import importlib.machinery
import inspect
import re

OPERATOR_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "dppbench", " operators")


def _class_to_module(class_name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', class_name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def _load_all_operators():
    ops = {}
    pkg_name = "_op_registry_pkg"
    if pkg_name not in sys.modules:
        pkg_spec = importlib.machinery.ModuleSpec(pkg_name, loader=None, is_package=True)
        pkg_module = importlib.util.module_from_spec(pkg_spec)
        pkg_module.__path__ = [OPERATOR_DIR]
        sys.modules[pkg_name] = pkg_module

    base_file = os.path.join(OPERATOR_DIR, "base_op.py")
    base_full_name = f"{pkg_name}.base_op"
    if base_full_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(base_full_name, base_file, submodule_search_locations=None)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[base_full_name] = mod
        spec.loader.exec_module(mod)

    SUBDIRS = ("integration", "cleaning", "preprocessing",
               "feature_engineering", "transformation")

    for sub in SUBDIRS:
        sub_full = f"{pkg_name}.{sub}"
        if sub_full not in sys.modules:
            sub_spec = importlib.machinery.ModuleSpec(sub_full, loader=None, is_package=True)
            sub_mod = importlib.util.module_from_spec(sub_spec)
            sub_mod.__path__ = [os.path.join(OPERATOR_DIR, sub)]
            sys.modules[sub_full] = sub_mod

    for sub in SUBDIRS:
        sub_dir = os.path.join(OPERATOR_DIR, sub)
        if not os.path.isdir(sub_dir):
            continue
        for fname in sorted(os.listdir(sub_dir)):
            if not fname.endswith(".py") or fname.startswith("_"):
                continue
            module_name = fname[:-3]
            filepath = os.path.join(sub_dir, fname)
            full_name = f"{pkg_name}.{sub}.{module_name}"
            if full_name in sys.modules:
                module = sys.modules[full_name]
            else:
                spec = importlib.util.spec_from_file_location(full_name, filepath, submodule_search_locations=None)
                module = importlib.util.module_from_spec(spec)
                sys.modules[full_name] = module
                try:
                    spec.loader.exec_module(module)
                except Exception:
                    del sys.modules[full_name]
                    continue
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (inspect.isclass(attr) and
                        hasattr(attr, 'transform') and
                        attr_name not in ("BaseOp", "TabularOp", "RecOp", "TextOp", "ImageOp")):
                    ops[attr_name] = attr
    return ops


def get_operator_descriptions():
    ops = _load_all_operators()
    descriptions = {}
    for name, cls in ops.items():
        try:
            sig = inspect.signature(cls.__init__)
            params = [p for p in sig.parameters if p != "self"]
            param_str = ", ".join(params)
        except (ValueError, TypeError):
            param_str = ""

        if hasattr(cls, 'get_op_description'):
            try:
                instance = object.__new__(cls)
                desc = instance.get_op_description()
                if desc:
                    descriptions[name] = desc.strip()
                    continue
            except Exception:
                pass

        op_type = getattr(cls, 'op_type', 'unknown') if hasattr(cls, 'op_type') else 'unknown'
        descriptions[name] = f"Operator: {name}({param_str})\nType: {op_type}"

    return descriptions


def get_operator_summary():
    descriptions = get_operator_descriptions()
    lines = []
    for name, desc in sorted(descriptions.items()):
        short_desc = desc.split('\n')[0] if '\n' in desc else desc
        if len(short_desc) > 200:
            short_desc = short_desc[:200] + "..."
        lines.append(f"- {name}: {short_desc}")
    return "\n".join(lines)


def get_operator_detail(op_name):
    descriptions = get_operator_descriptions()
    return descriptions.get(op_name, f"Operator '{op_name}' not found.")
