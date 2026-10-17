"""Import a verbatim legacy model file in isolation.

Some legacy files run `from utils import *` (unused names) or import `thop` at module level.
A temporary empty `utils` stub keeps the legacy repository off the import path."""
import importlib.util
import os
import sys
import types


def load_class(adapter_file, cls_name):
    path = os.path.join(os.path.dirname(os.path.abspath(adapter_file)), 'src', cls_name + '.py')
    mod_name = f'_vendored_{cls_name}'
    if mod_name in sys.modules:
        return getattr(sys.modules[mod_name], cls_name)
    saved = sys.modules.get('utils')
    sys.modules['utils'] = types.ModuleType('utils')
    try:
        spec = importlib.util.spec_from_file_location(mod_name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)
    finally:
        if saved is None:
            sys.modules.pop('utils', None)
        else:
            sys.modules['utils'] = saved
    return getattr(mod, cls_name)
