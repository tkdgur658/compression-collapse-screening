"""Model registry: name -> adapter module. Only this file and main.py list models."""
import importlib

MODELS = ['pbe', 'maresunet', 'utanet', 'mk_unet', 'mhorunet', 'tinyunet',
          'ucmnet', 'lbunet', 'cmunext', 'egeunet', 'malunet']


def get(name):
    if name not in MODELS:
        raise KeyError(f'unknown model {name}; registered: {MODELS}')
    return importlib.import_module(f'models.{name}.adapter')
