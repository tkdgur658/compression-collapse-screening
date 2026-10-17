"""Adapter for MHorUNet. src/MHorUNet.py is a verbatim copy of the legacy file (see manifest.json)."""
from models._vendored import load_class

NAME = 'mhorunet'
DEFAULT_CFG = {'ctor': {'num_classes': 1, 'input_channels': 3}}
NATIVE_SIDE_OUTPUTS = False


def build(cfg):
    return load_class(__file__, 'MHorUNet')(**cfg['ctor'])


def forward_train(model, img):
    o = model(img)
    if NATIVE_SIDE_OUTPUTS:
        side, out = o
        return {'out': out, 'side': tuple(side)}
    return {'out': o, 'side': None}


def extra_loss(o, mask, cfg):
    return 0.0


def infer(model, img):
    o = model(img)
    return o[1] if NATIVE_SIDE_OUTPUTS else o


def is_train_only(param_name):
    return False
