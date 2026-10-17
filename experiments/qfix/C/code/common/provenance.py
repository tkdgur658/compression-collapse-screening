"""Run provenance: code version, source hashes, environment, config hash."""
import hashlib
import json
import os
import platform
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def source_digest(root=ROOT):
    """sha256 over every tracked source file (.py/.json/.md under common/, models/, main.py)."""
    h = hashlib.sha256()
    files = []
    for sub in ('common', 'models', 'tools', 'configs'):
        for dp, dn, fn in os.walk(os.path.join(root, sub)):
            dn[:] = sorted(d for d in dn if d != '__pycache__')
            files += [os.path.join(dp, f) for f in sorted(fn) if f.endswith(('.py', '.json', '.yaml'))]
    files.append(os.path.join(root, 'main.py'))
    for f in files:
        h.update(os.path.relpath(f, root).replace(os.sep, '/').encode())
        with open(f, 'rb') as fh:
            h.update(hashlib.sha256(fh.read()).digest())
    return h.hexdigest()


def deployed_commit(root=ROOT):
    p = os.path.join(root, 'DEPLOYED_COMMIT')
    return open(p).read().strip() if os.path.exists(p) else 'unknown'


def config_hash(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]


def environment():
    return {
        'python': sys.version.split()[0], 'torch': torch.__version__,
        'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES', ''),
        'host': platform.node(),
    }
