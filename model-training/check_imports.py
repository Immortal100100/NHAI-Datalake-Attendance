import sys
sys.path.insert(0, 'src')

import numpy, torch, torch.nn, torch.optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
import cv2
import albumentations as A
import onnxruntime
import pandas

import importlib.util
spec = importlib.util.spec_from_file_location('finetune', 'src/finetune.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

print('[OK] All imports successful')
print(f'Device: {mod.device}')
print(f'ArcFace margin={mod.CFG["margin"]}, scale={mod.CFG["scale"]}')
head = mod.FineTuneHead(235)
n = sum(p.numel() for p in head.parameters())
print(f'Head params: {n/1e3:.1f} K  (was 78K before)')
proj = mod.ProjectionHead()
print(f'Projection: Linear(512->256->128) with PReLU + Dropout')
