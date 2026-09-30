"""Real model + project detector integration check, using recorded Webots frames."""
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sar.config import default_config
from sar.detection import TargetDetector

cfg = default_config().apply_overrides(ROOT / 'config_override.json')
detector = TargetDetector(cfg)
detector.yolo_warmup()
assert detector.yolo_device == 'cuda:0', detector.yolo_device
rgb = np.array(Image.open(ROOT / 'tests/fixtures/apple_as_ball.jpg').convert('RGB'))
latencies = []
for _ in range(12):
    boxes = detector.yolo_boxes(rgb.copy())
    latencies.append(detector.yolo_ms)
blob = detector.process(rgb)
assert blob is not None
assert detector.yolo_confirm(rgb, blob), 'recorded real apple was vetoed'
assert detector._yolo.predictor.device.type == 'cuda'
before = detector.yolo_ms
detector.yolo_boxes(rgb)
assert detector.yolo_ms == before, 'same-frame inference was repeated'
extinguisher = np.array(Image.open(ROOT / 'tests/fixtures/extinguisher.png').convert('RGB'))
assert detector.process(extinguisher) is None
result = dict(torch=torch.__version__, device=detector.yolo_device,
              gpu=torch.cuda.get_device_name(0), end_to_end_median_ms=statistics.median(latencies[2:]),
              apple_boxes=boxes, apple_accepted=True, extinguisher_rejected=True,
              same_frame_cached=True)
(ROOT / 'out/gpu_validation.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result, indent=2))
