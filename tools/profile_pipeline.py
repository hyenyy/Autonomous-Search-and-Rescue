"""Repeatable CPU pipeline benchmark; no Webots API required."""
import argparse
import cProfile
import io
import json
from pathlib import Path
import pstats
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sar.config import default_config
from sar.detection import TargetDetector
from sar.mapping import OccupancyGrid
from sar.state_machine import Mission


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', default='out/profile_pipeline.json')
    args = ap.parse_args()
    import cv2
    frame = cv2.imread(str(ROOT / 'out/live_cam.png'))
    if frame is None:
        raise RuntimeError('Capture out/live_cam.png before profiling')
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cfg = default_config()
    grid = OccupancyGrid(cfg)
    mission = Mission(cfg, grid)
    detector = TargetDetector(cfg)
    angles = np.linspace(-np.pi, np.pi, 360, endpoint=False)
    ranges = 2.0 / np.maximum(np.abs(np.cos(angles)), np.abs(np.sin(angles)))
    samples = []
    for _ in range(100):
        t = time.perf_counter()
        detector.process(frame)
        samples.append((time.perf_counter() - t) * 1000)
    profiler = cProfile.Profile()
    profiler.enable()
    for i in range(200):
        mission.step(i * .064, (0, 0, 0), angles, ranges, frame)
    profiler.disable()
    stream = io.StringIO()
    pstats.Stats(profiler, stream=stream).sort_stats('cumtime').print_stats(20)
    result = {'frame_shape': list(frame.shape), 'hsv_median_ms': float(np.median(samples)),
              'hsv_p95_ms': float(np.percentile(samples, 95)), 'profile': stream.getvalue()}
    path = ROOT / args.output
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
