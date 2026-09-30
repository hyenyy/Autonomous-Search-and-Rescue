"""Atomic live artifacts and measured controller timing (wall clock vs sim time)."""
from collections import defaultdict
import io
import json
import os
from pathlib import Path
import time
import uuid

import numpy as np


def atomic_write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    try:
        tmp.write_bytes(data)
        os.replace(tmp, path)
        return True
    except PermissionError:
        # Windows reader may briefly lock the old file. Keep it intact and retry
        # on the next publish, rather than truncate what the browser is reading.
        return False
    finally:
        tmp.unlink(missing_ok=True)


def write_json(path, data):
    return atomic_write(path, json.dumps(data, ensure_ascii=False, allow_nan=False,
                                        indent=2).encode('utf-8'))


def save_grid(path, grid, pose):
    buf = io.BytesIO()
    np.savez_compressed(buf, log_odds=grid.log, resolution=grid.res,
                        origin=[grid.origin_x, grid.origin_y], pose=pose,
                        occupied_threshold=grid.occ_th, free_threshold=grid.free_th)
    return atomic_write(path, buf.getvalue())


class RuntimeStats:
    def __init__(self):
        self.run_id = uuid.uuid4().hex
        self.started = self.last_wall = time.perf_counter()
        self.last_sim = 0.0
        self.count = 0
        self.totals = defaultdict(float)

    def record(self, **seconds):
        self.count += 1
        for key, value in seconds.items():
            self.totals[key] += value

    def snapshot(self, sim_time):
        wall = time.perf_counter()
        result = {
            'run_id': self.run_id, 'updated_at': time.time(), 'sim_time': sim_time,
            'wall_time': wall - self.started,
            'real_time_factor': (sim_time - self.last_sim) / max(wall - self.last_wall, 1e-6),
            'stage_ms': {k: 1000 * v / max(self.count, 1) for k, v in self.totals.items()},
        }
        self.last_sim, self.last_wall = sim_time, wall
        self.count = 0
        self.totals.clear()
        return result
