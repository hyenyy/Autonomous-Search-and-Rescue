"""Local scan-to-map translation correction; compass supplies heading.

This is local map matching, not global relocalization or loop closure.
"""
import numpy as np
import cv2


class ScanMatcher:
    def __init__(self, cfg):
        self.cfg = cfg
        self.corrections = 0
        self.last_correction = (0.0, 0.0)
        self._version = None
        self._field = None

    def match(self, grid, pose, angles, ranges):
        self.last_correction = (0.0, 0.0)
        angles, ranges = np.asarray(angles), np.asarray(ranges)
        valid = (np.isfinite(angles) & np.isfinite(ranges)
                 & (ranges > self.cfg.lidar.min_range)
                 & (ranges < self.cfg.lidar.max_range * .95))
        a, r = angles[valid][::2], ranges[valid][::2]
        if len(r) < 40:
            return pose
        if self._version != grid._version:
            occupied = grid.occupied_mask()
            if occupied.sum() < 30:
                return pose
            self._field = cv2.distanceTransform((~occupied).astype(np.uint8),
                                                cv2.DIST_L2, 5) * grid.res
            self._version = grid._version
        x = (pose[0] + r * np.cos(a + pose[2]) - grid.origin_x) / grid.res - .5
        y = (pose[1] + r * np.sin(a + pose[2]) - grid.origin_y) / grid.res - .5
        offsets = np.arange(-.125, .126, .025)
        dx, dy = np.meshgrid(offsets, offsets)
        shifts = np.column_stack((dx.ravel(), dy.ravel()))
        px = (x[None, :] + shifts[:, 0, None] / grid.res).astype(np.float32)
        py = (y[None, :] + shifts[:, 1, None] / grid.res).astype(np.float32)
        distances = cv2.remap(self._field, px, py, cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=1.0)
        # Trim new surfaces and moving objects; require broad static overlap.
        count = max(30, int(len(r) * .75))
        residual = np.sort(distances, axis=1)[:, :count].mean(axis=1)
        prior = int(np.argmin(np.linalg.norm(shifts, axis=1)))
        scores = residual + .035 * np.linalg.norm(shifts, axis=1)
        best = int(np.argmin(scores))
        if (residual[best] > .055 or residual[prior] - residual[best] < .006
                or (distances[best] < .10).mean() < .65):
            return pose
        delta = shifts[best]
        self.last_correction = tuple(float(v) for v in delta)
        self.corrections += 1
        return (pose[0] + float(delta[0]), pose[1] + float(delta[1]), pose[2])
