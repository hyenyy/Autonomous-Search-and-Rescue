import math
import unittest
import numpy as np
from sar.config import default_config
from sar.mapping import OccupancyGrid
from sar.localization import ScanMatcher


class ScanMatchingTests(unittest.TestCase):
    def setUp(self):
        self.cfg = default_config()
        self.cfg.map.half_size = 4
        self.grid = OccupancyGrid(self.cfg)
        self.angles = np.linspace(-math.pi, math.pi, 360, endpoint=False)
        self.ranges = 2 / np.maximum(np.abs(np.cos(self.angles)), np.abs(np.sin(self.angles)))
        for _ in range(5):
            self.grid.integrate_scan((0, 0, 0), self.angles, self.ranges)

    def test_encoder_slip_is_corrected_against_static_room(self):
        matcher = ScanMatcher(self.cfg)
        pose = matcher.match(self.grid, (.10, -.075, 0), self.angles, self.ranges)
        self.assertLess(math.hypot(pose[0], pose[1]), .055)
        self.assertEqual(pose[2], 0)

    def test_dynamic_returns_do_not_pull_pose_toward_person(self):
        ranges = self.ranges.copy()
        ranges[:50] = .8
        matcher = ScanMatcher(self.cfg)
        pose = matcher.match(self.grid, (.10, -.075, 0), self.angles, ranges)
        self.assertLess(math.hypot(pose[0], pose[1]), .06)

    def test_empty_map_and_invalid_scan_leave_odometry_unchanged(self):
        matcher = ScanMatcher(self.cfg)
        prior = (.10, -.075, 0)
        self.assertEqual(matcher.match(OccupancyGrid(self.cfg), prior, self.angles, self.ranges), prior)
        self.assertEqual(matcher.match(self.grid, prior, self.angles, np.full(360, np.nan)), prior)

    def test_no_overlap_cannot_teleport_robot(self):
        matcher = ScanMatcher(self.cfg)
        prior = (1.0, 1.0, 0)
        self.assertEqual(matcher.match(self.grid, prior, self.angles, self.ranges), prior)


if __name__ == '__main__':
    unittest.main()
