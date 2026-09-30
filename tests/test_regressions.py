"""Behavior contracts found during the September pipeline audit."""
import math
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from sar.config import default_config
from sar.detection import Detection, TargetDetector, rgb_to_hsv
from sar.mapping import OccupancyGrid
from sar.state_machine import Mission
from sar.exploration import FrontierExplorer


class ExplorationTests(unittest.TestCase):
    def test_orbit_without_progress_releases_frontier(self):
        cfg = default_config()
        grid = OccupancyGrid(cfg)
        explorer = FrontierExplorer(cfg, grid)
        explorer.current_target = (3, 0)
        with patch.object(explorer, 'target_still_frontier', return_value=True), \
             patch.object(explorer, '_clusters', return_value=[]):
            for t, xy in [(0, (0, 0)), (5, (.2, .2)), (10, (0, .4)),
                          (20, (-.2, .2)), (30, (0, 0))]:
                explorer.update(xy, now=t)
        self.assertIsNone(explorer.current_target)
        self.assertIn((3, 0), explorer.blacklist)

    def test_forward_progress_keeps_frontier(self):
        cfg = default_config()
        explorer = FrontierExplorer(cfg, OccupancyGrid(cfg))
        explorer.current_target = (8, 0)
        with patch.object(explorer, 'target_still_frontier', return_value=True):
            for t in range(0, 61, 5):
                self.assertEqual(explorer.update((t * .05, 0), now=t), (8, 0))


class MappingTests(unittest.TestCase):
    def test_hit_just_outside_map_does_not_alias_border(self):
        cfg = default_config()
        cfg.map.half_size = 1
        grid = OccupancyGrid(cfg)
        # Endpoint x=-1.01 is outside; truncation used to mark cell zero occupied.
        for _ in range(5):
            grid.integrate_scan((-.91, .025, math.pi), [0], [.10])
        self.assertFalse(grid.occupied_mask().any())

    def test_no_return_clears_to_range_limit_without_creating_wall(self):
        cfg = default_config()
        grid = OccupancyGrid(cfg)
        for _ in range(4):
            grid.integrate_scan((0, 0, 0), [0], [float('inf')])
        ix, iy = grid.world_to_grid(2, 0)
        self.assertTrue(grid.free_mask()[iy, ix])
        self.assertFalse(grid.occupied_mask().any())

    def test_nan_and_negative_return_do_not_modify_map(self):
        grid = OccupancyGrid(default_config())
        grid.integrate_scan((0, 0, 0), [0, 1, 2], [np.nan, -1, -np.inf])
        self.assertEqual(np.count_nonzero(grid.log), 0)

    def test_excluded_dynamic_return_does_not_create_wall(self):
        grid = OccupancyGrid(default_config())
        for _ in range(4):
            grid.integrate_scan((0, 0, 0), [0], [1.0], exclude=[True])
        self.assertEqual(np.count_nonzero(grid.log), 0)


class DetectionTests(unittest.TestCase):
    def test_recorded_apple_passes_color_and_geometry(self):
        from PIL import Image
        frame = np.array(Image.open(Path(__file__).parent / 'fixtures/apple_as_ball.jpg').convert('RGB'))
        detector = TargetDetector(default_config())
        self.assertIsNotNone(detector.process(frame))

    def test_recorded_extinguisher_is_not_a_floor_apple(self):
        from PIL import Image
        frame = np.array(Image.open(Path(__file__).parent / 'fixtures/extinguisher.png').convert('RGB'))
        detector = TargetDetector(default_config())
        for _ in range(5):
            self.assertIsNone(detector.process(frame))

    def test_low_poly_apple_mislabeled_as_ball_is_not_vetoed(self):
        cfg = default_config()
        cfg.detection.use_yolo = True
        detector = TargetDetector(cfg)
        blob = Detection(0, 100, .5, (10, 20), 10, .1, False, bbox=(10, 10, 20, 20))
        with patch.object(detector, '_predict', return_value=[('sports ball', .73, [9, 9, 21, 21])]):
            self.assertTrue(detector.yolo_confirm(np.zeros((32, 32, 3), np.uint8), blob))
        with patch.object(detector, '_predict', return_value=[('bottle', .74, [9, 9, 21, 21])]):
            self.assertFalse(detector.yolo_confirm(np.zeros((32, 32, 3), np.uint8), blob))

    def test_color_units_and_thresholds(self):
        pixels = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255],
                            [255, 255, 0], [255, 0, 255], [0, 255, 255]]], np.uint8)
        hsv = rgb_to_hsv(pixels)
        np.testing.assert_allclose(hsv[0, :, 0], [0, 120, 240, 60, 300, 180], atol=.001)
        np.testing.assert_allclose(hsv[0, :, 1:], 1, atol=1e-6)

    def test_yolo_failure_cannot_silently_accept_target(self):
        cfg = default_config()
        cfg.detection.use_yolo = True
        detector = TargetDetector(cfg)
        detector._yolo_failed = True
        blob = Detection(0, 100, .5, (10, 20), 10, .1, False, bbox=(10, 10, 20, 20))
        with self.assertRaises(RuntimeError):
            detector.yolo_confirm(np.zeros((32, 32, 3), np.uint8), blob)


class MissionTests(unittest.TestCase):
    def test_recovery_rear_obstacle_stops_reverse_immediately(self):
        cfg = default_config()
        mission = Mission(cfg, OccupancyGrid(cfg))
        mission.state = Mission.RECOVER
        mission._recover_phase = ('backup', 0)
        mission._v_prev = -.08
        angles = np.linspace(-math.pi, math.pi, 360, endpoint=False)
        ranges = np.full(360, 3.4)
        ranges[np.abs(angles) > 2.9] = .14
        v, _, _ = mission.step(.5, (0, 0, 0), angles, ranges, None)
        self.assertEqual(v, 0, 'rear guard must not be overridden by velocity smoothing')

    def test_new_invalid_blob_cannot_confirm_old_estimate(self):
        from PIL import Image
        frame = np.array(Image.open(Path(__file__).parent / 'fixtures/wall_text.png').convert('RGB'))
        cfg = default_config()
        mission = Mission(cfg, OccupancyGrid(cfg))
        mission.target_est = (1, 0)
        mission._est_hist = [(1, 0)] * 3
        mission.detector.consecutive = 4
        mission.detector.confirmed = True
        _, _, info = mission.step(1, (0, 0, 0), [0], [3.4], frame)
        self.assertFalse(info['found'])
        self.assertIsNone(info['target_est'])

    def test_lost_confirmed_target_does_not_remain_forever(self):
        cfg = default_config()
        mission = Mission(cfg, OccupancyGrid(cfg))
        mission.start_time = 0
        mission.target_found = True
        mission.target_est = (2, 0)
        mission._enter(Mission.GOTO_TARGET, 0)
        mission._do_goto(30, (0, 0, 0), [0], [3.4])
        self.assertFalse(mission.target_found)
        self.assertEqual(mission.state, Mission.EXPLORE)

    def test_confirmed_target_does_not_jump_to_another_red_object(self):
        cfg = default_config()
        mission = Mission(cfg, OccupancyGrid(cfg))
        mission.target_found = True
        mission.target_est = (0, 2)
        mission._enter(Mission.GOTO_TARGET, 0)
        mission.detector.last = Detection(0, 100, .5, (50, 70), 20,
            2*math.atan(.05/2), False, bbox=None)
        mission._update_target_estimate((0, 0, 0), [0], [3.4])
        self.assertEqual(mission.target_est, (0, 2))

    def test_recovery_preserves_return_destination(self):
        cfg = default_config()
        mission = Mission(cfg, OccupancyGrid(cfg))
        mission._recover_after = Mission.RETURN
        mission.state = Mission.RECOVER
        mission._finish_recover(20, (2, 0, 0))
        self.assertEqual(mission.state, Mission.RETURN)

    def test_wall_text_cannot_be_localized_below_floor(self):
        from PIL import Image
        frame = np.array(Image.open(Path(__file__).parent / 'fixtures/wall_text.png').convert('RGB'))
        cfg = default_config()
        mission = Mission(cfg, OccupancyGrid(cfg))
        mission.detector.process(frame)
        angles = np.linspace(-math.pi, math.pi, 360, endpoint=False)
        mission._update_target_estimate((0, 0, 0), angles, np.full(360, 3.4))
        self.assertIsNone(mission.target_est, 'letter fragment projects beneath the floor')

    def test_second_visit_enters_return_and_goes_home(self):
        cfg = default_config()
        cfg.mission.num_targets = 2
        grid = OccupancyGrid(cfg)
        grid.log.fill(-3)
        grid._version += 1
        mission = Mission(cfg, grid)
        mission.visited_targets = [(1, 2)]
        mission.target_est = (2.4, 0)
        mission.target_found = True
        mission.state = Mission.GOTO_TARGET
        mission.start_time = 0
        mission.state_since = 0
        angles, ranges = np.linspace(-math.pi, math.pi, 360), np.full(360, 3.4)
        mission.step(1, (2, 0, math.pi), angles, ranges, None)
        self.assertEqual(len(mission.visited_targets), 2)
        self.assertEqual(mission.state, Mission.RETURN)
        v, _, info = mission.step(1.064, (2, 0, math.pi), angles, ranges, None)
        self.assertEqual(info['goal'], mission.start_xy)
        self.assertGreater(v, 0, 'return must drive toward home')
        v, w, info = mission.step(15, (.1, 0, math.pi), angles, ranges, None)
        self.assertTrue(info['success'])
        self.assertEqual((v, w), (0, 0))

    def test_done_stops_immediately_even_after_reverse_or_turn(self):
        cfg = default_config()
        mission = Mission(cfg, OccupancyGrid(cfg))
        mission.state = Mission.RETURN
        mission.visited_targets = [(2, 0)]
        mission.target_reached = True
        mission._v_prev, mission._w_prev = -.1, 1.6
        for now in [1, 1.064, 1.128]:
            v, w, info = mission.step(now, (.1, 0, 0), [0], [3.4], None)
            self.assertEqual((v, w), (0, 0), 'DONE must bypass acceleration smoothing')
            self.assertTrue(info['success'])

    def test_return_uses_breadcrumbs_if_global_plan_fails(self):
        cfg = default_config()
        mission = Mission(cfg, OccupancyGrid(cfg))
        mission.state = Mission.RETURN
        for x in [0, .5, 1, 1.5, 2]:
            mission.crumbs.record((x, 0, 0))
        with patch.object(mission.planner, 'plan_to', return_value=False):
            v, _ = mission._do_return(10, (2, 0, math.pi), [0], [3.4])
        self.assertTrue(mission._crumb_mode)
        self.assertEqual(mission.planner.waypoints[-1], (0, 0))
        self.assertGreater(v, 0)

    def test_home_without_targets_is_not_mission_success(self):
        cfg = default_config()
        cfg.mission.num_targets = 2
        mission = Mission(cfg, OccupancyGrid(cfg))
        mission.state = Mission.RETURN
        _, _, info = mission.step(1, (0, 0, 0), np.linspace(-math.pi, math.pi, 360),
                                  np.full(360, 3.4), None)
        self.assertEqual(info['state'], Mission.DONE)
        self.assertFalse(info.get('success', True))

    def test_offline_success_requires_every_target_visited(self):
        from sim_offline import is_success
        r = dict(done=True, collisions=0, min_dist_to_target=.3,
                 final_dist_to_start=.1, visited=1, num_targets=2)
        self.assertFalse(is_success(r))


if __name__ == '__main__':
    unittest.main()
