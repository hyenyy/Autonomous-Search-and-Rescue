import math
import unittest
from unittest.mock import patch
import numpy as np
from sar.config import default_config
from sar.mapping import OccupancyGrid
from sar.odometry import PoseEstimator
from sar.detection import Detection
from sar.state_machine import Mission
from sar.exploration import FrontierExplorer


class LoopCauseTests(unittest.TestCase):
    def setUp(self):
        self.cfg = default_config()
        self.cfg.explore.spin_period = 0
        self.mission = Mission(self.cfg, OccupancyGrid(self.cfg))
        self.mission.state = Mission.EXPLORE
        self.mission.start_time = 0
        self.mission.detector.consecutive = 4
        self.mission.detector.confirmed = True
        self.angles = np.linspace(-math.pi, math.pi, 360, endpoint=False)
        self.ranges = np.full(360, 3.4)

    def test_height_rejected_red_patch_cannot_preempt_exploration(self):
        self.mission.detector.last = Detection(0, 300, .5, (360, 380), 20,
            .04, False, bbox=(310,360,330,380), aspect=1, fill=.8)
        with patch.object(self.mission.detector, 'process'), patch.object(self.mission, '_do_explore', return_value=(.1, 0)):
            self.mission.step(1, (0,0,0), self.angles, self.ranges, None)
        self.assertEqual(self.mission.state, Mission.EXPLORE)

    def test_geometrically_valid_distant_apple_can_still_be_approached(self):
        self.mission.detector.last = Detection(0, 200, .5, (240,260), 20,
            2*math.atan(.05/3), False, bbox=(310,240,330,260), aspect=1, fill=.8)
        with patch.object(self.mission.detector, 'process'):
            self.mission.step(1, (0,0,0), self.angles, self.ranges, None)
        self.assertEqual(self.mission.state, Mission.SEEK)

    def test_map_matching_cannot_invent_translation_during_commanded_spin(self):
        self.cfg.localization.enabled = True
        estimator = PoseEstimator(self.cfg, (1,2,0))
        with patch.object(estimator.matcher, 'match', return_value=(1,2.125,0)) as matcher:
            pose = estimator.correct_with_scan(self.mission.grid, self.angles, self.ranges, commanded_v=0)
        self.assertEqual(pose, (1,2,0))
        matcher.assert_not_called()

    def test_repeatedly_visited_region_loses_priority_to_new_boundary(self):
        grid = self.mission.grid
        explorer = FrontierExplorer(self.cfg, grid)
        for _ in range(4):
            explorer.observe_pose((1.1,.1))
            explorer.observe_pose((2.1,.1))
        old = [grid.world_to_grid(1.1,.1)] * 6
        new = [grid.world_to_grid(3.1,.1)] * 6
        with patch.object(explorer, '_clusters', return_value=[old,new]):
            self.assertEqual(explorer.update((0,0)), grid.grid_to_world(*new[0]))
