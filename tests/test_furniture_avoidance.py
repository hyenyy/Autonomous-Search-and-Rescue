import unittest
from unittest.mock import patch
import numpy as np
from sar.config import default_config
from sar.mapping import OccupancyGrid
from sar.state_machine import Mission
from sar.planning import LocalAvoider
from sar.exploration import FrontierExplorer


class FurnitureClassificationTests(unittest.TestCase):
    def setUp(self):
        self.cfg = default_config()
        self.grid = OccupancyGrid(self.cfg)
        self.mission = Mission(self.cfg, self.grid)

    def test_rejected_scattered_hits_do_not_trigger_person_escape(self):
        angles = np.array([-1.5, 0, 1.5])
        ranges = np.full(3, .3)
        dynamic = np.ones(3, dtype=bool)
        self.mission._update_person_est(1, (0, 0, 0), angles, ranges, dynamic)
        self.assertFalse(dynamic.any(), 'rejected rays must reach static mapping/avoidance')
        v, _, _ = self.mission.avoider.apply(0, 0, angles, ranges, 1, dynamic=dynamic)
        self.assertEqual(v, 0, 'stationary observation must not become a person escape')

    def test_stationary_furniture_is_released_and_can_be_mapped(self):
        angles = np.linspace(-.05, .05, 5)
        ranges = np.full(5, .35)
        for now in np.arange(0, 5, .1):
            dynamic = np.ones(5, dtype=bool)
            self.mission._update_person_est(float(now), (0, 0, 0), angles, ranges, dynamic)
            if now > 3.1:
                self.assertFalse(dynamic.any(), 'stationary ghost must not rearm next tick')
                self.assertIsNone(self.mission.planner.avoid_xy)
            self.grid.integrate_scan((0, 0, 0), angles, ranges, exclude=dynamic)
        self.assertGreater(self.grid.occupied_mask().sum(), 0)

    def test_moving_person_keeps_dynamic_protection(self):
        angles = np.linspace(-.05, .05, 5)
        for now in np.arange(0, 5, .1):
            ranges = np.full(5, 2 - now * .2)
            dynamic = np.ones(5, dtype=bool)
            self.mission._update_person_est(float(now), (0, 0, 0), angles, ranges, dynamic)
            self.assertTrue(dynamic.all())
        self.assertLess(self.mission.planner.avoid_vel[0], -.15)

    def test_person_track_cannot_reclassify_known_wall_as_moving(self):
        x, y = self.grid.grid_to_world(*self.grid.world_to_grid(.4, 0))
        ix, iy = self.grid.world_to_grid(x, y)
        self.grid.log[iy, ix] = 4
        self.mission.planner.avoid_xy = (x, y)
        mask = self.mission._classify_dynamic((0, 0, 0),
            np.array([np.arctan2(y, x)]), np.array([np.hypot(x, y)]))
        self.assertFalse(mask.any(), 'track proximity cannot override observed static occupancy')

    def test_person_track_retains_returns_in_previously_free_space(self):
        x, y = self.grid.grid_to_world(*self.grid.world_to_grid(.4, 0))
        ix, iy = self.grid.world_to_grid(x, y)
        self.grid.log[iy, ix] = -4
        self.mission.planner.avoid_xy = (x, y)
        mask = self.mission._classify_dynamic((0, 0, 0),
            np.array([np.arctan2(y, x)]), np.array([np.hypot(x, y)]))
        self.assertTrue(mask.all())


class FurnitureEscapeTests(unittest.TestCase):
    def setUp(self):
        self.avoider = LocalAvoider(default_config())
        self.angles = np.array([-.4, .4, np.pi])

    def test_nearest_leg_switch_does_not_flip_turn(self):
        turns = []
        for i in range(8):
            ranges = np.array([.17, .18, 2.]) if i % 2 else np.array([.18, .17, 2.])
            v, w, _ = self.avoider.apply(.15, .2, self.angles, ranges, i * .064)
            self.assertLessEqual(v, 0)
            turns.append(np.sign(w))
        self.assertEqual(len(set(turns)), 1)

    def test_threshold_noise_does_not_resume_forward_too_early(self):
        for i in range(10):
            d = .19 if i % 2 == 0 else .22
            v, _, _ = self.avoider.apply(.15, .2, self.angles, np.array([d, d, 2.]), i * .064)
            self.assertLessEqual(v, 0)

    def test_escape_is_bounded_and_rear_obstacle_stops_backup(self):
        self.avoider.apply(.15, .2, self.angles, np.array([.17, .18, 2.]), 0)
        v, _, _ = self.avoider.apply(.15, .2, self.angles, np.array([.17, .18, .18]), .1)
        self.assertEqual(v, 0)
        v, _, replan = self.avoider.apply(.15, .2, self.angles, np.array([.17, .18, 2.]), 2)
        self.assertEqual(v, 0)
        self.assertTrue(replan)

    def test_clear_path_releases_escape_and_in_place_scan_is_preserved(self):
        v, w, _ = self.avoider.apply(0, -.7, self.angles, np.array([.17, .18, 2.]), 0)
        self.assertEqual((v, w), (0, -.7))
        self.avoider.apply(.15, .2, self.angles, np.array([.17, .18, 2.]), .1)
        for t in [.4, .6, .8, 1.]:
            v, _, _ = self.avoider.apply(.15, .2, self.angles, np.full(3, 2.), t)
        self.assertGreater(v, 0)


class RevisitedFrontierTests(unittest.TestCase):
    def test_failed_point_does_not_discard_other_end_of_long_frontier(self):
        cfg = default_config()
        grid = OccupancyGrid(cfg)
        explorer = FrontierExplorer(cfg, grid)
        comp = [grid.world_to_grid(1, y) for y in np.arange(-1, 1.01, .05)]
        explorer.blacklist = [explorer._representative(comp)]
        with patch.object(explorer, '_clusters', return_value=[comp]):
            goal = explorer.update((0, 0, 0))
        self.assertIsNotNone(goal, 'one failed point is not an exhausted boundary')
        self.assertGreaterEqual(np.linalg.norm(np.array(goal) - explorer.blacklist[0]),
                                cfg.explore.blacklist_radius)

    def test_completed_boundary_is_not_selected_again_from_another_position(self):
        cfg = default_config()
        grid = OccupancyGrid(cfg)
        explorer = FrontierExplorer(cfg, grid)
        near = [grid.world_to_grid(.5, 0)] * 6
        fresh = [grid.world_to_grid(2, 0)] * 6
        old_goal = grid.grid_to_world(*near[0])
        explorer.current_target = old_goal
        with patch.object(explorer, '_clusters', return_value=[near, fresh]):
            goal = explorer.update((*old_goal, 0))
            self.assertEqual(goal, grid.grid_to_world(*fresh[0]))
            explorer.current_target = None
            goal = explorer.update((0, 0, 0))
            self.assertEqual(goal, grid.grid_to_world(*fresh[0]))

    def test_frontier_under_robot_is_not_a_new_travel_goal(self):
        cfg = default_config()
        grid = OccupancyGrid(cfg)
        explorer = FrontierExplorer(cfg, grid)
        comp = [grid.world_to_grid(0, 0)] * 6
        with patch.object(explorer, '_clusters', return_value=[comp]):
            self.assertIsNone(explorer.update((0, 0, 0)))


class RecoveryFootprintTests(unittest.TestCase):
    def test_escape_can_return_through_gap_to_nearby_goal(self):
        from sim_offline import build_scenario
        cfg = default_config()
        mission = Mission(cfg, OccupancyGrid(cfg))
        world, _, _ = build_scenario('table', False)
        pose, goal = (-2.38, -1.55, 0), (-2.4, -2.4)
        angles = np.linspace(-np.pi, np.pi, 360, endpoint=False)
        ranges = world.raycast(*pose[:2], angles, cfg.lidar.max_range)
        heading = mission._best_gap_heading(pose, angles, ranges, goal_xy=goal)
        endpoint = np.array(pose[:2]) + .6 * np.array([np.cos(heading), np.sin(heading)])
        self.assertLess(np.linalg.norm(endpoint - goal), .4,
                        'clear space beyond the nearby return goal is unnecessary')

    def test_escape_heading_fits_robot_through_door_not_just_lidar_ray(self):
        from sim_offline import build_scenario
        cfg = default_config()
        mission = Mission(cfg, OccupancyGrid(cfg))
        world, _, _ = build_scenario('table', False)
        pose = (-2.342, -2.256, 2.516)
        angles = np.linspace(-np.pi, np.pi, 360, endpoint=False)
        ranges = world.raycast(*pose[:2], angles + pose[2], cfg.lidar.max_range)
        heading = mission._best_gap_heading(pose, angles, ranges, goal_xy=(0, 1))
        for distance in np.linspace(0, .65, 40):
            self.assertFalse(world.collides(
                pose[0] + distance * np.cos(heading),
                pose[1] + distance * np.sin(heading), cfg.robot.robot_radius),
                'an escape direction must fit the whole robot through the opening')

    def test_creep_can_advance_past_corner_outside_immediate_circular_footprint(self):
        cfg = default_config()
        mission = Mission(cfg, OccupancyGrid(cfg))
        mission._recover_phase = ('creep', 0)
        mission._creep_start = (0, 0)
        mission._creep_goal = (1, 0)
        mission._recover_after = Mission.EXPLORE
        # Corner at (0.15, 0.12) leaves ~0.115m forward travel before
        # touching the circular footprint (radius including margin = 0.125m).
        v, _ = mission._do_recover(.5, (0, 0, 0),
                                  np.array([0, np.arctan2(.12, .15), 3.14]),
                                  np.array([2, np.hypot(.15, .12), 2]))
        self.assertGreater(v, 0)

    def test_creep_stops_for_corner_inside_robot_width_outside_front_sector(self):
        cfg = default_config()
        mission = Mission(cfg, OccupancyGrid(cfg))
        mission._recover_phase = ('creep', 0)
        mission._creep_start = (0, 0)
        mission._creep_goal = (1, 0)
        mission._recover_after = Mission.EXPLORE
        v, _, = mission._do_recover(.5, (0, 0, 0),
                                    np.array([0, .8, -1.5, 3.14]),
                                    np.array([2, .14, 2, 2]))
        self.assertEqual(v, 0, 'diagonal table leg lies inside the swept robot footprint')
