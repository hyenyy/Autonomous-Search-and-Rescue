import unittest
from sar.config import default_config
from sar.mapping import OccupancyGrid
from sar.state_machine import Mission
from sar.planning import Planner


class SteeringTests(unittest.TestCase):
    def test_new_map_cells_do_not_trigger_repeated_scan_at_same_spot(self):
        cfg = default_config()
        mission = Mission(cfg, OccupancyGrid(cfg))
        mission._last_theta = 0
        mission._do_spin(1, (0, 0, 0), [0], [3.4])
        mission.grid.log[:30, :30] = -2
        self.assertFalse(mission._spin_worthwhile(30, (.2, 0, 0)))
        self.assertFalse(mission._spin_worthwhile(3, (2, 0, 0)))
        self.assertTrue(mission._spin_worthwhile(30, (2, 0, 0)))

    def test_small_heading_noise_does_not_alternate_steering(self):
        cfg = default_config()
        planner = Planner(cfg, OccupancyGrid(cfg))
        planner.waypoints = [(2, 0)]
        for heading in [-.02, .02, -.01, .01]:
            v, w = planner.follow((0, 0, heading))
            self.assertGreater(v, 0)
            self.assertEqual(w, 0)
        self.assertLess(planner.follow((0, 0, .3))[1], 0)
