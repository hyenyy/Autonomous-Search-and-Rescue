import unittest
from unittest.mock import patch
import numpy as np
from sar.config import default_config
from sar.detection import TargetDetector, Detection


class MissedAppleTests(unittest.TestCase):
    def test_large_red_background_does_not_hide_separate_floor_apple(self):
        cfg = default_config()
        cfg.camera.width, cfg.camera.height = 320, 240
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        frame[5:95, 10:130, 0] = 255  # large red sign, invalid vertical position
        y, x = np.ogrid[:240, :320]
        frame[(x - 240)**2 + (y - 130)**2 <= 9**2] = [255, 0, 0]
        det = TargetDetector(cfg).process(frame)
        self.assertIsNotNone(det)
        self.assertGreater(det.bbox[0], 220)

    def test_large_furniture_box_behind_apple_cannot_veto_it(self):
        cfg = default_config()
        cfg.detection.use_yolo = True
        detector = TargetDetector(cfg)
        blob = Detection(0, 300, .5, (240, 260), 20, .04, False,
                         bbox=(310, 240, 330, 260))
        with patch.object(detector, '_predict', return_value=[('couch', .85, [0, 0, 600, 400])]):
            self.assertTrue(detector.yolo_confirm(np.zeros((480,640,3), np.uint8), blob))
