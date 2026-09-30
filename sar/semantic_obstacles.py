"""카메라 의미 인식과 2D LiDAR를 결합한 정적 장애물 보강."""
import math

import numpy as np

from .geometry import wrap_angle


class TableObstacleMapper:
    """YOLO 테이블 박스 안의 가까운 LiDAR 다리들을 연결한다.

    2D LiDAR는 상판을 보지 못해 다리 사이를 free로 칠한다. 카메라가
    그 점들이 같은 테이블 소속임을 알려주면, 가장 바깥쪽의 가까운 다리
    두 개 사이만 가상 선분으로 막는다. 카메라 광선 전체나 테이블 뒤쪽을
    막지 않으므로 A*는 선분 끝을 돌아 반대편 공간으로 갈 수 있다.
    """

    def __init__(self, cfg, grid):
        self.cfg = cfg
        self.grid = grid
        self._tracks = {}

    def _camera_bearing(self, x_px, image_width):
        f_px = (image_width / 2) / math.tan(self.cfg.camera.hfov / 2)
        return math.atan2(image_width / 2 - x_px, f_px) \
            + self.cfg.camera.mount_yaw

    def _leg_centers(self, angles, ranges, center, half_width):
        tcfg = self.cfg.table
        a = np.asarray(angles, dtype=float)
        r = np.asarray(ranges, dtype=float)
        rel = wrap_angle(a - center)
        inside = np.abs(rel) <= half_width * 1.08 + 0.01
        valid = inside & np.isfinite(r) \
            & (r > self.cfg.lidar.min_range) & (r < tcfg.max_lidar_range)
        if not valid.any():
            return []

        # 배경 벽을 버리고 카메라 박스 안의 가장 가까운 물체 띠만 남긴다.
        r_min = float(r[valid].min())
        near = valid & (r <= r_min + tcfg.leg_depth_band)
        ids = np.flatnonzero(near)
        if ids.size < 2:
            return []
        order = ids[np.argsort(rel[ids])]

        all_rel = np.sort(rel[np.flatnonzero(inside)])
        diffs = np.abs(np.diff(all_rel))
        scan_step = float(np.median(diffs[diffs > 1e-6])) \
            if np.any(diffs > 1e-6) else math.radians(1.0)
        split_gap = max(math.radians(2.5), scan_step * 2.5)

        clusters = [[int(order[0])]]
        for idx in order[1:]:
            idx = int(idx)
            prev = clusters[-1][-1]
            if abs(float(rel[idx] - rel[prev])) > split_gap:
                clusters.append([idx])
            else:
                clusters[-1].append(idx)

        centers = []
        for cluster in clusters:
            cc = np.asarray(cluster, dtype=int)
            centers.append((float(np.median(a[cc])),
                            float(np.median(r[cc]))))
        return centers

    def update(self, now, pose, angles, ranges, image_width, table_boxes):
        """새로 추가한 가상 점유 셀 수를 반환한다."""
        tcfg = self.cfg.table
        if not tcfg.enabled or not table_boxes or image_width <= 0:
            self._expire(now)
            return 0

        total_added = 0
        for _name, conf, (x0, _y0, x1, _y1) in table_boxes:
            if conf < tcfg.yolo_conf \
                    or (x1 - x0) / image_width < tcfg.min_box_width_ratio:
                continue
            b_left = self._camera_bearing(x0, image_width)
            b_right = self._camera_bearing(x1, image_width)
            span = abs(float(wrap_angle(b_left - b_right)))
            if span <= 0.02:
                continue
            center = float(wrap_angle(b_right + span / 2))
            legs = self._leg_centers(angles, ranges, center, span / 2)
            if len(legs) < 2:
                continue

            points = []
            for bearing, distance in legs:
                world_angle = pose[2] + bearing
                points.append((pose[0] + distance * math.cos(world_angle),
                               pose[1] + distance * math.sin(world_angle)))
            # 화면에서 가장 떨어진 두 다리를 테이블의 보이는 가장자리로 사용.
            best = None
            best_gap = -1.0
            for i in range(len(points)):
                for j in range(i + 1, len(points)):
                    gap = math.hypot(points[i][0] - points[j][0],
                                     points[i][1] - points[j][1])
                    if gap > best_gap:
                        best, best_gap = (points[i], points[j]), gap
            if best is None or not (tcfg.min_leg_gap <= best_gap
                                    <= tcfg.max_leg_gap):
                continue

            p0, p1 = best
            mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
            q = tcfg.track_resolution
            key = (int(round(mx / q)), int(round(my / q)))
            old = self._tracks.get(key)
            count = 1
            marked = False
            if old is not None and now - old["last"] <= tcfg.track_timeout:
                count = old["count"] + 1
                marked = old["marked"]
            self._tracks[key] = {
                "count": count, "last": now, "p0": p0, "p1": p1,
                "marked": marked,
            }
            if count >= tcfg.confirm_frames and not marked:
                added = self.grid.mark_virtual_segment(
                    p0, p1, tcfg.barrier_thickness)
                total_added += added
                self._tracks[key]["marked"] = True
                if added:
                    print(f"[table] 테이블 다리 연결 가상장애물 추가 "
                          f"gap={best_gap:.2f}m cells={added}")
        self._expire(now)
        return total_added

    def _expire(self, now):
        timeout = self.cfg.table.track_timeout * 3.0
        self._tracks = {
            key: value for key, value in self._tracks.items()
            if value["marked"] or now - value["last"] <= timeout
        }
