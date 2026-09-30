"""Breadcrumb 기록/복귀 — A* 실패 시의 안전망.

지나온 포즈를 일정 간격으로 기록해 두고, 복귀 시 역순으로 따라간다.
지나온 길 = 통과 가능이 증명된 경로라는 것이 장점. A* 지름길 복귀가
실패하거나 지도가 엉망일 때 이것으로 살아 돌아온다.
"""
from .geometry import dist


class Breadcrumbs:
    def __init__(self, cfg):
        self.spacing = cfg.mission.crumb_spacing
        self.crumbs = []              # [(x, y), ...] 시작점부터 순서대로

    def record(self, pose):
        p = (pose[0], pose[1])
        if not self.crumbs or dist(self.crumbs[-1], p) >= self.spacing:
            self.crumbs.append(p)

    def return_path(self, pose):
        """현재 위치에서 시작점까지의 waypoint 목록 (역순 crumbs)."""
        if not self.crumbs:
            return []
        # 현재 위치에서 가장 가까운 crumb부터 거꾸로
        nearest = min(range(len(self.crumbs)),
                      key=lambda i: dist(self.crumbs[i], pose))
        path = self.crumbs[: nearest + 1]
        return list(reversed(path))
