"""거리 기반 배터리 추정과 안전 귀환 판단.

실제 배터리 API가 없는 Webots 월드에서도 이동 거리와 회전량으로 잔량을
추정한다. 귀환 판단은 고정 임계값이 아니라, 지금까지 지나온 breadcrumb를
따라 집까지 가는 데 필요한 에너지와 안전 여유분을 비교한다.
"""
import math

from .geometry import dist, wrap_angle


class EnergyManager:
    def __init__(self, cfg):
        self.cfg = cfg.energy
        self.percent = max(0.0, min(100.0, self.cfg.initial_percent))
        self._last_pose = None
        self.travelled = 0.0
        self.turned = 0.0

    def update(self, pose):
        """새 pose로 이동·회전 소비량을 누적하고 현재 잔량을 반환."""
        if self._last_pose is None:
            self._last_pose = tuple(pose)
            return self.percent
        ds = dist(self._last_pose, pose)
        dtheta = abs(wrap_angle(pose[2] - self._last_pose[2]))
        self.travelled += ds
        self.turned += dtheta
        used = (ds * self.cfg.percent_per_meter
                + dtheta * self.cfg.percent_per_radian)
        self.percent = max(0.0, self.percent - used)
        self._last_pose = tuple(pose)
        return self.percent

    @staticmethod
    def _path_length(pose, path):
        if not path:
            return 0.0
        total = dist(pose, path[0])
        total += sum(dist(a, b) for a, b in zip(path, path[1:]))
        return total

    def return_distance(self, pose, breadcrumbs):
        """현재 위치에서 breadcrumb 역추적 시 예상되는 귀환 거리."""
        return self._path_length(pose, breadcrumbs.return_path(pose))

    def required_return_percent(self, pose, breadcrumbs):
        distance = self.return_distance(pose, breadcrumbs)
        return (distance * self.cfg.percent_per_meter
                * self.cfg.return_multiplier)

    def should_return(self, pose, breadcrumbs):
        """귀환 예상 소비량과 안전 여유를 남길 수 없으면 True."""
        if not self.cfg.enabled:
            return False
        required = self.required_return_percent(pose, breadcrumbs)
        return self.percent <= required + self.cfg.reserve_percent
