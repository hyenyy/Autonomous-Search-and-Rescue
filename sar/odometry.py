"""바퀴 인코더 기반 차동구동 odometry.

시작 포즈는 대회에서 제공되므로(config.mission.start_*) 절대좌표의 기준이 된다.
짧은 주행(3시간 대회의 수 분짜리 미션)에서는 시뮬레이션 특성상 미끄러짐이
거의 없어 odometry 단독으로도 복귀 정밀도를 확보할 수 있다는 전제.
드리프트가 크면 scan matching을 붙일 자리는 PoseEstimator 하나로 격리돼 있다.
"""
import math

from .geometry import wrap_angle


def compass_values_to_raw(values):
    """Webots Compass values → '기울기 +1' 원시 헤딩각.

    Webots 컴퍼스는 월드 북쪽 벡터를 **센서(로봇) 프레임에서 본 값**을
    반환한다. 로봇이 CCW로 돌면 atan2(v[1], v[0]) = phi_N - theta 로
    오히려 감소한다(기울기 -1). 가산 오프셋 캘리브레이션은 부호 반전을
    보정할 수 없으므로, 여기서 부호를 뒤집어 theta - phi_N (기울기 +1)
    규약으로 통일한다. (멀티에이전트 리뷰에서 발견된 치명 결함 수정)
    """
    return -math.atan2(values[1], values[0])


class DiffDriveOdometry:
    def __init__(self, cfg, start_pose=(0.0, 0.0, 0.0)):
        self.r = cfg.robot.wheel_radius
        self.L = cfg.robot.wheel_base
        self.x, self.y, self.theta = start_pose
        self._last_left = None
        self._last_right = None

    @property
    def pose(self):
        return (self.x, self.y, self.theta)

    def update(self, left_pos, right_pos, heading=None):
        """left/right_pos: 바퀴 누적 회전각(rad). 매 스텝 호출.

        heading이 주어지면(rad, 이미 월드 프레임으로 보정된 절대 헤딩)
        회전은 그것을 신뢰하고 인코더는 이동량(ds)만 담당한다 —
        헤딩 드리프트(odometry 오차의 주범)가 제거된다.
        """
        if self._last_left is None:
            self._last_left, self._last_right = left_pos, right_pos
            if heading is not None:
                self.theta = heading
            return self.pose
        dl = (left_pos - self._last_left) * self.r
        dr = (right_pos - self._last_right) * self.r
        self._last_left, self._last_right = left_pos, right_pos

        ds = (dl + dr) / 2.0
        if heading is None:
            dtheta = (dr - dl) / self.L
            mid = self.theta + dtheta / 2.0     # 중점 적분(2차)
            new_theta = wrap_angle(self.theta + dtheta)
        else:
            dtheta = wrap_angle(heading - self.theta)
            mid = self.theta + dtheta / 2.0
            new_theta = heading
        self.x += ds * math.cos(mid)
        self.y += ds * math.sin(mid)
        self.theta = new_theta
        return self.pose


class PoseEstimator:
    """위치 추정 단일 창구.

    - 인코더 odometry 기본
    - compass 원시각(convention 무관)이 들어오면 시작 헤딩(대회 제공값)으로
      오프셋을 자동 캘리브레이션해 절대 헤딩으로 사용
    당일 드리프트가 더 문제면 여기에만 scan-matching을 추가한다.
    (다른 모듈은 전부 estimator.pose 만 바라본다.)
    """

    def __init__(self, cfg, start_pose):
        self.odom = DiffDriveOdometry(cfg, start_pose)
        self._start_theta = start_pose[2]
        self._compass_offset = None

    @property
    def pose(self):
        return self.odom.pose

    def update(self, left_pos, right_pos, compass_raw=None):
        """compass_raw: 컴퍼스에서 얻은 원시 각(rad, 임의 기준)."""
        heading = None
        if compass_raw is not None:
            if self._compass_offset is None:
                self._compass_offset = wrap_angle(
                    self._start_theta - compass_raw)
            heading = wrap_angle(compass_raw + self._compass_offset)
        return self.odom.update(left_pos, right_pos, heading=heading)
