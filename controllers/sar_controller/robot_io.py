"""Webots 디바이스 접근 계층 — Webots API는 전부 이 파일에만 있다.

당일 실제 로봇이 주어지면 이 파일과 config만 손보면 된다.
디바이스는 이름이 아니라 타입으로 자동 탐색하므로, 이름이 달라도
대부분 그대로 동작한다. (안 되면 NAME_HINTS만 수정)
"""
import math

import numpy as np
from controller import Robot  # Webots가 제공

from sar.geometry import wrap_angle
from sar.odometry import compass_values_to_raw

# 이름 힌트: 자동 탐색이 애매할 때 참고 (소문자 부분일치)
NAME_HINTS = {
    "left_motor": ("left wheel motor", "left motor", "motor_left", "left"),
    "right_motor": ("right wheel motor", "right motor", "motor_right", "right"),
}


class RobotIO:
    def __init__(self, cfg):
        self.cfg = cfg
        self.robot = Robot()
        self.timestep = int(self.robot.getBasicTimeStep())

        self.lidar = None
        self.camera = None
        self.compass = None
        self.left_motor = None
        self.right_motor = None
        self.left_sensor = None
        self.right_sensor = None
        self._discover()
        self._enable()
        self._lidar_angles = None
        self._autocalibrate()

    def _autocalibrate(self):
        """디바이스가 보고하는 실제 값으로 config를 갱신 — 수작업 최소화."""
        if self.camera is not None:
            self.cfg.camera.hfov = float(self.camera.getFov())
            self.cfg.camera.width = int(self.camera.getWidth())
            self.cfg.camera.height = int(self.camera.getHeight())
        if self.lidar is not None:
            try:
                mr = float(self.lidar.getMaxRange())
                if math.isfinite(mr) and mr > 0.5:
                    self.cfg.lidar.max_range = mr * 0.97
            except Exception:
                pass
        print(f"[robot_io] autocal: cam {self.cfg.camera.width}x"
              f"{self.cfg.camera.height} fov={self.cfg.camera.hfov:.3f}, "
              f"lidar max={self.cfg.lidar.max_range:.2f}, "
              f"compass={'yes' if self.compass else 'no'}")

    # ── 디바이스 탐색 ─────────────────────────────────────────
    def _discover(self):
        from controller import Camera, Compass, Lidar, Motor, PositionSensor
        motors, psensors = [], []
        for i in range(self.robot.getNumberOfDevices()):
            dev = self.robot.getDeviceByIndex(i)
            if isinstance(dev, Lidar) and self.lidar is None:
                self.lidar = dev
            elif isinstance(dev, Camera) and self.camera is None:
                self.camera = dev
            elif isinstance(dev, Compass) and self.compass is None:
                self.compass = dev
            elif isinstance(dev, Motor):
                motors.append(dev)
            elif isinstance(dev, PositionSensor):
                psensors.append(dev)

        def pick(devs, side):
            hints = NAME_HINTS[f"{side}_motor"]
            for h in hints:
                for d in devs:
                    if h in d.getName().lower():
                        return d
            return None

        self.left_motor = pick(motors, "left")
        self.right_motor = pick(motors, "right")
        if self.left_motor is None or self.right_motor is None:
            names = [m.getName() for m in motors]
            raise RuntimeError(f"바퀴 모터 자동 탐색 실패. 모터 목록: {names} "
                               f"→ robot_io.NAME_HINTS 수정 필요")
        # 위치 센서: 모터에 직결된 센서 우선 (가장 확실), 이름 매칭 폴백
        try:
            self.left_sensor = self.left_motor.getPositionSensor()
            self.right_sensor = self.right_motor.getPositionSensor()
        except Exception:
            pass
        if self.left_sensor is None or self.right_sensor is None:
            for s in psensors:
                n = s.getName().lower()
                if "left" in n and self.left_sensor is None:
                    self.left_sensor = s
                elif "right" in n and self.right_sensor is None:
                    self.right_sensor = s
        if self.left_sensor is None or self.right_sensor is None:
            names = [s.getName() for s in psensors]
            raise RuntimeError(f"바퀴 위치 센서 탐색 실패. 센서 목록: {names} "
                               f"— odometry 불가, robot_io 수정 필요")

    def _enable(self):
        ts = self.timestep
        if self.lidar:
            self.lidar.enable(ts)
        if self.camera:
            self.camera.enable(ts)
        if self.compass:
            self.compass.enable(ts)
        for s in (self.left_sensor, self.right_sensor):
            if s:
                s.enable(ts)
        for m in (self.left_motor, self.right_motor):
            m.setPosition(float("inf"))
            m.setVelocity(0.0)

    # ── 센서 읽기 ────────────────────────────────────────────
    def step(self):
        return self.robot.step(self.timestep)

    def wheel_positions(self):
        """(left_rad, right_rad) 누적 회전각."""
        return (self.left_sensor.getValue(), self.right_sensor.getValue())

    def compass_raw(self):
        """컴퍼스 원시각(rad, 기울기 +1 규약) 또는 None.

        부호/기준 변환은 sar.odometry.compass_values_to_raw 한 곳에서만.
        PoseEstimator가 시작 헤딩(제공값)으로 상수 오프셋을 캘리브레이션한다.
        """
        if self.compass is None:
            return None
        v = self.compass.getValues()
        if v is None or not all(math.isfinite(x) for x in v[:2]):
            return None
        if math.hypot(v[0], v[1]) < 0.05:
            # XY 성분 퇴화(예: 센서가 수직으로 기울어 (0,0,1)) → 사용 불가
            return None
        return compass_values_to_raw(v)

    def lidar_scan(self):
        """(angles, ranges) — angles: 로봇 프레임 bearing (CCW+, 0=전방)."""
        if self.lidar is None:
            return None, None
        buf = self.lidar.getRangeImage()
        if not buf:                          # 첫 틱 등 데이터 미준비
            return None, None
        ranges = np.array(buf, dtype=np.float64)
        n = self.lidar.getHorizontalResolution()
        if ranges.size < n:
            return None, None
        layers = self.lidar.getNumberOfLayers()
        if layers > 1:                       # 다층이면 중앙층만
            ranges = ranges.reshape(layers, n)[layers // 2]
        if self._lidar_angles is None or self._lidar_angles.size != n:
            c = self.cfg.lidar
            fov = self.lidar.getFov()
            # Webots 관례: index0 = FOV 왼쪽 끝(+fov/2), index 증가 = CW.
            # 로봇별로 다를 수 있어 config로 오버라이드 가능하게 둠.
            if abs(fov - 2 * math.pi) < 0.1:  # 360도 라이다
                sign = 1.0 if c.ccw else -1.0
                idx = np.arange(n)
                self._lidar_angles = wrap_angle(
                    c.index0_angle + sign * idx * (2 * math.pi / n))
            else:                             # 제한 FOV 라이다
                self._lidar_angles = np.linspace(fov / 2, -fov / 2, n)
        return self._lidar_angles, ranges

    def camera_image(self):
        """(H,W,3) uint8 RGB 또는 None."""
        if self.camera is None:
            return None
        w = self.camera.getWidth()
        h = self.camera.getHeight()
        buf = self.camera.getImage()
        if buf is None:
            return None
        bgra = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
        return bgra[..., 2::-1].copy()        # BGRA → RGB

    # ── 구동 ─────────────────────────────────────────────────
    def drive(self, v, w):
        cfg = self.cfg.robot
        wl = (v - w * cfg.wheel_base / 2) / cfg.wheel_radius
        wr = (v + w * cfg.wheel_base / 2) / cfg.wheel_radius
        lim = cfg.max_wheel_speed
        scale = max(1.0, abs(wl) / lim, abs(wr) / lim)
        self.left_motor.setVelocity(wl / scale)
        self.right_motor.setVelocity(wr / scale)
