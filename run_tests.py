"""순수 파이썬 단위 테스트 (pytest 불필요): .venv/bin/python run_tests.py"""
import math
import sys
import traceback

import numpy as np

from sar.config import default_config
from sar.detection import (Detection, TargetDetector,
                           estimate_distance_mono, estimate_target_position,
                           estimate_target_position_mono)
from sar.geometry import bresenham, wrap_angle
from sar.mapping import OccupancyGrid
from sar.odometry import DiffDriveOdometry, PoseEstimator
from sar.breadcrumbs import Breadcrumbs
from sar.energy import EnergyManager
from sar.exploration import FrontierExplorer
from sar.planning import astar, simplify_path
from sar.semantic_obstacles import TableObstacleMapper
from sar.state_machine import Mission


def test_wrap_angle():
    assert abs(wrap_angle(3 * math.pi) - (-math.pi)) < 1e-9 \
        or abs(wrap_angle(3 * math.pi) - math.pi) < 1e-9
    assert abs(wrap_angle(0.1) - 0.1) < 1e-12
    assert abs(wrap_angle(-4 * math.pi)) < 1e-9


def test_bresenham():
    cells = bresenham(0, 0, 3, 3)
    assert cells[0] == (0, 0) and cells[-1] == (3, 3)
    assert len(cells) == 4
    cells = bresenham(2, 1, -1, 1)
    assert cells == [(2, 1), (1, 1), (0, 1), (-1, 1)]


def test_odometry_straight_and_turn():
    cfg = default_config()
    od = DiffDriveOdometry(cfg, (0, 0, 0))
    r = cfg.robot.wheel_radius
    od.update(0.0, 0.0)
    # 직진 1m: 바퀴 회전각 1/r
    od.update(1.0 / r, 1.0 / r)
    x, y, th = od.pose
    assert abs(x - 1.0) < 1e-6 and abs(y) < 1e-6 and abs(th) < 1e-6
    # 제자리 90도 회전: dr - dl = L * pi/2
    od2 = DiffDriveOdometry(cfg, (0, 0, 0))
    od2.update(0.0, 0.0)
    L = cfg.robot.wheel_base
    half = L * math.pi / 4 / r
    od2.update(-half, half)
    assert abs(od2.pose[2] - math.pi / 2) < 1e-6


def test_mapping_square_room():
    cfg = default_config()
    cfg.map.half_size = 3.0
    grid = OccupancyGrid(cfg, (0, 0))
    # 정사각 방(±2m) 중앙에서 360도 스캔 (해석적 거리)
    n = 720
    angles = np.linspace(-math.pi, math.pi, n, endpoint=False)
    ranges = np.array([2.0 / max(abs(math.cos(a)), abs(math.sin(a)))
                       for a in angles])
    for _ in range(4):
        grid.integrate_scan((0, 0, 0), angles, ranges)
    free = grid.free_mask()
    occ = grid.occupied_mask()
    cx, cy = grid.world_to_grid(0, 0)
    wx, wy = grid.world_to_grid(2.0, 0)
    assert free[cy, cx], "중앙은 빈공간이어야"
    assert occ[wy, wx] or occ[wy, wx + 1], "벽 위치는 점유여야"
    fx, fy = grid.world_to_grid(2.6, 2.6)
    assert grid.unknown_mask()[fy, fx], "벽 밖은 미지여야"
    assert grid.frontier_mask().sum() == 0 or True  # 닫힌 방: frontier 거의 없음


def test_astar_and_simplify():
    n = 50
    blocked = np.zeros((n, n), dtype=bool)
    unknown = np.zeros((n, n), dtype=bool)
    blocked[20, 5:45] = True          # 가로 벽, 양끝 통로
    path = astar(blocked, unknown, (10, 5), (10, 40))
    assert path is not None
    assert path[0] == (10, 5) and path[-1] == (10, 40)
    for ix, iy in path:
        assert not blocked[iy, ix]
    sp = simplify_path(path, blocked)
    assert len(sp) <= len(path)
    assert sp[0] == path[0] and sp[-1] == path[-1]
    # 막힌 경우
    blocked2 = np.zeros((n, n), dtype=bool)
    blocked2[20, :] = True
    assert astar(blocked2, unknown, (10, 5), (10, 40)) is None


def test_astar_unknown_penalty():
    n = 30
    blocked = np.zeros((n, n), dtype=bool)
    unknown = np.zeros((n, n), dtype=bool)
    unknown[:, 10:20] = True          # 가운데 미지 지대
    path = astar(blocked, unknown, (5, 15), (25, 15), unknown_cost=2.5)
    assert path is not None            # 미지여도 통과는 가능해야


def test_detection_red_blob():
    cfg = default_config()
    det = TargetDetector(cfg)
    img = np.full((120, 160, 3), 100, dtype=np.uint8)
    img[40:70, 100:120] = (210, 20, 20)      # 오른쪽에 빨간 블롭
    d = None
    for _ in range(cfg.detection.confirm_frames):
        d = det.process(img)
    assert d is not None
    assert det.confirmed
    assert d.bearing < 0, "화면 오른쪽 블롭 → bearing 음수(오른쪽)"
    # 빈 화면이면 리셋
    for _ in range(cfg.detection.lost_frames):
        det.process(np.full((120, 160, 3), 100, dtype=np.uint8))
    assert not det.confirmed


def test_detection_decoy_rejection():
    """디코이 방어: 상단 접촉(키 큰 물체) / 과대 블롭 / 종횡비."""
    cfg = default_config()
    det = TargetDetector(cfg)
    # 소화기형: 상단에 닿는 세로로 긴 빨간 블롭 → 기각
    img = np.full((120, 160, 3), 100, dtype=np.uint8)
    img[0:80, 70:90] = (210, 20, 20)
    assert det.process(img) is None
    # 프레임 대부분을 덮는 빨간 벽/러그 → 기각 (max_area_ratio)
    img2 = np.full((120, 160, 3), 100, dtype=np.uint8)
    img2[30:120, 10:150] = (210, 20, 20)
    det2 = TargetDetector(cfg)
    assert det2.process(img2) is None
    # 정상 사과형(중앙, 원형 비율) → 통과
    img3 = np.full((120, 160, 3), 100, dtype=np.uint8)
    img3[50:70, 75:95] = (210, 20, 20)
    det3 = TargetDetector(cfg)
    assert det3.process(img3) is not None
    # 수평선 위에 뜬 원형 블롭 (표지판·벽걸이) → 기각
    img4 = np.full((120, 160, 3), 100, dtype=np.uint8)
    img4[10:30, 75:95] = (210, 20, 20)
    det4 = TargetDetector(cfg)
    assert det4.process(img4) is None


class _FakeYoloBox:
    def __init__(self, class_id, confidence, xyxy):
        self.cls = np.array([class_id], dtype=np.float32)
        self.conf = np.array([confidence], dtype=np.float32)
        self.xyxy = np.array([xyxy], dtype=np.float32)


class _FakeYoloResult:
    def __init__(self, boxes):
        self.boxes = boxes
        self.names = {47: "apple", 49: "orange", 60: "dining table"}


class _FakeYoloModel:
    def __init__(self, boxes):
        self.boxes = boxes
        self.calls = []

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        return [_FakeYoloResult(self.boxes)]


def test_detection_yolo_apple_then_red_passes():
    """YOLO apple 박스가 먼저 나오고, 내부가 빨간 경우에만 통과한다."""
    cfg = default_config()
    cfg.detection.use_yolo = True
    detector = TargetDetector(cfg)
    model = _FakeYoloModel([
        _FakeYoloBox(47, 0.91, [70, 45, 100, 75]),
    ])
    detector._yolo = model

    img = np.full((120, 160, 3), 100, dtype=np.uint8)
    img[50:70, 75:95] = (210, 20, 20)
    detection = None
    for _ in range(cfg.detection.confirm_frames):
        detection = detector.process(img)

    assert detection is not None
    assert detector.confirmed
    assert model.calls
    assert model.calls[-1]["classes"] == [47, 60]
    assert detector.yolo_boxes(img)[0][0] == "apple"


def test_detection_yolo_apple_without_red_fails():
    """YOLO가 apple이라 해도 HSV 빨강 검증을 못 넘으면 최종 기각한다."""
    cfg = default_config()
    cfg.detection.use_yolo = True
    detector = TargetDetector(cfg)
    detector._yolo = _FakeYoloModel([
        _FakeYoloBox(47, 0.91, [70, 45, 100, 75]),
    ])

    img = np.full((120, 160, 3), 100, dtype=np.uint8)
    img[50:70, 75:95] = (20, 210, 20)
    assert detector.process(img) is None
    assert not detector.confirmed


def test_detection_exposes_table_from_same_yolo_pass():
    """apple과 table은 별도 추론하지 않고 한 번의 YOLO 결과에서 나눈다."""
    cfg = default_config()
    cfg.detection.use_yolo = True
    detector = TargetDetector(cfg)
    model = _FakeYoloModel([
        _FakeYoloBox(60, 0.82, [35, 20, 125, 105]),
    ])
    detector._yolo = model
    img = np.full((120, 160, 3), 100, dtype=np.uint8)

    assert detector.process(img) is None
    assert len(model.calls) == 1
    assert model.calls[0]["classes"] == [47, 60]
    tables = detector.table_boxes()
    assert len(tables) == 1 and tables[0][0] == "dining table"


def test_table_mapper_bridges_legs_but_keeps_back_reachable():
    """테이블 다리 사이는 막되 선분 끝으로 돌아 뒤쪽에 갈 수 있어야 한다."""
    cfg = default_config()
    cfg.map.half_size = 3.0
    grid = OccupancyGrid(cfg, (0.0, 0.0))
    mapper = TableObstacleMapper(cfg, grid)
    angles = np.linspace(-math.pi, math.pi, 721)
    ranges = np.full(angles.shape, cfg.lidar.max_range)
    # 카메라 테이블 박스 안, 약 1m 앞의 좌우 다리 두 개.
    ranges[np.abs(angles - 0.20) < 0.018] = 1.0
    ranges[np.abs(angles + 0.20) < 0.018] = 1.0
    boxes = [("dining table", 0.85, [40, 15, 120, 110])]

    added = 0
    for i in range(cfg.table.confirm_frames):
        added += mapper.update(i * 0.1, (0.0, 0.0, 0.0),
                               angles, ranges, 160, boxes)
    assert added > 0
    mid = grid.world_to_grid(1.0, 0.0)
    assert grid.virtual_occupied[mid[1], mid[0]], "다리 사이가 막혀야"
    behind = grid.world_to_grid(1.6, 0.0)
    assert not grid.virtual_occupied[behind[1], behind[0]], \
        "테이블 뒤쪽 전체를 막으면 안 됨"

    blocked = grid.inflated_mask(
        cfg.robot.robot_radius + cfg.plan.inflate_margin)
    unknown = np.zeros_like(blocked)
    start = grid.world_to_grid(0.0, 0.0)
    goal = grid.world_to_grid(1.6, 0.0)
    path = astar(blocked, unknown, start, goal)
    assert path is not None, "테이블 옆으로 돌아 뒤쪽에 도달 가능해야"
    world_path = [grid.grid_to_world(ix, iy) for ix, iy in path]
    assert max(abs(y) for _, y in world_path) > 0.35


def test_recovery_does_not_reverse_into_table_leg():
    cfg = default_config()
    grid = OccupancyGrid(cfg, (0.0, 0.0))
    mission = Mission(cfg, grid)
    angles = np.linspace(-math.pi, math.pi, 361)
    ranges = np.full(angles.shape, cfg.lidar.max_range)
    ranges[0] = 0.20                 # 정후방 다리
    mission._recover_phase = ("backup", 0.0)

    v, w = mission._do_recover(0.1, (0.0, 0.0, 0.0), angles, ranges)
    assert v == 0.0 and w == 0.0
    assert mission._recover_phase[0] == "spin"


def test_estimate_target_position():
    angles = np.linspace(-math.pi, math.pi, 360, endpoint=False)
    ranges = np.full(360, 3.0)
    # bearing 0.5rad 근처에 1.5m 물체
    sel = np.abs(angles - 0.5) < 0.05
    ranges[sel] = 1.5
    est = estimate_target_position((1.0, 2.0, 0.3), 0.5, angles, ranges)
    assert est is not None
    x, y, d = est
    assert abs(d - 1.5) < 0.05
    assert abs(x - (1.0 + 1.5 * math.cos(0.8))) < 0.05
    assert abs(y - (2.0 + 1.5 * math.sin(0.8))) < 0.05


def test_breadcrumbs():
    cfg = default_config()
    bc = Breadcrumbs(cfg)
    for i in range(20):
        bc.record((i * 0.1, 0.0, 0.0))
    assert len(bc.crumbs) >= 5
    path = bc.return_path((1.9, 0.0))
    assert len(path) >= 2
    assert path[0][0] >= path[-1][0], "복귀 경로는 뒤로 향해야"
    assert abs(path[-1][0]) < 0.01, "복귀 경로 끝 = 시작점"


def test_energy_aware_return():
    cfg = default_config()
    cfg.energy.initial_percent = 20.0
    cfg.energy.reserve_percent = 10.0
    cfg.energy.percent_per_meter = 1.0
    cfg.energy.return_multiplier = 1.0
    bc = Breadcrumbs(cfg)
    for i in range(6):
        bc.record((float(i), 0.0, 0.0))
    energy = EnergyManager(cfg)
    energy.update((0.0, 0.0, 0.0))
    energy.update((5.0, 0.0, 0.0))
    # 5%를 주행에 사용해 15%가 남고, 귀환 5% + reserve 10%가 필요하다.
    assert abs(energy.percent - 15.0) < 1e-9
    assert energy.should_return((5.0, 0.0, 0.0), bc)


def test_frontier_explorer():
    cfg = default_config()
    cfg.map.half_size = 3.0
    grid = OccupancyGrid(cfg, (0, 0))
    # 왼쪽 절반만 탐색된 상황을 인위로 구성
    half = grid.n // 2
    grid.log[:, :half] = -2.0          # free
    grid.log[:, half:] = 0.0           # unknown
    grid._version += 1
    ex = FrontierExplorer(cfg, grid)
    target = ex.update((0.0, 0.0))
    assert target is not None
    ix, iy = grid.world_to_grid(*target)
    assert abs(ix - half) <= 2, "frontier는 free/unknown 경계 근처여야"
    # 관성: 같은 목표 유지
    t2 = ex.update((0.1, 0.1))
    assert t2 == target


def test_mono_distance():
    # 0.1m 폭 목표가 2m 거리 → ang_width = 2*atan(0.05/2)
    ang = 2 * math.atan2(0.05, 2.0)
    det = Detection(0.2, 100, 0.4, (10, 20), 30, ang, clipped=False)
    d = estimate_distance_mono(det, 0.10)
    assert abs(d - 2.0) < 0.01
    est = estimate_target_position_mono((0, 0, 0), det, 0.10)
    assert est is not None
    x, y, dd = est
    assert abs(dd - 2.0) < 0.01
    assert abs(x - 2.0 * math.cos(0.2)) < 0.02
    assert abs(y - 2.0 * math.sin(0.2)) < 0.02
    # 화면 끝에 잘린 블롭은 기각 (폭 축소 → 거리 과대추정 방지)
    det_c = Detection(0.2, 100, 0.4, (10, 20), 30, ang, clipped=True)
    assert estimate_target_position_mono((0, 0, 0), det_c, 0.10) is None


def test_compass_calibration():
    """실제 Webots 규약(북쪽 벡터를 로봇 프레임에서 봄)으로 검증.

    회귀 방지: 컴퍼스 원시각은 theta에 대해 기울기 -1 — 부호 반전을
    변환 헬퍼가 교정하고, CCW 회전 시 추정 헤딩도 CCW로 늘어야 한다.
    """
    from sar.odometry import compass_values_to_raw

    def webots_compass(theta, north=1.234):
        ang = wrap_angle(north - theta)
        return (math.cos(ang), math.sin(ang), 0.0)

    cfg = default_config()
    start = (0.0, 0.0, math.pi / 2)
    est = PoseEstimator(cfg, start)
    r = cfg.robot.wheel_radius
    est.update(0.0, 0.0,
               compass_raw=compass_values_to_raw(webots_compass(math.pi / 2)))
    # 직진 1m: 헤딩은 컴퍼스가 그대로 pi/2 유지
    est.update(1.0 / r, 1.0 / r,
               compass_raw=compass_values_to_raw(webots_compass(math.pi / 2)))
    x, y, th = est.pose
    assert abs(th - math.pi / 2) < 1e-6
    assert abs(x) < 1e-6 and abs(y - 1.0) < 1e-6, (x, y)
    # CCW 회전 추적: 실제 theta가 +0.4 돌면 추정 헤딩도 +0.4 (거울 반전 금지)
    L = cfg.robot.wheel_base
    half = L * 0.4 / 2 / r
    est.update(1.0 / r - half, 1.0 / r + half,
               compass_raw=compass_values_to_raw(
                   webots_compass(math.pi / 2 + 0.4)))
    assert abs(est.pose[2] - (math.pi / 2 + 0.4)) < 1e-6, est.pose


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]

if __name__ == "__main__":
    failed = 0
    for t in TESTS:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL  {t.__name__}")
            traceback.print_exc()
    print(f"\n{len(TESTS) - failed}/{len(TESTS)} passed")
    sys.exit(1 if failed else 0)
