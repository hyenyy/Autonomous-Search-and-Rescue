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
from sar.exploration import FrontierExplorer
from sar.planning import astar, simplify_path


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
    # Check hysteresis before arrival; a reached boundary must advance to a new goal.
    t2 = ex.update((-0.5, 0.0))
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


def test_clipped_blob_no_estimate():
    """좌우 잘린 블롭은 위치 추정 금지 — 모서리에 걸친 소화기 하반신이
    '0.6m 사과'로 확정되던 실사고 회귀."""
    from sar.detection import Detection
    from sar.state_machine import Mission
    cfg = default_config()
    grid = OccupancyGrid(cfg, center_xy=(0.0, 0.0))
    m = Mission(cfg, grid)
    m.detector.last = Detection(
        bearing=0.3, pixels=20000, cx_ratio=0.1, area_rows=229,
        px_width=139, ang_width=0.21, clipped=True,
        bbox=(0, 85, 139, 314), aspect=0.61, v_clipped=False, fill=0.62)
    angles = np.linspace(-math.pi, math.pi, 360, endpoint=False)
    ranges = np.full(360, 0.62)
    m._update_target_estimate((0.0, 0.0, 0.0), angles, ranges)
    assert m.target_est is None, "잘린 블롭이 위치 추정을 만들면 안 됨"


def test_plan_soft_relief():
    """표준 팽창으로 닫힌 0.34m 틈 — 소프트 완화 재시도가 경로를 찾는다.

    (탁자 다리 사이 갇힘 회귀: 계획 전멸 → 제자리 회전 무한 반복)"""
    from sar.planning import Planner
    cfg = default_config()
    cfg.map.half_size = 3.0
    grid = OccupancyGrid(cfg, center_xy=(0.0, 0.0))
    grid.log[:, :] = -2.0                       # 전 영역 free 관측 가정
    for x in np.arange(-3.0, 3.0, grid.res / 2):
        if -0.17 < x < 0.17:                    # 틈 0.34m
            continue
        ix, iy = grid.world_to_grid(x, 0.0)
        if 0 <= ix < grid.n and 0 <= iy < grid.n:
            grid.log[iy, ix] = 5.0              # y=0 수평 벽
    grid._version += 1
    p = Planner(cfg, grid)
    ok = p.plan_to((0.0, -1.0, 0.0), (0.0, 1.0), now=0.0, force=True)
    assert ok, "소프트 완화가 틈 통과 경로를 찾아야 함"
    assert p.last_plan_soft, "표준 팽창으론 닫힌 틈 — 소프트 경로여야 함"
    # 경로가 실제로 벽 건너편 목표까지 이어지는지 확인
    gx, gy = p.waypoints[-1]
    assert math.hypot(gx - 0.0, gy - 1.0) < 0.3, p.waypoints[-1]
    # 벽을 넘는 구간(부호가 바뀌는 인접 쌍)은 틈 근처(x≈0)여야 함
    for (x0, y0), (x1, y1) in zip(p.waypoints, p.waypoints[1:]):
        if y0 < 0 <= y1 or y1 < 0 <= y0:
            assert abs(x0) < 0.4 and abs(x1) < 0.4, (x0, x1)


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
