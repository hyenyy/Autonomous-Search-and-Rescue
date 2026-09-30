"""각도/좌표 유틸리티. 좌표 규약(전체 코드 공통):

- 월드 프레임: x 전방(East), y 좌측(North), theta는 +x축 기준 CCW 라디안 (ENU)
- 로봇 프레임: x 전방, y 좌측 (FLU). bearing +값 = 왼쪽(CCW)
- 그리드: cell = (ix, iy), ix = (x - origin_x)/resolution. numpy 배열은 [iy, ix] 순서
"""
import math
import numpy as np


def wrap_angle(a):
    """[-pi, pi) 로 정규화. 스칼라/배열 모두 지원."""
    return (a + math.pi) % (2 * math.pi) - math.pi


def rot2d(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]])


def robot_to_world(pose, pts_xy):
    """pose=(x,y,theta), pts_xy: (N,2) 로봇 프레임 점들 → 월드 프레임."""
    R = rot2d(pose[2])
    return np.asarray(pts_xy) @ R.T + np.array(pose[:2])


def scan_to_points(angles, ranges):
    """(bearing, range) → 로봇 프레임 (N,2) 직교좌표."""
    angles = np.asarray(angles)
    ranges = np.asarray(ranges)
    return np.stack([ranges * np.cos(angles), ranges * np.sin(angles)], axis=1)


def bresenham(ix0, iy0, ix1, iy1):
    """정수 셀 (ix0,iy0)→(ix1,iy1) 선분이 지나는 셀 목록 (끝점 포함)."""
    cells = []
    dx = abs(ix1 - ix0)
    dy = abs(iy1 - iy0)
    sx = 1 if ix0 < ix1 else -1
    sy = 1 if iy0 < iy1 else -1
    err = dx - dy
    x, y = ix0, iy0
    while True:
        cells.append((x, y))
        if x == ix1 and y == iy1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy
    return cells


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])
