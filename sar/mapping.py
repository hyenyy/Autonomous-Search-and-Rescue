"""NumPy Occupancy Grid (log-odds).

배열 인덱스는 [iy, ix]. 셀 상태는 log-odds 실수 하나로 관리:
  > occ_threshold  : 장애물
  < free_threshold : 빈공간
  그 사이          : 미지(unknown)
"""
import math

import numpy as np


class OccupancyGrid:
    def __init__(self, cfg, center_xy=(0.0, 0.0)):
        m = cfg.map
        self.res = m.resolution
        self.n = int(round(2 * m.half_size / m.resolution))
        self.origin_x = center_xy[0] - m.half_size
        self.origin_y = center_xy[1] - m.half_size
        self.log = np.zeros((self.n, self.n), dtype=np.float32)
        # 카메라 의미 인식으로 보강한 정적 장애물. log-odds와 분리해
        # 이후 LiDAR free ray가 테이블 다리 사이를 다시 지우지 못하게 한다.
        self.virtual_occupied = np.zeros((self.n, self.n), dtype=bool)
        self.l_occ = m.l_occ
        self.l_free = m.l_free
        self.l_clamp = m.l_clamp
        self.occ_th = m.occ_threshold
        self.free_th = m.free_threshold
        self._lidar_max = cfg.lidar.max_range
        self._inflate_cache = None  # (log 버전 번호, 반경) 키 캐시
        self._version = 0

    # ── 좌표 변환 ──────────────────────────────────────────────
    def world_to_grid(self, x, y):
        # floor 사용: int()는 0 방향 절단이라 원점보다 음수인 좌표가
        # 셀 0으로 별칭되는 잠복 결함이 있다
        ix = int(math.floor((x - self.origin_x) / self.res))
        iy = int(math.floor((y - self.origin_y) / self.res))
        return ix, iy

    def grid_to_world(self, ix, iy):
        return (
            self.origin_x + (ix + 0.5) * self.res,
            self.origin_y + (iy + 0.5) * self.res,
        )

    def in_bounds(self, ix, iy):
        return 0 <= ix < self.n and 0 <= iy < self.n

    # ── 갱신 ──────────────────────────────────────────────────
    def integrate_scan(self, pose, angles, ranges, subsample=1, exclude=None):
        """pose에서의 LiDAR 스캔을 흡수. angles: 로봇 프레임 bearing.

        exclude: 빔별 bool — True인 빔(동적 장애물로 분류된 것)은 통째로
        제외한다. 사람의 이동 궤적이 지도에 점유 흔적으로 쌓이면
        동적 분류('원래 free였던 셀' 판정)가 무력화되기 때문.

        벡터화 구현: 각 빔을 res*0.7 간격으로 샘플링해 free 셀을 일괄 마킹.
        (bresenham 대비 대각 셀을 드물게 건너뛰지만 매핑 용도로는 충분)
        """
        x, y, th = pose
        ix0, iy0 = self.world_to_grid(x, y)
        if not self.in_bounds(ix0, iy0):
            return
        angles = np.asarray(angles, dtype=np.float64)
        ranges = np.asarray(ranges, dtype=np.float64)
        if exclude is not None:
            keep = ~np.asarray(exclude, dtype=bool)
            angles, ranges = angles[keep], ranges[keep]
        angles = angles[::subsample]
        ranges = ranges[::subsample]
        valid = np.isfinite(ranges) & (ranges > 0.01)
        angles, ranges = angles[valid], ranges[valid]
        if angles.size == 0:
            return
        # max_range 초과분은 "그 거리까지 비어있음"으로만 사용
        hit = ranges < self._lidar_max * 0.99
        r = np.minimum(ranges, self._lidar_max)

        step = self.res * 0.7
        n_s = int(self._lidar_max / step) + 1
        t = (np.arange(n_s) + 0.5) * step                       # (S,)
        ca = np.cos(th + angles)
        sa = np.sin(th + angles)
        px = x + ca[:, None] * t[None, :]                       # (B,S)
        py = y + sa[:, None] * t[None, :]
        free_m = t[None, :] < (r[:, None] - self.res * 0.6)
        ix = ((px - self.origin_x) / self.res).astype(np.int64)
        iy = ((py - self.origin_y) / self.res).astype(np.int64)
        inb = (ix >= 0) & (ix < self.n) & (iy >= 0) & (iy < self.n)
        m = free_m & inb
        if m.any():
            flat = np.unique(iy[m] * self.n + ix[m])
            self.log.ravel()[flat] -= self.l_free
        # 빔 끝점(실제 hit만) → 점유 증가
        if hit.any():
            hx = x + ca[hit] * r[hit]
            hy = y + sa[hit] * r[hit]
            hix = ((hx - self.origin_x) / self.res).astype(np.int64)
            hiy = ((hy - self.origin_y) / self.res).astype(np.int64)
            inb2 = (hix >= 0) & (hix < self.n) & (hiy >= 0) & (hiy < self.n)
            if inb2.any():
                flat2 = np.unique(hiy[inb2] * self.n + hix[inb2])
                self.log.ravel()[flat2] += self.l_occ
        np.clip(self.log, -self.l_clamp, self.l_clamp, out=self.log)
        self._version += 1

    # ── 조회 ──────────────────────────────────────────────────
    def occupied_mask(self):
        return (self.log > self.occ_th) | self.virtual_occupied

    def free_mask(self):
        return (self.log < self.free_th) & ~self.virtual_occupied

    def unknown_mask(self):
        return (self.log >= self.free_th) & (self.log <= self.occ_th) \
            & ~self.virtual_occupied

    def inflated_mask(self, radius_m):
        """장애물을 radius_m 만큼 팽창한 통행불가 마스크 (버전 캐시)."""
        r_cells = max(1, int(math.ceil(radius_m / self.res)))
        key = (self._version, r_cells)
        if self._inflate_cache and self._inflate_cache[0] == key:
            return self._inflate_cache[1]
        occ = self.occupied_mask()
        out = occ.copy()
        for dy in range(-r_cells, r_cells + 1):
            for dx in range(-r_cells, r_cells + 1):
                if dx * dx + dy * dy > r_cells * r_cells or (dx == 0 and dy == 0):
                    continue
                shifted = np.zeros_like(occ)
                ys = slice(max(0, dy), self.n + min(0, dy))
                xs = slice(max(0, dx), self.n + min(0, dx))
                ys_src = slice(max(0, -dy), self.n + min(0, -dy))
                xs_src = slice(max(0, -dx), self.n + min(0, -dx))
                shifted[ys, xs] = occ[ys_src, xs_src]
                out |= shifted
        self._inflate_cache = (key, out)
        return out

    def mark_virtual_segment(self, p0, p1, thickness_m):
        """월드 좌표 선분 주변을 영구 가상 장애물로 표시한다.

        테이블의 앞쪽 두 다리를 연결해 '들어갈 수 있는 틈'이 아니라
        '돌아가야 하는 가장자리'로 보이게 한다. 선분 끝 바깥은 열려 있어
        테이블 뒤 공간은 계속 탐색할 수 있다.
        """
        x0, y0 = p0
        x1, y1 = p1
        pad = max(float(thickness_m), self.res)
        ix0, iy0 = self.world_to_grid(min(x0, x1) - pad,
                                      min(y0, y1) - pad)
        ix1, iy1 = self.world_to_grid(max(x0, x1) + pad,
                                      max(y0, y1) + pad)
        ix0, iy0 = max(0, ix0), max(0, iy0)
        ix1, iy1 = min(self.n - 1, ix1), min(self.n - 1, iy1)
        if ix0 > ix1 or iy0 > iy1:
            return 0

        xs = self.origin_x + (np.arange(ix0, ix1 + 1) + 0.5) * self.res
        ys = self.origin_y + (np.arange(iy0, iy1 + 1) + 0.5) * self.res
        xx, yy = np.meshgrid(xs, ys)
        vx, vy = x1 - x0, y1 - y0
        denom = vx * vx + vy * vy
        if denom <= 1e-12:
            d2 = (xx - x0) ** 2 + (yy - y0) ** 2
        else:
            t = np.clip(((xx - x0) * vx + (yy - y0) * vy) / denom,
                        0.0, 1.0)
            d2 = (xx - (x0 + t * vx)) ** 2 + (yy - (y0 + t * vy)) ** 2
        local = d2 <= pad * pad
        view = self.virtual_occupied[iy0:iy1 + 1, ix0:ix1 + 1]
        added = int(np.count_nonzero(local & ~view))
        if added:
            view |= local
            self._version += 1
            self._inflate_cache = None
        return added

    def frontier_mask(self):
        """빈공간 셀 중 미지 셀과 4-이웃으로 접한 셀."""
        free = self.free_mask()
        unk = self.unknown_mask()
        near_unk = np.zeros_like(unk)
        near_unk[1:, :] |= unk[:-1, :]
        near_unk[:-1, :] |= unk[1:, :]
        near_unk[:, 1:] |= unk[:, :-1]
        near_unk[:, :-1] |= unk[:, 1:]
        return free & near_unk
