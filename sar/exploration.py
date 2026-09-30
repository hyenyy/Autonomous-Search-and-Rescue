"""Frontier 기반 탐색 목표 선택.

- frontier = 빈공간과 미지 영역의 경계 셀
- 클러스터링(BFS) 후 점수 = 클러스터 크기 / 직선거리
- 관성(hysteresis): 한번 고른 목표는 도달/소멸/실패 전까지 유지 (진동 방지)
- 실패한 목표는 블랙리스트(반경 내 재선택 금지)
"""
from collections import deque

from .geometry import dist


class FrontierExplorer:
    def __init__(self, cfg, grid):
        self.cfg = cfg
        self.grid = grid
        self.current_target = None       # (x, y) 월드 좌표
        self.blacklist = []              # [(x, y), ...]
        self.just_reached = False        # frontier 도착 직후 1틱 True

    def _clusters(self):
        mask = self.grid.frontier_mask()
        ys, xs = mask.nonzero()
        cells = set(zip(xs.tolist(), ys.tolist()))
        clusters = []
        while cells:
            seed = cells.pop()
            comp = [seed]
            q = deque([seed])
            while q:
                cx, cy = q.popleft()
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        nb = (cx + dx, cy + dy)
                        if nb in cells:
                            cells.remove(nb)
                            comp.append(nb)
                            q.append(nb)
            if len(comp) >= self.cfg.explore.min_cluster:
                clusters.append(comp)
        return clusters

    def _representative(self, comp):
        """클러스터 대표점: 중심에 가장 가까운 실제 frontier 셀 (도달성 확보)."""
        mx = sum(c[0] for c in comp) / len(comp)
        my = sum(c[1] for c in comp) / len(comp)
        best = min(comp, key=lambda c: (c[0] - mx) ** 2 + (c[1] - my) ** 2)
        return self.grid.grid_to_world(*best)

    def _blacklisted(self, xy):
        r = self.cfg.explore.blacklist_radius
        return any(dist(xy, b) < r for b in self.blacklist)

    def target_reached(self, pose):
        return (self.current_target is not None
                and dist(pose, self.current_target) < self.cfg.explore.reach_tolerance)

    def target_still_frontier(self):
        """현재 목표 주변이 여전히 미탐색 경계인지 (탐색되면 자연 소멸)."""
        if self.current_target is None:
            return False
        ix, iy = self.grid.world_to_grid(*self.current_target)
        mask = self.grid.frontier_mask()
        r = 3
        y0, y1 = max(0, iy - r), min(self.grid.n, iy + r + 1)
        x0, x1 = max(0, ix - r), min(self.grid.n, ix + r + 1)
        return bool(mask[y0:y1, x0:x1].any())

    def fail_current(self):
        """현재 목표 도달 실패 → 블랙리스트 후 재선택 유도."""
        if self.current_target is not None:
            self.blacklist.append(self.current_target)
            self.current_target = None

    def update(self, pose, person_xy=None):
        """탐색 목표 반환 ((x,y) 또는 None=frontier 소진).

        person_xy가 있으면 그 주변 frontier의 점수를 깎는다 — 움직이는
        사람이 있는 쪽은 뒤로 미루고 반대 방향부터 탐색 (동선 겹침 회피).
        """
        # 관성: 유효한 기존 목표 유지
        if self.current_target is not None:
            if self.target_reached(pose):
                self.just_reached = True     # 새 시야 — 카메라 스캔 타이밍
                self.current_target = None
            elif not self.target_still_frontier():
                self.current_target = None
            else:
                return self.current_target

        best, best_score = None, -1.0
        for comp in self._clusters():
            rep = self._representative(comp)
            if self._blacklisted(rep):
                continue
            d = max(dist(pose, rep), 0.3)
            score = len(comp) / d
            if person_xy is not None and dist(rep, person_xy) < 2.5:
                score *= 0.35        # 사람 있는 방향은 후순위
            if score > best_score:
                best, best_score = rep, score
        self.current_target = best
        return best
