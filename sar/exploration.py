"""Frontier 기반 탐색 목표 선택.

- frontier = 빈공간과 미지 영역의 경계 셀
- 클러스터링(BFS) 후 크기·거리·구역 재방문 횟수로 우선순위 결정
- 관성(hysteresis): 한번 고른 목표는 도달/소멸/실패 전까지 유지 (진동 방지)
- 실패한 목표는 블랙리스트(반경 내 재선택 금지)
"""
from collections import deque
import math

from .geometry import dist


class FrontierExplorer:
    def __init__(self, cfg, grid):
        self.cfg = cfg
        self.grid = grid
        self.current_target = None       # (x, y) 월드 좌표
        self.blacklist = []              # [(x, y), ...]
        self.completed_targets = []      # Goal exclusion only; routes may pass through.
        self.region_visits = {}
        self._last_visit_pose = None
        self._last_visit_region = None
        self.just_reached = False        # frontier 도착 직후 1틱 True
        self._progress_target = None
        self._progress_distance = float('inf')
        self._progress_known = 0
        self._progress_since = 0.0
        self._progress_sample = -1e9

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

    @staticmethod
    def _region(xy):
        return (math.floor(xy[0] / .75), math.floor(xy[1] / .75))

    def observe_pose(self, pose):
        if self._last_visit_pose is not None and dist(pose, self._last_visit_pose) < .4:
            return
        region = self._region(pose)
        self._last_visit_pose = pose[:2]
        if region != self._last_visit_region:
            self.region_visits[region] = min(8, self.region_visits.get(region, 0) + 1)
            self._last_visit_region = region

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

    def update(self, pose, person_xy=None, furniture=(), now=None):
        """탐색 목표 반환 ((x,y) 또는 None=frontier 소진).

        person_xy가 있으면 그 주변 frontier의 점수를 깎는다 — 움직이는
        사람이 있는 쪽은 뒤로 미루고 반대 방향부터 탐색 (동선 겹침 회피).
        furniture(YOLO 가구 구역) 근처 frontier도 후순위 — 탁자 밑
        미탐색 셀을 굳이 기어들어가 밝히지 않는다.
        """
        self.observe_pose(pose)
        # A robot can orbit without being motionless. Require actual approach
        # or new observed area, rather than resetting the watchdog on motion.
        if self.current_target is not None and now is not None:
            distance = dist(pose, self.current_target)
            if self.current_target != self._progress_target:
                self._progress_target = self.current_target
                self._progress_distance = distance
                self._progress_since = now
                self._progress_known = int((~self.grid.unknown_mask()).sum())
            if distance < self._progress_distance - self.cfg.explore.progress_distance:
                self._progress_distance = distance
                self._progress_since = now
            if now - self._progress_sample >= 1.0:
                self._progress_sample = now
                known = int((~self.grid.unknown_mask()).sum())
                if known > self._progress_known + 80:
                    self._progress_known = known
                    self._progress_since = now
            if now - self._progress_since > self.cfg.explore.progress_timeout:
                print(f"[explore] no progress toward {self.current_target}; selecting another frontier")
                self.fail_current()
                self._progress_target = None
        # 관성: 유효한 기존 목표 유지
        if self.current_target is not None:
            if self.target_reached(pose):
                self.just_reached = True     # 새 시야 — 카메라 스캔 타이밍
                self.completed_targets.append(self.current_target)
                self.current_target = None
            elif not self.target_still_frontier():
                self.current_target = None
            else:
                return self.current_target

        best, best_score = None, -1.0
        for comp in self._clusters():
            rep = self._representative(comp)
            tolerance = self.cfg.explore.reach_tolerance
            def fresh(xy):
                return not self._blacklisted(xy) and dist(pose, xy) >= tolerance and not any(
                    dist(xy, previous) < tolerance for previous in self.completed_targets)
            if not fresh(rep):
                # A long boundary can have a centroid under the robot while its
                # ends are still unexplored. Keep eligible cells of that cluster.
                remaining = [cell for cell in comp if fresh(self.grid.grid_to_world(*cell))]
                if not remaining:
                    continue
                rep = self._representative(remaining)
            d = max(dist(pose, rep), 0.3)
            # 거리 지수 1.7: 선형(size/d)은 먼 대형 frontier가 지금 있는
            # 방의 작은 frontier를 이겨 "방에 들어갔다가 바로 나가는"
            # 왕복 동선을 만든다 — 가까운 방부터 끝내고 이동해야 한다
            score = len(comp) / (d ** 1.7)
            score /= 1 + 4 * self.region_visits.get(self._region(rep), 0)
            if person_xy is not None and dist(rep, person_xy) < 2.5:
                score *= 0.35        # 사람 있는 방향은 후순위
            if any(dist(rep, f) < 0.7 for f in furniture):
                score *= 0.25        # 가구 밑 frontier는 최후순위
            if score > best_score:
                best, best_score = rep, score
        self.current_target = best
        return best
