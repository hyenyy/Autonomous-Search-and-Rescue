"""경로 계획: A* (global) + pure pursuit 추종 + 반응형 회피 (local).

- A*: 팽창 지도 위 8-연결. 미지 셀은 통과 가능하되 비용 페널티
  (frontier 목표는 정의상 미지 영역 경계라서 완전 차단하면 계획 불가).
- 추종: lookahead 점을 향한 헤딩 P제어.
- 회피: 전방 섹터 최소거리로 감속/정지, 좌우 여유 비교로 회피 방향 편향.
"""
import heapq
import math

import numpy as np

from .geometry import bresenham, dist, wrap_angle

SQRT2 = math.sqrt(2.0)


def astar(blocked, unknown, start, goal, unknown_cost=2.5, max_pop=120000,
          penalty=None, penalty_cost=5.0):
    """blocked/unknown: [iy,ix] bool 배열. start/goal: (ix,iy).

    반환: [(ix,iy), ...] 경로 (start, goal 포함) 또는 None.
    goal이 지도 밖이면 경계로 클램프 (동결보다 접근이 낫다).
    max_pop: 탐색량 상한 — 대형 그리드에서 제어 루프 보호.
    penalty: 통과 가능하지만 비싼 셀 마스크 (예: 사람 주변) — 경로가
    사람의 현재 위치에서 먼 쪽으로 우회하게 만든다.
    """
    n_y, n_x = blocked.shape
    sx, sy = start
    gx, gy = goal
    # start도 경계 클램프 — 그리드 가장자리 포즈에서 IndexError 방지
    sx = min(max(sx, 1), n_x - 2)
    sy = min(max(sy, 1), n_y - 2)
    gx = min(max(gx, 1), n_x - 2)
    gy = min(max(gy, 1), n_y - 2)
    if blocked[sy, sx]:
        # 시작 셀이 팽창 영역에 물려 있으면 근처 자유 셀에서 출발
        s = _nearest_unblocked(blocked, start)
        if s is None:
            return None
        sx, sy = s
    if blocked[gy, gx]:
        g = _nearest_unblocked(blocked, goal)
        if g is None:
            return None
        gx, gy = g

    open_heap = [(0.0, (sx, sy))]
    g_cost = {(sx, sy): 0.0}
    came = {}
    neighbors = [(-1, -1, SQRT2), (0, -1, 1.0), (1, -1, SQRT2),
                 (-1, 0, 1.0), (1, 0, 1.0),
                 (-1, 1, SQRT2), (0, 1, 1.0), (1, 1, SQRT2)]
    pops = 0
    while open_heap:
        pops += 1
        if pops > max_pop:
            return None
        _, cur = heapq.heappop(open_heap)
        if cur == (gx, gy):
            path = [cur]
            while cur in came:
                cur = came[cur]
                path.append(cur)
            path.reverse()
            return path
        cx, cy = cur
        base = g_cost[cur]
        for dx, dy, step in neighbors:
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < n_x and 0 <= ny < n_y):
                continue
            if blocked[ny, nx]:
                continue
            # 대각 이동은 양옆 직교 셀도 비어 있어야 (코너 클리핑 방지)
            if dx != 0 and dy != 0 \
                    and (blocked[cy, nx] or blocked[ny, cx]):
                continue
            cost = step * (unknown_cost if unknown[ny, nx] else 1.0)
            if penalty is not None and penalty[ny, nx]:
                cost *= penalty_cost
            ng = base + cost
            key = (nx, ny)
            if ng < g_cost.get(key, float("inf")):
                g_cost[key] = ng
                came[key] = cur
                h = math.hypot(gx - nx, gy - ny)
                heapq.heappush(open_heap, (ng + h, key))
    return None


def _nearest_unblocked(blocked, cell, max_r=30):
    cx, cy = cell
    n_y, n_x = blocked.shape
    for r in range(1, max_r + 1):
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if max(abs(dx), abs(dy)) != r:
                    continue
                x, y = cx + dx, cy + dy
                if 0 <= x < n_x and 0 <= y < n_y and not blocked[y, x]:
                    return (x, y)
    return None


def los_clear(blocked, p0, p1):
    """셀 p0→p1 직선이 통과 가능한가 — 0.4셀 간격 세밀 샘플링.

    bresenham은 대각 스텝에서 직교 이웃을 안 밟아 벽 모서리를
    스치는 경로를 허용한다 (코너 클리핑) — 세밀 샘플링으로 방지.
    """
    x0, y0 = p0
    x1, y1 = p1
    n_y, n_x = blocked.shape
    length = math.hypot(x1 - x0, y1 - y0)
    steps = max(2, int(length / 0.4) + 1)
    for k in range(steps + 1):
        t = k / steps
        cx = int(round(x0 + (x1 - x0) * t))
        cy = int(round(y0 + (y1 - y0) * t))
        if not (0 <= cx < n_x and 0 <= cy < n_y) or blocked[cy, cx]:
            return False
        # 샘플 지점의 4-이웃 중 진행방향 양옆도 확인 (모서리 스침 방지)
        fx = x0 + (x1 - x0) * t - cx
        fy = y0 + (y1 - y0) * t - cy
        nx2 = cx + (1 if fx > 0.25 else -1 if fx < -0.25 else 0)
        ny2 = cy + (1 if fy > 0.25 else -1 if fy < -0.25 else 0)
        if 0 <= nx2 < n_x and blocked[cy, nx2]:
            return False
        if 0 <= ny2 < n_y and blocked[ny2, cx]:
            return False
    return True


def simplify_path(path, blocked):
    """line-of-sight 단축: 중간 경유지 제거."""
    if not path or len(path) <= 2:
        return path
    out = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1:
            if los_clear(blocked, path[i], path[j]):
                break
            j -= 1
        out.append(path[j])
        i = j
    return out


class Planner:
    """A* 계획 + 월드좌표 waypoint 관리."""

    def __init__(self, cfg, grid):
        self.cfg = cfg
        self.grid = grid
        self.waypoints = []          # [(x, y), ...] 월드 좌표
        self._goal = None
        self._last_plan_t = -1e9
        self._last_fail_t = -1e9     # 계획 실패 백오프용
        self.avoid_xy = None         # 동적 장애물(사람) 현재 위치 — 소프트 회피
        self.avoid_vel = None        # 사람 속도 벡터 (예측 캡슐 페널티용)
        self.avoid_territory = []    # 사람이 다녀간 자리들 (순찰 영토)
        self.furniture_xy = []       # YOLO 가구 구역 — 밑으로 안 파고들게
        self.last_plan_soft = False  # 마지막 경로가 팽창 완화(halo 통과)였나
        self.soft_spots = []         # 경로 중 halo(좁은 틈) 구간 월드 좌표
        # 좁은 틈 정렬 기동: (정렬점, 틈 중앙, 출구점) — 문·가구 틈은
        # 곡선으로 스치지 않고 "전방 정렬 → 회전 → 중앙 직진 관통"
        self.gap_maneuver = None
        self._gap_ax = (1.0, 0.0)
        self._gap_staged = False
        self._gap_half = 0.0

    @property
    def goal(self):
        return self._goal

    def plan_to(self, pose, goal_xy, now, force=False):
        """필요 시(주기/목표 변경) 재계획. 성공 여부 반환."""
        goal_changed = (self._goal is None
                        or dist(self._goal, goal_xy) > self.cfg.plan.goal_tolerance / 2)
        # 사람이 보이는 동안엔 더 자주 재계획 (예측 캡슐이 금방 낡는다)
        period = self.cfg.plan.replan_period if self.avoid_xy is None \
            else min(self.cfg.plan.replan_period, 0.8)
        if not force and not goal_changed \
                and now - self._last_plan_t < period \
                and self.waypoints:
            return True
        # 직전 계획 실패 직후엔 잠시 쉼 — 매 틱 전수 A* 재시도로
        # 제어 루프가 굶는 것을 방지 (계획 실패 백오프)
        if not force and not goal_changed \
                and now - self._last_fail_t < 0.5:
            return False
        self._last_plan_t = now
        self._goal = tuple(goal_xy)
        radius = self.cfg.robot.robot_radius + self.cfg.plan.inflate_margin
        # 이중 팽창: hard(물리 한계)=차단, halo(표준 여유 구간)=고비용
        # 통과 허용. halo를 항상 페널티로 두면 A*가 "우회로가 틈보다
        # 8배 이상 길 때"만 좁은 틈을 고른다 — 예전 2단계(표준 실패
        # 시에만 완화)는 먼 우회로가 존재하는 한 좁은 틈을 한 번도
        # 시도하지 않아 거실 가구 사이에서 왔던 길만 되돌아갔다.
        soft_blocked = self.grid.inflated_mask(radius)
        blocked = self.grid.inflated_mask(
            max(self.grid.res * 1.5,
                self.cfg.robot.robot_radius - self.grid.res * 0.5))
        halo = soft_blocked & ~blocked
        unknown = self.grid.unknown_mask()
        start = self.grid.world_to_grid(pose[0], pose[1])
        goal = self.grid.world_to_grid(*goal_xy)
        # A* 탐색 영역을 '관측된 영역 bounding box + 여유'로 제한.
        # 미지 셀은 통과 가능(비용 페널티)이라, 제한하지 않으면 실패 케이스에서
        # 그리드 전체(수십만 셀)의 미지 바다를 탐색하며 제어 루프가 멈춘다.
        known = ~unknown
        ys, xs = known.nonzero()
        outside = None
        if ys.size:
            m = int(1.0 / self.grid.res)          # 여유 1m
            y0 = max(0, ys.min() - m)
            y1 = min(self.grid.n, ys.max() + m + 1)
            x0 = max(0, xs.min() - m)
            x1 = min(self.grid.n, xs.max() + m + 1)
            outside = np.ones_like(blocked)
            outside[y0:y1, x0:x1] = False
            blocked = blocked | outside
        penalty = halo.copy()
        if outside is not None:
            penalty &= ~outside
        if self.furniture_xy:
            # 가구(탁자 다리 사이 등) 구역: 통과 불가는 아니지만 비싸게 —
            # 다른 길이 있으면 밑으로 파고들지 않는다
            g = self.grid
            r_c = max(2, int(0.45 / g.res))
            for fx, fy in self.furniture_xy:
                pcx, pcy = g.world_to_grid(fx, fy)
                y0f, y1f = max(0, pcy - r_c), min(g.n, pcy + r_c + 1)
                x0f, x1f = max(0, pcx - r_c), min(g.n, pcx + r_c + 1)
                if y0f >= y1f or x0f >= x1f:
                    continue
                yy, xx = np.ogrid[y0f:y1f, x0f:x1f]
                penalty[y0f:y1f, x0f:x1f] |= \
                    (xx - pcx) ** 2 + (yy - pcy) ** 2 <= r_c * r_c
        if self.avoid_xy is not None:
            # 사람의 '예측 쓸기 경로' 캡슐: 현재 위치 + 속도×[0, 2.5s]
            # 를 따라 원을 찍는다. 등 뒤(-이동 반대쪽)는 페널티가 없으므로
            # 경로가 자연스럽게 "떠나는 사람 뒤로 건너기"가 된다.
            if penalty is None:
                penalty = np.zeros_like(blocked)
            g = self.grid
            px, py = self.avoid_xy
            vx, vy = self.avoid_vel or (0.0, 0.0)
            speed = math.hypot(vx, vy)
            if speed > 0.12:
                horizon = 2.5
                samples = [(px + vx * t, py + vy * t)
                           for t in np.linspace(0.0, horizon, 7)]
                r_m = 0.55
            else:               # 거의 정지: 그냥 주변 회피
                samples = [(px, py)]
                r_m = 0.9
            # 순찰 영토(다녀간 자리들)도 페널티 — 순찰 축과 나란히
            # 이동해야 할 때 측면 차선을 잡게 한다 (반경은 작게)
            samples.extend(self.avoid_territory[::2])
            r_c = max(2, int(r_m / g.res))
            for sx_, sy_ in samples:
                pcx, pcy = g.world_to_grid(sx_, sy_)
                y0, y1 = max(0, pcy - r_c), min(g.n, pcy + r_c + 1)
                x0, x1 = max(0, pcx - r_c), min(g.n, pcx + r_c + 1)
                if y0 >= y1 or x0 >= x1:
                    continue
                yy, xx = np.ogrid[y0:y1, x0:x1]
                penalty[y0:y1, x0:x1] |= \
                    (xx - pcx) ** 2 + (yy - pcy) ** 2 <= r_c * r_c
        path = astar(blocked, unknown, start, goal,
                     unknown_cost=self.cfg.plan.unknown_cost,
                     penalty=penalty, penalty_cost=8.0)
        if path is None:
            self.last_plan_soft = False
            self.waypoints = []
            self._last_fail_t = now
            return False
        # 경로가 halo(좁은 틈)를 지나면 '소프트 경로'. 완화는 경로 전체가
        # 아니라 좁은 구간(soft_spots) 근처에서만 — 전 구간 완화는 열린
        # 공간에서도 저속·저마진이 걸려 느리고 위험해진다.
        g = self.grid
        self.soft_spots = [
            g.grid_to_world(cx, cy) for cx, cy in path[::2]
            if 0 <= cy < g.n and 0 <= cx < g.n and halo[cy, cx]][:80]
        self.last_plan_soft = bool(self.soft_spots)
        # 첫 번째 좁은 틈에 대한 정렬 관통 기동 계산: 곡선 추종으로
        # 비스듬히 스치지 말고 "틈 전방 정렬점 → 제자리 회전 → 중앙
        # 직진 관통" (문 앞 버벅임 제거)
        self.gap_maneuver = None
        in_halo = [0 <= cy < g.n and 0 <= cx < g.n and bool(halo[cy, cx])
                   for cx, cy in path]
        if any(in_halo):
            s_i = in_halo.index(True)
            e_i = s_i
            while e_i + 1 < len(path) and in_halo[e_i + 1]:
                e_i += 1
            entry = g.grid_to_world(*path[s_i])
            exit_ = g.grid_to_world(*path[e_i])
            ctr = ((entry[0] + exit_[0]) / 2, (entry[1] + exit_[1]) / 2)
            dx_, dy_ = exit_[0] - entry[0], exit_[1] - entry[1]
            l_ = math.hypot(dx_, dy_)
            if l_ < 1e-6:      # 한 셀짜리 틈: 축은 앞뒤 경로에서
                p0 = g.grid_to_world(*path[max(0, s_i - 1)])
                p1 = g.grid_to_world(*path[min(len(path) - 1, e_i + 1)])
                dx_, dy_ = p1[0] - p0[0], p1[1] - p0[1]
                l_ = math.hypot(dx_, dy_) or 1.0
            ax = (dx_ / l_, dy_ / l_)
            half = l_ / 2
            stg = (ctr[0] - ax[0] * (half + 0.40),
                   ctr[1] - ax[1] * (half + 0.40))
            lv = (ctr[0] + ax[0] * (half + 0.35),
                  ctr[1] + ax[1] * (half + 0.35))
            proj = (pose[0] - ctr[0]) * ax[0] + (pose[1] - ctr[1]) * ax[1]
            # 짧은 조임 구간(진짜 문/틈)만 — 벽을 낀 긴 halo 구간(좁은
            # 방 내부 이동 전체 등)에 이 축 계산을 적용하면 축·중앙이
            # 엉뚱해져 문설주로 돌진한다. 정렬·출구점이 차단 셀이어도
            # 취소 (일반 추종 + 완화가 대신 처리).
            si, sj = g.world_to_grid(*stg)
            li, lj = g.world_to_grid(*lv)
            pts_free = (0 <= sj < g.n and 0 <= si < g.n
                        and 0 <= lj < g.n and 0 <= li < g.n
                        and not blocked[sj, si] and not blocked[lj, li])
            if l_ <= 0.55 and pts_free and proj <= half + 0.05:
                self.gap_maneuver = (stg, ctr, lv)
                self._gap_ax = ax
                self._gap_half = half
                # 이미 틈 어귀/안이면 정렬 단계 생략, 곧장 관통
                self._gap_staged = proj > -(half + 0.25)
        # LOS 단축 시 사람 캡슐(penalty)도 벽 취급 — 지름길이 캡슐을
        # 관통하면 A*의 우회가 무의미해진다
        los_blocked = blocked | penalty if penalty is not None else blocked
        path = simplify_path(path, los_blocked)
        self.waypoints = [self.grid.grid_to_world(ix, iy) for ix, iy in path]
        return True

    def follow(self, pose):
        """(v, w) 추종 명령. 경로 없으면 (0,0)."""
        cfg = self.cfg
        if not self.waypoints:
            return 0.0, 0.0
        # 도달한 waypoint 제거 (임계 작게 — 코너 커팅으로 벽을 스치지 않게)
        while len(self.waypoints) > 1 \
                and dist(pose, self.waypoints[0]) < cfg.plan.lookahead * 0.45:
            self.waypoints.pop(0)
        # 좁은 틈 정렬 관통 기동이 활성이고 틈이 가까우면 pure pursuit
        # 대신 3점 기동: 정렬점까지 → (제자리 회전) → 출구점까지 직진.
        # lookahead 곡선 추종은 문설주를 비스듬히 스치며 버벅인다.
        if self.gap_maneuver is not None:
            stg, ctr, lv = self.gap_maneuver
            ax = self._gap_ax
            proj = (pose[0] - ctr[0]) * ax[0] + (pose[1] - ctr[1]) * ax[1]
            if proj > self._gap_half + 0.28 or dist(pose, lv) < 0.13:
                self.gap_maneuver = None         # 통과 완료
            elif dist(pose, ctr) < 1.3:
                if not self._gap_staged:
                    target = stg
                    if dist(pose, stg) < 0.13:
                        self._gap_staged = True
                        target = lv
                else:
                    target = lv
                heading = math.atan2(target[1] - pose[1],
                                     target[0] - pose[0])
                err = wrap_angle(heading - pose[2])
                w = max(-cfg.robot.max_w,
                        min(cfg.robot.max_w, cfg.plan.k_heading * err))
                # 정렬 전(오차 큼)엔 제자리 회전, 정렬되면 직진
                v = cfg.robot.max_v \
                    * max(0.0, 1.0 - abs(err) / (math.pi / 3))
                if abs(err) < 0.6:
                    w *= max(0.35, v / cfg.robot.max_v)
                return v, w
        # lookahead 점 선택
        target = self.waypoints[0]
        for wp in self.waypoints:
            target = wp
            if dist(pose, wp) >= cfg.plan.lookahead:
                break
        heading = math.atan2(target[1] - pose[1], target[0] - pose[0])
        err = wrap_angle(heading - pose[2])
        if abs(err) < .035:  # Ignore insignificant heading noise during path following.
            err = 0.0
        w = max(-cfg.robot.max_w, min(cfg.robot.max_w, cfg.plan.k_heading * err))
        # 오차가 크면 전진 억제 (제자리 회전 우선)
        v = cfg.robot.max_v * max(0.0, 1.0 - abs(err) / (math.pi / 2))
        # 저속 트위칭 방지 (RPP Eq.7의 원리): 헤딩오차가 작을 땐 조향을
        # 속도 비율로 스케일 — 감속 구간에서 회전이 지배하지 않게.
        # 큰 오차(제자리 선회가 목적)일 땐 스케일하지 않는다.
        if abs(err) < 0.6:
            w *= max(0.35, v / cfg.robot.max_v)
        return v, w

    def distance_to_goal(self, pose):
        return dist(pose, self._goal) if self._goal else float("inf")


class LocalAvoider:
    """LiDAR 기반 반응형 회피. follow 출력 (v,w)를 안전하게 보정.

    동적 장애물(사람)은 로봇보다 빠를 수 있다는 전제로 별도 규칙:
    - 0.95m 안에서 접근 중 + 위협이 진행방향 안 → 정지 양보
    - 위협이 진행방향 밖(옆/뒤에서 접근) → 정지 대신 전속 이탈
    - 0.5m 안 → 위협 반대쪽으로 적극 탈출
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.blocked_since = None    # 정지 상태 시작 시각 (양보 타이머)
        self._dyn_hist = []          # [(t, d_min)] 동적 장애물 거리 이력
        self._yield_since = None
        self._commit_until = -1e9    # "지금 건너자" 커밋 윈도우 만료 시각
        self.last_mode = "clear"     # 디버그: 마지막으로 탄 분기
        # 초근접 실명 대비: 위협이 LiDAR 최소거리 안으로 들어와 안 보이게
        # 되면, 잠시 직전 탈출 명령을 유지한다
        self._blind_until = -1e9
        self._blind_cmd = (0.0, 0.0)
        self._last_dyn_min = None
        self._stop_turn = None       # 정지-회전 방향 래치 (좌우 진동 방지)
        self._front_hist = []        # 전방거리 중앙값 필터 (채터링 컷)
        self._stopped = False        # 정지 히스테리시스 상태
        self.relax_until = -1e9      # 이 시각까지 안전 마진 일시 완화
                                     # (갇힘 탈출 직후·소프트 경로 추종 중)
        self._esc_latch = None       # (world_ang, until) 탈출 방향 래치

    def _dyn_trend(self, now, d_min):
        """(approaching, receding): 0.6s 창에서의 거리 추세."""
        self._dyn_hist.append((now, d_min))
        while self._dyn_hist and self._dyn_hist[0][0] < now - 0.6:
            self._dyn_hist.pop(0)
        if len(self._dyn_hist) < 3:
            return False, False
        d0 = self._dyn_hist[0][1]
        return d_min < d0 - 0.015, d_min > d0 + 0.02

    @staticmethod
    def _miss_lat(pose, person_xy, person_vel):
        """사람 진행 선로 기준 내 측방 이격 (m). 판정 불가/뒤쪽이면 None."""
        if pose is None or person_xy is None or person_vel is None:
            return None
        vx_, vy_ = person_vel
        sp = math.hypot(vx_, vy_)
        if sp < 0.1:
            return None
        dx_ = pose[0] - person_xy[0]
        dy_ = pose[1] - person_xy[1]
        ux, uy = vx_ / sp, vy_ / sp
        proj = dx_ * ux + dy_ * uy
        if proj < -0.1:
            return None
        return math.hypot(dx_ - proj * ux, dy_ - proj * uy)

    @staticmethod
    def _safe_pass(pose, person_xy, person_vel):
        """상대의 진행 선로가 나를 충분히 비껴가는가 (miss distance 판정).

        비껴가는 궤적이면 양보 없이 지나간다 — 복도에서 마주칠 때마다
        정지하면 영원히 통과하지 못한다. 판정 불가(속도 미상)면 False
        (보수적으로 양보).
        """
        if pose is None or person_xy is None or person_vel is None:
            return False
        vx_, vy_ = person_vel
        sp = math.hypot(vx_, vy_)
        if sp < 0.1:
            return False
        dx_ = pose[0] - person_xy[0]
        dy_ = pose[1] - person_xy[1]
        ux, uy = vx_ / sp, vy_ / sp
        proj = dx_ * ux + dy_ * uy
        if proj < -0.1:          # 로봇이 상대 진행 방향의 '뒤' → 무관
            return True
        lat = math.hypot(dx_ - proj * ux, dy_ - proj * uy)
        return lat > 0.5         # 측방 여유 0.5m 이상 → 비껴감

    @staticmethod
    def _clearance(a, r, phi, halfwidth=0.5):
        """로봇 프레임 방향 phi 섹터의 최소 LiDAR 거리."""
        sel = np.abs(wrap_angle(a - phi)) < halfwidth
        if not sel.any():
            return 10.0
        return float(r[sel].min())

    def _cmd_toward(self, world_ang, c, pose):
        """월드 방향 world_ang로의 (v,w) — 전방이면 전진, 후방이면 후진."""
        cfg = self.cfg
        err = wrap_angle(world_ang - pose[2])
        speed = cfg.robot.max_v * max(0.3, min(1.0, (c - 0.2) / 0.4))
        if abs(err) <= math.pi / 2:
            v_e = speed
            w_e = max(-cfg.robot.max_w, min(cfg.robot.max_w, 2.5 * err))
        else:
            err_b = wrap_angle(world_ang + math.pi - pose[2])
            v_e = -speed * 0.85
            w_e = max(-cfg.robot.max_w, min(cfg.robot.max_w, 2.5 * err_b))
        return v_e, w_e

    def _escape_cmd(self, d_min, b, pose, person_xy, person_vel, a, r,
                    now=0.0):
        """탈출 명령 계산 — 후보 방향을 LiDAR 여유로 점수화해 선택.

        후보: ① 사람 진행 선로에서 수직 이탈, ② 선로를 따라 사람 반대쪽
        도주(좁은 복도의 후퇴), ③ 그냥 후진. 벽에 막힌 방향은 버린다.
        속도 정보가 없으면 bearing 휴리스틱 폴백.
        """
        cfg = self.cfg
        if person_xy is not None and person_vel is not None and pose is not None:
            vx_, vy_ = person_vel
            sp = math.hypot(vx_, vy_)
            if sp > 0.1:
                # 방향 래치(1s): 매 틱 재선정은 여유 요동으로 전진↔후진이
                # 진동(디더링)해 탈출이 제자리걸음이 된다
                if self._esc_latch is not None and now < self._esc_latch[1]:
                    ang0 = self._esc_latch[0]
                    c0 = self._clearance(a, r, wrap_angle(ang0 - pose[2]))
                    if c0 > 0.28:
                        return self._cmd_toward(ang0, c0, pose)
                dx_ = pose[0] - person_xy[0]
                dy_ = pose[1] - person_xy[1]
                ux, uy = vx_ / sp, vy_ / sp
                proj = dx_ * ux + dy_ * uy
                lx, ly = dx_ - proj * ux, dy_ - proj * uy   # 선로 수직 성분
                if math.hypot(lx, ly) < 0.05:               # 정확히 선로 위
                    lx, ly = -uy, ux
                cands = []
                # ① 수직 이탈 (선호) — 반대쪽 수직(선로 횡단)도 후보로:
                #    이탈 쪽이 벽이면 건너편이 유일한 활로다 (순찰 반환점
                #    코너에 몰리는 케이스)
                cands.append((math.atan2(ly, lx), 0.45))
                cands.append((math.atan2(-ly, -lx), 0.05))
                # ② 선로 따라 도주: 사람 이동 방향으로 앞서 있으면 그쪽으로.
                #    단, 후진으로만 가능한 도주는 최고 0.85×max_v — 사람
                #    (0.2m/s)보다 느려 반드시 따라잡히는 필패 수이므로
                #    강한 감점 (수직 이탈이 조금이라도 열려 있으면 그쪽)
                flee = (ux, uy) if proj >= 0 else (-ux, -uy)
                flee_ang = math.atan2(flee[1], flee[0])
                fwd_ok = abs(wrap_angle(flee_ang - pose[2])) <= math.pi / 2
                cands.append((flee_ang, 0.15 if fwd_ok else -1.2))
                # ③ 후진 (현 헤딩 반대) — 로봇을 마주보는 사람에게 이건
                #    ②의 후진 도주와 같은 필패 기동이므로 강한 감점.
                #    (감점 없던 시절: 후방이 트이면 수직 이탈을 이겨서
                #    복도에서 0.18 vs 0.2 후진 추격전 → 추돌)
                cands.append((pose[2] + math.pi, -0.8))
                best, best_score = None, -1e9
                for world_ang, bonus in cands:
                    phi = wrap_angle(world_ang - pose[2])
                    c = self._clearance(a, r, phi)
                    if c < 0.3:                 # 벽/물체에 막힘
                        continue
                    # 여유 점수 상한 1.2: '넓게 트였다'는 이유만으로 나쁜
                    # 방향(후진 도주)이 좋은 방향(수직 이탈)을 이기지 못하게
                    score = min(c, 1.2) + bonus
                    if score > best_score:
                        best, best_score = (world_ang, c), score
                if best is not None:
                    world_ang, c = best
                    self._esc_latch = (world_ang, now + 1.0)
                    return self._cmd_toward(world_ang, c, pose)
                return 0.0, 0.0                 # 사방이 막힘: 정지가 최선
        # 폴백: bearing 휴리스틱 (전진 선택 시 전방 여유 확인)
        if abs(b) < 0.45:
            v_e = -cfg.robot.max_v * 0.7
        elif abs(b) < math.pi / 2:
            fwd = self._clearance(a, r, 0.0)
            v_e = cfg.robot.max_v * 0.9 if fwd > 0.35 else -cfg.robot.max_v * 0.6
        else:
            v_e = cfg.robot.max_v
        w_e = -math.copysign(cfg.robot.max_w * 0.5, b)
        return v_e, w_e

    def apply(self, v, w, angles, ranges, now, dynamic=None,
              pose=None, person_xy=None, person_vel=None):
        """반환: (v, w, blocked_long) — blocked_long=True면 오래 막힘(재계획 신호)."""
        cfg = self.cfg
        angles = np.asarray(angles)
        ranges = np.asarray(ranges)
        valid = np.isfinite(ranges) & (ranges > cfg.lidar.min_range)
        if not valid.any():
            return v, w, False
        a, r = angles[valid], ranges[valid]

        blocked_long = False
        dyn_action = None      # 동적 규칙이 명령을 제안했는가

        # 0) 동적 장애물 규칙 — (v, w)를 '제안'만 한다. 조기 반환 금지:
        #    정적 안전 클램프(아래)가 모든 제안에 최종 적용되어야
        #    커밋/스프린트가 벽을 뚫고 가는 사고가 원천 차단된다.
        if dynamic is not None:
            dyn = np.asarray(dynamic)[valid]
            if dyn.any():
                rd, ad = r[dyn], a[dyn]
                j = int(np.argmin(rd))
                d_min, b = float(rd[j]), float(ad[j])
                self._last_dyn_min = (now, d_min, b)
                approaching, receding = self._dyn_trend(now, d_min)
                committing = now < self._commit_until
                # 양보 중 상대가 충분한 거리에서 멀어지기 시작 → 커밋
                # (커밋은 '전방 횡단'용 — 뒤쪽 위협에 커밋하면 추격당한다)
                if self._yield_since is not None and receding and d_min > 0.6 \
                        and abs(b) <= math.pi / 2:
                    self._commit_until = now + cfg.plan.dyn_commit_time
                    self._yield_since = None
                    committing = True
                # 커밋 중이라도 위협이 정면에서 다시 접근하면 커밋 중단
                if committing and approaching and d_min < 0.85 \
                        and abs(b) < 0.7:
                    self._commit_until = -1e9
                    committing = False
                escape_d = cfg.plan.dyn_commit_escape if committing \
                    else cfg.plan.dyn_escape_dist
                passing = self._safe_pass(pose, person_xy, person_vel)
                if d_min < escape_d and passing and d_min > 0.3:
                    # 비껴가는 궤적 — 탈출로 진행을 버리지 않고 통과
                    self.last_mode = dyn_action = "pass"
                elif d_min < escape_d:
                    self.last_mode = dyn_action = "escape"
                    v, w = self._escape_cmd(d_min, b, pose,
                                            person_xy, person_vel, a, r, now)
                    self._yield_since = self._yield_since or now
                    self._commit_until = -1e9             # 커밋 취소
                    if d_min < 0.4:
                        # 초근접: 위협이 LiDAR 사각으로 들어가면 잠시
                        # '정지+회전'만 유지 (과거 후진 명령을 재생하면
                        # 뒤에서 오는 위협에게 후진하는 사고가 난다)
                        self._blind_until = now + 0.4
                        self._blind_cmd = (0.0, w)
                elif committing:
                    self.last_mode = dyn_action = "commit"
                    v = max(v, cfg.robot.max_v * 0.9)
                elif d_min < cfg.plan.dyn_yield_dist and approaching \
                        and not passing:
                    lat_ = self._miss_lat(pose, person_xy, person_vel)
                    # 서 있어도 안전한 자리인가: 선로 이격 0.35m+ 확인
                    # 됐거나 아직 충분히 멀 때만. 속도 미상(반환점 등)
                    # 이면 근접 정지 양보 금지 — 그 자리가 선로 위일
                    # 수 있다.
                    safe_spot = (lat_ is not None and lat_ >= 0.35) \
                        or d_min >= 0.7
                    if abs(b) < 1.05 and safe_spot:
                        # 위협이 진행방향 안 → 양보 (멀면 서행, 가까우면 정지)
                        self.last_mode = dyn_action = "yield"
                        if self._yield_since is None:
                            self._yield_since = now
                        long_wait = now - self._yield_since \
                            > cfg.plan.person_wait * 2
                        if long_wait:
                            self._yield_since = now   # 에지 트리거 재무장
                            blocked_long = True
                        if d_min > 0.7:
                            v, w = v * 0.35, w
                        else:
                            v, w = 0.0, 0.0
                    else:
                        # 옆/뒤에서 접근 → 멈추면 치인다. 직진 전속(구
                        # 방식)은 순찰선과 나란할 때 0.21 vs 0.2 무한
                        # 추격전 — 수직 이탈 우선의 탈출 계산을 쓴다
                        self.last_mode = dyn_action = "sprint"
                        self._yield_since = None
                        v, w = self._escape_cmd(d_min, b, pose,
                                                person_xy, person_vel, a, r, now)
                        if abs(v) < 0.05:
                            v = cfg.robot.max_v      # 폴백: 그냥 내빼기
                else:
                    self.last_mode = "watch"
            else:
                # 동적 장애물이 갑자기 안 보임: 초근접 실명일 수 있다
                if now < self._blind_until:
                    self.last_mode = dyn_action = "blind"
                    v, w = self._blind_cmd
                else:
                    self.last_mode = "clear"
                    self._dyn_hist.clear()
                    self._yield_since = None

        # 1) 정적 안전 클램프 — 동적 제안 포함 모든 (v, w)에 최종 적용
        # 빔을 (전방거리, 측방오프셋)으로 분해 — '충돌 코스에 있는 물체'만
        # 정지·감속 사유가 된다. 각도 부채꼴 방식은 문설주(비스듬 30~40도,
        # 측방으론 로봇 폭 밖)를 정면 장애물로 오인해 문 통과를 막았다.
        forward = r * np.cos(a)
        lateral = r * np.sin(a)
        rr = cfg.robot.robot_radius
        # 갇힘 탈출 직후/소프트(팽창 완화) 경로 추종 중에는 마진을 일시
        # 축소 — 표준 마진으로는 물리적으로 지나갈 수 있는 틈도 정지
        # 대상이라, 계획이 허용한 틈을 회피가 도로 막는 모순이 생긴다.
        # 접촉 한계(danger는 로봇 반경+3cm 아래로는 안 내려감)는 유지.
        relaxed = now < self.relax_until
        if dynamic is not None and np.asarray(dynamic)[valid].any():
            relaxed = False    # 사람이 보이는 동안엔 마진 완화 금지 —
                               # 완화된 danger가 사람 접촉 비상까지 늦춘다
        # 완화 바닥값: 중앙값 필터 지연(~0.2s) 동안의 이동분을 견딜 표면
        # 여유(4cm+)는 남긴다 — 이보다 깎으면 좁은 공간에서 실접촉이 난다
        stop_d = max(rr + 0.05, cfg.plan.stop_dist * 0.7) if relaxed \
            else cfg.plan.stop_dist
        danger_d = min(cfg.plan.danger_dist, rr + 0.04) if relaxed \
            else cfg.plan.danger_dist
        corr_m = 0.02 if relaxed else cfg.plan.corridor_margin
        lat_m = 0.03 if relaxed else 0.05

        # 1a) 비상: 반경 danger_dist 안 + 로봇 폭에 실제로 걸치는 물체.
        #     단, 전진하지 않는 제자리 회전은 원형 로봇에겐 접촉 불가 —
        #     정지 회전(SPIN 등)까지 방향을 뒤집으면 스핀이 영원히 안
        #     끝나는 데드락이 된다 (동적 위협이 없을 때만 면제).
        touching = (r < danger_d) & (np.abs(lateral) < rr + lat_m)
        if touching.any() and (abs(v) > 0.02 or dyn_action is not None):
            j = int(np.argmin(np.where(touching, r, np.inf)))
            a_min = a[j]
            if abs(a_min) < math.pi / 2:      # 앞쪽 → 후진하며 회피
                rear = self._clearance(a, r, math.pi)
                v_e = -0.08 if rear > 0.25 else 0.0
            else:                              # 뒤쪽 → 전진해서 이탈
                fwd = self._clearance(a, r, 0.0)
                v_e = cfg.robot.max_v * 0.7 if fwd > 0.3 else 0.0
            w_e = -math.copysign(cfg.robot.max_w * 0.4, a_min)
            return v_e, w_e, blocked_long

        # 전방 코리도: 진행 폭(robot_radius+corridor_margin) 안의 최소 전방거리
        corridor = (forward > 0.02) \
            & (np.abs(lateral) < rr + corr_m)
        front_raw = float(forward[corridor].min()) if corridor.any() \
            else float("inf")
        # 3샘플 중앙값 — LiDAR 노이즈 한 방으로 정지/재개가 튀는 것 방지
        self._front_hist.append(front_raw)
        if len(self._front_hist) > 3:
            self._front_hist.pop(0)
        front_min = sorted(self._front_hist)[len(self._front_hist) // 2]

        left = (a > 0.15) & (a < 1.4)
        right = (a < -0.15) & (a > -1.4)
        left_min = r[left].min() if left.any() else cfg.lidar.max_range
        right_min = r[right].min() if right.any() else cfg.lidar.max_range

        resume_dist = stop_d * 1.3
        if front_min < stop_d \
                or (self._stopped and front_min < resume_dist):
            self._stopped = True
            # 전진 금지 (후진 탈출은 허용). 동적 제안이 없을 때만
            # 벽 회피 회전을 덧씌운다 (탈출 조향 존중).
            v = min(v, 0.0)
            if dyn_action is None:
                # 회전 방향 래치: 매 틱 좌우 여유로 재선택하면 대칭
                # 상황에서 좌↔우 진동하며 시간·동선을 낭비한다
                if self._stop_turn is None:
                    self._stop_turn = 1.0 if left_min > right_min else -1.0
                w = self._stop_turn * cfg.robot.max_w * 0.5
                if self.blocked_since is None:
                    self.blocked_since = now
                elif now - self.blocked_since > cfg.plan.static_wait:
                    # 정적 막힘은 빠르게 우회 재계획 (사람 양보와 별도 타이머)
                    blocked_long = True
                    self.blocked_since = now
        else:
            self._stopped = False
            self._stop_turn = None       # 전방이 뚫리면 래치 해제
            if dyn_action is None:
                self.blocked_since = None
            if front_min < cfg.plan.slow_dist and v > 0:
                scale = (front_min - stop_d) / \
                        (cfg.plan.slow_dist - stop_d)
                v *= max(0.15, scale)
            # 틈 중앙 조준: 좁은 통로(한쪽이 0.5m 이내)에서는 전방이
            # 뚫려 있어도 항상 빈 공간 중앙을 향해 조향 — 문·가구 틈을
            # 중앙 정렬로 통과한다
            if dyn_action is None and v > 0 \
                    and min(left_min, right_min) < 0.5:
                delta = float(left_min - right_min)
                if abs(delta) < 0.03:
                    delta = 0.0          # 데드밴드 3cm (노이즈 부호 플립 컷)
                # 좁을수록 게인을 낮춰 경로추종과의 조향 싸움 방지
                gap = float(left_min + right_min)
                k_c = 1.2 * max(0.35, min(1.0, (gap - 0.35) / 0.55))
                bias = max(-0.55, min(0.55, k_c * delta))
                w = max(-cfg.robot.max_w, min(cfg.robot.max_w, w + bias))
        # 완화 모드 속도 상한: 마진 4~5cm로 좁은 틈을 지나는 중 —
        # 순항 속도로 코너를 자르면 문설주를 스친다. 저속 통과 강제.
        if relaxed and v > 0.09:
            v = 0.09
        return v, w, blocked_long
