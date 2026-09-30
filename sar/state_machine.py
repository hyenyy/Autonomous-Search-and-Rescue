"""미션 상태 기계: SPIN → EXPLORE → GOTO_TARGET → RETURN → DONE (+RECOVER).

모든 상태에서 LocalAvoider 회피는 항상 적용된다.
main 루프(오프라인 sim / Webots controller)는 매 스텝:
    mission.step(now, pose, angles, ranges, camera_img) → (v, w, info)
만 호출하면 된다. 지도 갱신은 main 루프 책임.
"""
import math

from .breadcrumbs import Breadcrumbs
from .detection import (TargetDetector, estimate_target_position,
                        estimate_target_position_mono)
from .energy import EnergyManager
from .exploration import FrontierExplorer
from .geometry import dist, wrap_angle
from .planning import LocalAvoider, Planner
from .semantic_obstacles import TableObstacleMapper


class Mission:
    # 상태 이름 상수
    SPIN = "SPIN"                # 제자리 1회전 스캔
    EXPLORE = "EXPLORE"
    SEEK = "SEEK"                # 목표 보임 but 거리 미확정 → bearing 추종
    GOTO_TARGET = "GOTO_TARGET"
    RETURN = "RETURN"            # A* 복귀 (실패 시 crumb 폴백)
    RECOVER = "RECOVER"
    DONE = "DONE"

    def __init__(self, cfg, grid):
        self.cfg = cfg
        self.grid = grid
        self.planner = Planner(cfg, grid)
        self.explorer = FrontierExplorer(cfg, grid)
        self.detector = TargetDetector(cfg)
        self.crumbs = Breadcrumbs(cfg)
        self.energy = EnergyManager(cfg)
        self.avoider = LocalAvoider(cfg)
        self.table_mapper = TableObstacleMapper(cfg, grid)

        self.start_xy = (cfg.mission.start_x, cfg.mission.start_y)
        self.state = self.SPIN
        self.state_since = 0.0
        self.spin_accum = 0.0
        self._last_theta = None
        self._spin_return_state = self.EXPLORE
        self.last_spin_t = 0.0
        self._initial_spin_done = False   # 최초 1회만 360도 (지도 시드)
        self._spin_phase = "left"         # 부분 스윕 상태
        self._force_full_spin = False     # frontier 소진 시 마지막 360도
        self._final_spin_done = False

        self.target_est = None       # (x, y) 현재 추적 중인 목표 추정
        self._est_hist = []          # 최근 채택된 추정들 (안정성 게이트용)
        self.visited_targets = []    # 방문 완료한 목표 위치들
        self._seek_block_until = -1e9
        self.target_found = False    # '현재' 목표 확정 여부
        self.target_reached = False  # '모든' 목표 방문 완료 여부
        self.return_reason = None    # MISSION_COMPLETE/LOW_BATTERY/...
        self.table_cells_added = 0   # 카메라+LiDAR 가상 테이블 장애물 셀

        # stuck 감지
        self._pose_hist = []         # [(t, x, y)]
        self._recover_phase = None   # ("backup"|"spin", 시작시각)
        self._recover_after = self.EXPLORE
        self._stuck_count = 0

        # crumb 복귀 모드 플래그
        self._crumb_mode = False
        self._crumb_since = None     # crumb 폴백 진입 시각 (주기적 A* 재시도)
        self.start_time = None
        self._last_map_t = -1e9      # 지도 갱신 주기 관리 (Mission이 직접)
        self._amnesty_count = 0      # frontier 블랙리스트 사면 횟수
        self._goto_fail_since = None # GOTO 계획 연속 실패 시각
        # 명령 평활화 상태 (문 앞 떨림 제거)
        self._v_prev = 0.0
        self._w_prev = 0.0
        self._last_step_t = None

    # ── 내부 유틸 ─────────────────────────────────────────────
    def _enter(self, state, now):
        self.state = state
        self.state_since = now

    def _update_stuck(self, now, pose):
        self._pose_hist.append((now, pose[0], pose[1]))
        horizon = self.cfg.mission.stuck_time
        while self._pose_hist and self._pose_hist[0][0] < now - horizon - 0.5:
            self._pose_hist.pop(0)
        if not self._pose_hist or now - self._pose_hist[0][0] < horizon:
            return False
        t0, x0, y0 = self._pose_hist[0]
        moved = max(math.hypot(x - x0, y - y0) for _, x, y in self._pose_hist)
        return moved < self.cfg.mission.stuck_dist

    def _update_target_estimate(self, pose, angles, ranges):
        """목표 위치 추정: LiDAR 거리(일관성 통과 시) > 단안(blob 크기).

        - LiDAR est는 "그 거리라면 blob이 이 크기여야 한다" 검사를 통과해야
          채택 (목표가 사거리 밖/스캔 평면 아래일 때 배경 벽 오인 방지)
        - 사과처럼 낮은 물체는 LiDAR에 안 잡히므로 단안 추정이 주력
        - 확정(target_found)은 최근 추정 K개가 서로 뭉쳐 있을 때만
          (_estimate_stable) — 스치는 오추정 한 방으로 확정되지 않게
        """
        det = self.detector.last
        if det is None:
            return
        dcfg = self.cfg.detection
        # 확정용 형태 게이트: 온전한 원형(가림 해소)일 때만 위치 추정 채택.
        # 가려진 반달 사과는 단안 거리가 2배로 튀므로 SEEK로 더 접근시킨다.
        if not det.v_clipped and not (0.55 <= det.aspect <= 1.7):
            return
        # 채움비 확정 게이트: 사과는 민무늬 원(≥0.6), 라벨/그래픽 있는
        # 캔·병은 마스크에 구멍이 나 채움비가 낮다 (빨간 콜라캔 오인 방지)
        if det.fill < 0.6:
            return
        accepted = None

        est = estimate_target_position(pose, det.bearing, angles, ranges)
        acc_d = None
        if est is not None:
            x, y, d = est
            if 0.25 <= d <= self.cfg.lidar.max_range * 0.85:
                expected_ang = 2 * math.atan2(dcfg.target_width / 2, d)
                ratio = det.ang_width / max(expected_ang, 1e-6)
                if dcfg.size_ratio_lo <= ratio <= dcfg.size_ratio_hi:
                    accepted, acc_d = (x, y), d
        if accepted is None:
            mono = estimate_target_position_mono(pose, det, dcfg.target_width)
            if mono is not None:
                accepted, acc_d = (mono[0], mono[1]), mono[2]
        if accepted is None:
            return
        # 이미 방문한 목표 주변 추정은 버린다.
        # 빨간 캔 같은 디코이는 이 단계 전에 YOLO apple(47) 필터에서 제거된다.
        r_v = self.cfg.mission.visited_radius
        if any(dist(accepted, vt) < r_v for vt in self.visited_targets):
            return

        # 원거리 추정은 '확정' 근거로 쓰지 않는다 (양자화 오차 큼).
        # SEEK가 접근한 뒤 가까운 거리에서 다시 재도록 이력에서 제외.
        if acc_d is not None and acc_d > dcfg.est_confirm_dist:
            if self.target_est is None:
                self.target_est = accepted   # 방향 참고용으로만 유지
            return

        self._est_hist.append(accepted)
        if len(self._est_hist) > 6:
            self._est_hist.pop(0)
        if self.target_est is None:
            self.target_est = accepted
        else:  # 지수 평활로 안정화
            a = 0.35
            self.target_est = (self.target_est[0] * (1 - a) + accepted[0] * a,
                               self.target_est[1] * (1 - a) + accepted[1] * a)

    def _looking_at_visited(self, pose, bearing):
        """탐지 bearing이 이미 방문한 목표 방향인가 — SEEK 재추적 차단.

        방문 반경 필터는 '위치 추정'만 막고 SEEK 진입은 못 막아서,
        확정 없는 bearing 추종이 방문한 사과를 들이받는 사고 방지.
        """
        for vt in self.visited_targets:
            d = dist(pose, vt)
            if d > 3.5:
                continue
            b_to = wrap_angle(math.atan2(vt[1] - pose[1],
                                         vt[0] - pose[0]) - pose[2])
            if abs(wrap_angle(b_to - bearing)) < 0.25:
                return True
        return False

    def _estimate_stable(self):
        """최근 추정들이 반경 안에 뭉쳐 있는가 (확정 게이트)."""
        k = self.cfg.detection.est_stable_count
        if len(self._est_hist) < k:
            return False
        pts = self._est_hist[-k:]
        r = self.cfg.detection.est_stable_radius
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                if dist(pts[i], pts[j]) > r:
                    return False
        return True

    def _classify_dynamic(self, pose, angles, ranges):
        """빔 끝점이 '확실히 free로 알던 셀'에 찍히면 동적 장애물로 분류.

        벽 스침 노이즈 배제: 끝점 셀의 4-이웃에 점유 셀이 있으면 제외.
        """
        import numpy as np
        g = self.grid
        a = np.asarray(angles)
        r = np.asarray(ranges)
        dyn = np.zeros(r.shape, dtype=bool)
        ok = np.isfinite(r) & (r > self.cfg.lidar.min_range) \
            & (r < self.cfg.lidar.max_range * 0.95)
        if not ok.any():
            return dyn
        idx = ok.nonzero()[0]
        ex = pose[0] + r[idx] * np.cos(pose[2] + a[idx])
        ey = pose[1] + r[idx] * np.sin(pose[2] + a[idx])
        ix = ((ex - g.origin_x) / g.res).astype(np.int64)
        iy = ((ey - g.origin_y) / g.res).astype(np.int64)
        inb = (ix >= 1) & (ix < g.n - 1) & (iy >= 1) & (iy < g.n - 1)
        if not inb.any():
            return dyn
        sel = idx[inb]
        sx, sy = ix[inb], iy[inb]
        was_free = g.log[sy, sx] < self.cfg.plan.dyn_free_logodds
        near_wall = np.zeros(sx.shape, dtype=bool)
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            # '진짜 벽'(강한 점유) 이웃만 제외 사유 — 사람 잔상(약한 점유)
            # 이웃 때문에 분류가 무너지지 않게 임계를 높게 둔다
            near_wall |= g.log[sy + dy, sx + dx] \
                > self.cfg.plan.dyn_wall_logodds
        dyn[sel] = was_free & ~near_wall
        # 위치 연속성: 직전 사람 위치 주변에 찍힌 빔은 셀 상태와 무관하게
        # 동적 취급 — 부분 분류 → 지도 오염 → 분류 붕괴 악순환을 끊는다
        if self.planner.avoid_xy is not None:
            ax, ay = self.planner.avoid_xy
            near_p = (ex[inb] - ax) ** 2 + (ey[inb] - ay) ** 2 \
                < self.cfg.plan.dyn_track_radius ** 2
            dyn[sel] |= near_p
        return dyn

    def _update_person_est(self, now, pose, angles, ranges, dyn):
        """동적 빔 끝점 중심 = 사람 현재 위치 + 속도 추정.

        속도 벡터는 planner의 '예측 쓸기 경로(캡슐)' 페널티에 쓰인다 —
        사람이 앞으로 지나갈 길은 피하고, 등 뒤로 건너는 경로가 나온다.
        """
        import numpy as np
        if dyn is not None and dyn.sum() >= 3:
            a = np.asarray(angles)[dyn]
            r = np.asarray(ranges)[dyn]
            # 사람은 좁은 각도 클러스터로 보인다 — 산발적으로 흩어진
            # 동적 빔(벽 그레이징 오분류)은 사람 위치로 쓰지 않는다
            ca, sa = np.cos(a), np.sin(a)
            spread = 1.0 - math.hypot(float(ca.mean()), float(sa.mean()))
            if spread > 0.12:      # ±약 30도 이상 흩어짐 → 기각
                if now - getattr(self, "_person_seen_t", -1e9) > 2.0:
                    self.planner.avoid_xy = None
                    self.planner.avoid_vel = None
                    self._person_hist = []
                return
            cx = float(np.mean(pose[0] + r * np.cos(pose[2] + a)))
            cy = float(np.mean(pose[1] + r * np.sin(pose[2] + a)))
            self.planner.avoid_xy = (cx, cy)
            self._person_seen_t = now
            # 영토 기억: 사람이 다녀간 자리 전체를 경로 페널티로 —
            # 순찰 축을 종단해야 할 때 A*가 측면 차선을 잡게 만든다
            terr = getattr(self, "_territory", [])
            if not terr or (terr[-1][1] - cx) ** 2 \
                    + (terr[-1][2] - cy) ** 2 > 0.15 ** 2:
                terr.append((now, cx, cy))
            self._territory = [p for p in terr if now - p[0] < 25.0][-40:]
            self.planner.avoid_territory = [(p[1], p[2])
                                            for p in self._territory]
            # 속도 추정: ~0.7s 창의 위치 변화 (EMA로 완만하게)
            hist = getattr(self, "_person_hist", [])
            hist.append((now, cx, cy))
            while hist and hist[0][0] < now - 0.7:
                hist.pop(0)
            self._person_hist = hist
            if len(hist) >= 4 and hist[-1][0] - hist[0][0] > 0.25:
                dt_ = hist[-1][0] - hist[0][0]
                vx = (hist[-1][1] - hist[0][1]) / dt_
                vy = (hist[-1][2] - hist[0][2]) / dt_
                old = self.planner.avoid_vel or (0.0, 0.0)
                al = 0.4
                self.planner.avoid_vel = (old[0] * (1 - al) + vx * al,
                                          old[1] * (1 - al) + vy * al)
        elif now - getattr(self, "_person_seen_t", -1e9) > 2.0:
            self.planner.avoid_xy = None
            self.planner.avoid_vel = None
            self._person_hist = []

    def _approach_goal(self, pose):
        """목표 앞 standoff 지점을 goal로 (목표 자체에 박지 않도록)."""
        tx, ty = self.target_est
        d = dist(pose, self.target_est)
        s = self.cfg.mission.target_standoff
        if d < 1e-6:
            return (tx, ty)
        ratio = max(0.0, (d - s) / d)
        return (pose[0] + (tx - pose[0]) * ratio,
                pose[1] + (ty - pose[1]) * ratio)

    # ── 메인 스텝 ─────────────────────────────────────────────
    def step(self, now, pose, angles, ranges, camera_img):
        cfg = self.cfg
        if self.start_time is None:
            self.start_time = now
            self.state_since = now
            self.last_spin_t = now
        if self._last_theta is None:
            self._last_theta = pose[2]

        # 공통 갱신 (복귀 중엔 crumb 기록 중지 — 복귀 자취가 스스로를
        # 오염시켜 crumb 폴백이 꼬리를 무는 것 방지)
        if self.state not in (self.RETURN, self.DONE):
            self.crumbs.record(pose)
        self.energy.update(pose)
        self.detector.process(camera_img)
        table_added = self.table_mapper.update(
            now, pose, angles, ranges,
            camera_img.shape[1] if camera_img is not None else 0,
            self.detector.table_boxes())
        if table_added:
            self.table_cells_added += table_added
            # 기존 경로가 방금 생성한 테이블 장벽을 관통할 수 있으므로 즉시 갱신.
            if self.planner.goal is not None:
                self.planner.plan_to(pose, self.planner.goal, now, force=True)
        if self.detector.visible:
            self._update_target_estimate(pose, angles, ranges)
        if self.detector.confirmed and not self.target_found \
                and self.target_est is not None and self._estimate_stable():
            # detector.process()에서 이미 YOLO apple → HSV red 검증을 마쳤다.
            self.target_found = True

        # 목표 확정(위치 추정 성공) → 접근 전환 (복귀/완료 전이면)
        # RECOVER는 제외: 래치가 1틱 만에 복구를 선점하면 stuck 영구화
        if self.target_found and not self.target_reached \
                and self.state in (self.SPIN, self.EXPLORE, self.SEEK):
            self._enter(self.GOTO_TARGET, now)
        # 목표 보이지만 위치 미확정 → bearing 추종으로 접근해서 재측정
        elif not self.target_found and not self.target_reached \
                and self.detector.confirmed \
                and now > self._seek_block_until \
                and self.detector.last is not None \
                and not self._looking_at_visited(
                    pose, self.detector.last.bearing) \
                and self.state in (self.SPIN, self.EXPLORE):
            self._enter(self.SEEK, now)

        # 포기 시간 초과 → 복귀
        if cfg.mission.give_up_time > 0 and not self.target_reached \
                and self.state in (self.SPIN, self.EXPLORE, self.SEEK,
                                   self.GOTO_TARGET) \
                and now - self.start_time > cfg.mission.give_up_time:
            self.return_reason = "TIMEOUT"
            self._enter(self.RETURN, now)

        # 에너지 안전 계층은 목표 추적보다 우선한다. 고정 20% 임계값이
        # 아니라 현재 위치에서 breadcrumb로 귀환하는 데 필요한 양과
        # reserve를 비교한다. RECOVER 중이면 복구 직후 곧바로 귀환한다.
        active = (self.SPIN, self.EXPLORE, self.SEEK, self.GOTO_TARGET)
        if self.state in active and self.energy.should_return(pose, self.crumbs):
            self.return_reason = "LOW_BATTERY"
            print(f"[mission] 저전력 안전 귀환 — battery="
                  f"{self.energy.percent:.1f}% required="
                  f"{self.energy.required_return_percent(pose, self.crumbs):.1f}%")
            self._enter(self.RETURN, now)
        elif self.state == self.RECOVER \
                and self.energy.should_return(pose, self.crumbs):
            self.return_reason = "LOW_BATTERY"
            self._recover_after = self.RETURN

        # stuck → RECOVER (SPIN/DONE/RECOVER 제외)
        if self.state in (self.EXPLORE, self.SEEK, self.GOTO_TARGET,
                          self.RETURN) \
                and self._update_stuck(now, pose):
            self._recover_after = self.state
            self._recover_phase = ("backup", now)
            self._stuck_count += 1
            self._pose_hist.clear()
            if self._stuck_count >= 3 and self.state == self.EXPLORE:
                self.explorer.fail_current()
                self._stuck_count = 0
            self._enter(self.RECOVER, now)

        # 동적 장애물 분류 + 사람 위치 추정 (경로의 소프트 회피용)
        dyn = self._classify_dynamic(pose, angles, ranges)
        self._update_person_est(now, pose, angles, ranges, dyn)

        # 지도 갱신 — 동적 빔은 제외 (사람 궤적이 지도를 오염시키면
        # 동적 분류 자체가 무력화되므로, 분류 → 제외 → 통합 순서 고정)
        if now - self._last_map_t >= self.cfg.map.update_period:
            self._last_map_t = now
            self.grid.integrate_scan(pose, angles, ranges,
                                     subsample=self.cfg.lidar.subsample,
                                     exclude=dyn)

        v, w = 0.0, 0.0
        handler = {
            self.SPIN: self._do_spin,
            self.EXPLORE: self._do_explore,
            self.SEEK: self._do_seek,
            self.GOTO_TARGET: self._do_goto,
            self.RETURN: self._do_return,
            self.RECOVER: self._do_recover,
            self.DONE: self._do_done,
        }[self.state]
        v, w = handler(now, pose, angles, ranges)

        # 회피는 전 상태 공통 (후진 recovery 중에는 전방 회피 제외)
        if self.state not in (self.RECOVER, self.DONE):
            v, w, blocked_long = self.avoider.apply(
                v, w, angles, ranges, now, dynamic=dyn, pose=pose,
                person_xy=self.planner.avoid_xy,
                person_vel=self.planner.avoid_vel)
            # SPIN은 제자리 스캔이 목적 — 실제 탈출 상황이 아니면
            # 어떤 제안도 전진시키지 못하게 (오분류가 나선 전진 유발 방지)
            if self.state == self.SPIN \
                    and self.avoider.last_mode not in ("escape", "blind"):
                v = 0.0
            if blocked_long:
                # 오래 막힘: 지도에 반영됐을 것 → 강제 재계획
                self.planner.plan_to(pose, self.planner.goal or pose[:2],
                                     now, force=True) \
                    if self.planner.goal else None
                if self.state == self.EXPLORE:
                    self.explorer.fail_current()

        # 명령 평활화 — 채터로 인한 '문 앞 떨림' 제거.
        # 안전 비대칭: 감속·정지·후진은 즉시 반영, 가속만 제한.
        # 조향은 변화율 제한 (원형 로봇의 회전은 접촉과 무관 → 안전 영향 없음)
        dt_cmd = 0.064 if self._last_step_t is None \
            else max(1e-3, min(0.2, now - self._last_step_t))
        self._last_step_t = now
        dv_max = 0.35 * dt_cmd            # 가속 한계 (0→최고속 ~0.6s)
        dw_max = 6.0 * dt_cmd             # 조향 변화 한계 (풀스윙 ~0.5s)
        if v > self._v_prev:
            v = min(v, self._v_prev + dv_max)
        w = max(self._w_prev - dw_max, min(self._w_prev + dw_max, w))
        self._v_prev, self._w_prev = v, w

        self._last_theta = pose[2]
        info = {
            "state": self.state,
            "dyn_count": int(dyn.sum()) if dyn is not None else 0,
            "target_est": self.target_est,
            "goal": self.planner.goal,
            "waypoints": list(self.planner.waypoints),
            "crumbs": list(self.crumbs.crumbs),
            "found": self.target_found,
            "visited": len(self.visited_targets),
            "reached": self.target_reached,
            "battery": self.energy.percent,
            "return_required": self.energy.required_return_percent(
                pose, self.crumbs),
            "return_reason": self.return_reason,
            "table_cells": self.table_cells_added,
        }
        return v, w, info

    # ── 상태별 로직 ───────────────────────────────────────────
    def _do_spin(self, now, pose, angles, ranges):
        """카메라 스캔 회전. LiDAR는 360도라 회전이 필요 없다 — 이 회전은
        오직 60도 카메라로 목표를 훑기 위한 것.

        최초 1회: 360도 풀스핀 (지도 시드 + 주변 전체 카메라 확인).
        이후 주기 스캔: 전방 ±spin_arc/2 스윕만 (한 바퀴 8초 → 2~3초).
        """
        cfg = self.cfg
        dth = wrap_angle(pose[2] - self._last_theta)
        full = not self._initial_spin_done or self._force_full_spin
        # 시간 상한: 어떤 외부 간섭(회피 개입 등)이 있어도 스핀은 반드시
        # 끝난다 — 스핀 무한 루프 데드락 방지
        limit = (2 * math.pi / (cfg.robot.max_w * 0.5) + 5.0) if full \
            else (cfg.explore.spin_arc / (cfg.robot.max_w * 0.6) * 2 + 4.0)
        if now - self.state_since > limit:
            self._initial_spin_done = True
            self._force_full_spin = False
            self.spin_accum = 0.0
            self.last_spin_t = now
            self._enter(self._spin_return_state, now)
            return 0.0, 0.0
        if full:
            self.spin_accum += abs(dth)
            if self.spin_accum >= 2 * math.pi:
                self._initial_spin_done = True
                self._force_full_spin = False
                self.spin_accum = 0.0
                self.last_spin_t = now
                self._enter(self._spin_return_state, now)
                return 0.0, 0.0
            return 0.0, cfg.robot.max_w * 0.5
        # 부분 스윕: 좌로 arc/2 → 우로 arc (=우측 -arc/2에서 종료)
        self.spin_accum += dth
        half = cfg.explore.spin_arc / 2
        if self._spin_phase == "left":
            if self.spin_accum >= half:
                self._spin_phase = "right"
            return 0.0, cfg.robot.max_w * 0.6
        if self.spin_accum <= -half:
            self.spin_accum = 0.0
            self.last_spin_t = now
            self._enter(self._spin_return_state, now)
            return 0.0, 0.0
        return 0.0, -cfg.robot.max_w * 0.6

    def _do_explore(self, now, pose, angles, ranges):
        cfg = self.cfg
        # 주기적 카메라 스캔 (진입 시각을 즉시 기록 — 스핀이 SEEK 등에
        # 선점돼도 곧바로 재진입하며 겉도는 것 방지)
        if cfg.explore.spin_period > 0 \
                and now - self.last_spin_t > cfg.explore.spin_period:
            self.spin_accum = 0.0
            self._spin_phase = "left"
            self.last_spin_t = now
            self._spin_return_state = self.EXPLORE
            self._enter(self.SPIN, now)
            return 0.0, 0.0

        target = self.explorer.update(pose, person_xy=self.planner.avoid_xy)
        # frontier 도착 = 새 시야가 열린 순간 → 카메라 스윕 1회
        if self.explorer.just_reached:
            self.explorer.just_reached = False
            self.spin_accum = 0.0
            self._spin_phase = "left"
            self.last_spin_t = now
            self._spin_return_state = self.EXPLORE
            self._enter(self.SPIN, now)
            return 0.0, 0.0
        if target is None:
            # frontier 소진: 블랙리스트 때문일 수 있으니 사면 후 재시도
            # (최대 2회 — 진짜 소진이면 복귀 fail-safe)
            if self.explorer.blacklist and self._amnesty_count < 2:
                self._amnesty_count += 1
                self.explorer.blacklist.clear()
                return 0.0, 0.0
            # 복귀 전 마지막 360도 — 카메라가 못 훑은 방향 최종 확인
            if not self._final_spin_done:
                self._final_spin_done = True
                self._force_full_spin = True
                self.spin_accum = 0.0
                self.last_spin_t = now
                self._spin_return_state = self.EXPLORE
                self._enter(self.SPIN, now)
                return 0.0, 0.0
            self._enter(self.RETURN, now)
            if self.return_reason is None:
                self.return_reason = "SEARCH_EXHAUSTED"
            return 0.0, 0.0
        ok = self.planner.plan_to(pose, target, now)
        if not ok:
            self.explorer.fail_current()
            return 0.0, 0.0
        return self.planner.follow(pose)

    def _do_seek(self, now, pose, angles, ranges):
        """목표가 보이지만 LiDAR 거리 확정 전: bearing을 향해 전진.

        가까워지면 LiDAR가 잡히고 → target_est 성공 → GOTO_TARGET 전환.
        (전환은 step() 공통부에서 처리)
        """
        cfg = self.cfg
        if not self.detector.confirmed or self.detector.last is None:
            self._enter(self.EXPLORE, now)   # 시야 상실 → 탐색 재개
            return 0.0, 0.0
        # 시간 상한: 위치 확정이 계속 안 되는 대상(방문한 사과·원거리 디코이)
        # 을 무한 응시하지 않는다
        if now - self.state_since > 25.0:
            self._seek_block_until = now + 12.0
            self._enter(self.EXPLORE, now)
            return 0.0, 0.0
        det = self.detector.last
        # 방문한 목표를 향한 시선이면 추종 중단
        if self._looking_at_visited(pose, det.bearing):
            self._seek_block_until = now + 10.0
            self._enter(self.EXPLORE, now)
            return 0.0, 0.0
        # 근접 정지 안전망: 확정 전에 목표에 지나치게 접근 금지
        # (LiDAR에 안 보이는 사과는 반응형 회피가 못 막는다)
        if det.ang_width > 0.25:      # 사과 기준 ~0.4m 이내
            b0 = det.bearing
            w0 = max(-cfg.robot.max_w, min(cfg.robot.max_w, 2.0 * b0))
            return 0.0, w0
        b = det.bearing
        w = max(-cfg.robot.max_w, min(cfg.robot.max_w, 2.0 * b))
        v = cfg.robot.max_v * max(0.2, 1.0 - abs(b) / (math.pi / 3))
        return v, w

    def _do_goto(self, now, pose, angles, ranges):
        cfg = self.cfg
        if self.target_est is None:      # 추정 소실 (이례적) → 탐색 복귀
            self.target_found = False
            self._enter(self.EXPLORE, now)
            return 0.0, 0.0
        goal = self._approach_goal(pose)
        if dist(pose, goal) < cfg.mission.target_reach_tol:
            # 현재 목표 방문 완료 — 전부 채웠으면 복귀, 남았으면 탐색 재개
            self.visited_targets.append(self.target_est)
            k = len(self.visited_targets)
            print(f"[mission] 목표 {k}/{cfg.mission.num_targets} 방문 완료 "
                  f"({self.target_est[0]:+.2f},{self.target_est[1]:+.2f})")
            self.target_found = False
            self.target_est = None
            self._est_hist.clear()
            self._seek_block_until = now + 8.0   # 방금 그 사과 응시 방지
            if k >= cfg.mission.num_targets:
                self.target_reached = True
                self.return_reason = "MISSION_COMPLETE"
                self._enter(self.RETURN, now)
            else:
                self._enter(self.EXPLORE, now)
            return 0.0, 0.0
        ok = self.planner.plan_to(pose, goal, now)
        if not ok:
            # 계획 실패가 이어지면 확정을 풀고 탐색으로 (동결 방지).
            # 백오프는 Planner가 처리하므로 여기서 재시도 연타 금지.
            if self._goto_fail_since is None:
                self._goto_fail_since = now
            elif now - self._goto_fail_since > 3.0:
                self.target_found = False
                self.target_est = None
                self._est_hist.clear()
                self._goto_fail_since = None
                self._enter(self.SEEK if self.detector.confirmed
                            else self.EXPLORE, now)
            return 0.0, 0.0
        self._goto_fail_since = None
        return self.planner.follow(pose)

    def _do_return(self, now, pose, angles, ranges):
        cfg = self.cfg
        if dist(pose, self.start_xy) < cfg.mission.return_tolerance:
            self._enter(self.DONE, now)
            return 0.0, 0.0
        # crumb 폴백은 영구 래치가 아니라 주기적으로 A* 재시도
        # (일시적 계획 실패 한 번으로 지름길 복귀를 포기하지 않게)
        if self._crumb_mode and self._crumb_since is not None \
                and now - self._crumb_since > 4.0:
            self._crumb_mode = False
        if cfg.mission.use_astar_return and not self._crumb_mode:
            ok = self.planner.plan_to(pose, self.start_xy, now)
            if ok:
                return self.planner.follow(pose)
            self._crumb_mode = True   # A* 실패 → breadcrumb 폴백
            self._crumb_since = now
        # crumb 복귀: waypoint를 직접 주입
        path = self.crumbs.return_path(pose)
        if path:
            self.planner.waypoints = path
            self.planner._goal = self.start_xy
            return self.planner.follow(pose)
        # crumb도 없으면 시작점 직진 시도
        self.planner.waypoints = [self.start_xy]
        self.planner._goal = self.start_xy
        return self.planner.follow(pose)

    def _do_recover(self, now, pose, angles, ranges):
        phase, t0 = self._recover_phase
        if phase == "backup":
            # 기존의 맹목적인 1.2초 후진은 테이블 반대편 다리에 다시
            # 부딪힐 수 있다. 후방 로봇 폭 안의 실제 여유를 매 틱 확인한다.
            import numpy as np
            a = np.asarray(angles)
            r = np.asarray(ranges)
            valid = np.isfinite(r) & (r > self.cfg.lidar.min_range)
            rear_forward = r * np.cos(wrap_angle(a - math.pi))
            rear_lateral = r * np.sin(wrap_angle(a - math.pi))
            rear_corridor = valid & (rear_forward > 0.02) \
                & (np.abs(rear_lateral) < self.cfg.robot.robot_radius
                   + self.cfg.plan.corridor_margin)
            rear_min = float(rear_forward[rear_corridor].min()) \
                if rear_corridor.any() else self.cfg.lidar.max_range
            if rear_min <= self.cfg.table.recover_rear_stop:
                self._recover_phase = ("spin", now)
                return 0.0, 0.0
            if now - t0 < 1.2:
                return -0.08, 0.0
            self._recover_phase = ("spin", now)
            return 0.0, 0.0
        else:  # spin: 여유 있는 쪽으로 90도 가량 회전
            if now - t0 < 1.3:
                import numpy as np
                a = np.asarray(angles)
                r = np.asarray(ranges)
                left = r[(a > 0.3) & (a < 1.6)]
                right = r[(a < -0.3) & (a > -1.6)]
                lmin = left[np.isfinite(left)].min() if left.size else 10.0
                rmin = right[np.isfinite(right)].min() if right.size else 10.0
                turn = 1.0 if lmin > rmin else -1.0
                return 0.0, turn * self.cfg.robot.max_w * 0.6
            # 재계획 강제 후 이전 상태로
            self._recover_phase = None
            if self.planner.goal:
                self.planner.plan_to(pose, self.planner.goal, now, force=True)
            self._enter(self._recover_after, now)
            return 0.0, 0.0

    def _do_done(self, now, pose, angles, ranges):
        return 0.0, 0.0
