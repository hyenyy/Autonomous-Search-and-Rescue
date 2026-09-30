"""미션 상태 기계: SPIN → EXPLORE → GOTO_TARGET → RETURN → DONE (+RECOVER).

모든 상태에서 LocalAvoider 회피는 항상 적용된다.
main 루프(오프라인 sim / Webots controller)는 매 스텝:
    mission.step(now, pose, angles, ranges, camera_img) → (v, w, info)
만 호출하면 된다. 지도 갱신은 동적 장애물 분류 뒤 이 모듈에서 수행한다.
"""
import math

from .breadcrumbs import Breadcrumbs
from .detection import (TargetDetector, estimate_target_position,
                        estimate_target_position_mono)
from .exploration import FrontierExplorer
from .geometry import dist, wrap_angle
from .planning import LocalAvoider, Planner


class Mission:
    # 상태 이름 상수
    SPIN = "SPIN"                # 제자리 1회전 스캔
    WALL_FOLLOW = "WALL_FOLLOW"  # 벽 추종(우수법) — 지도 골격 완성
    EXPLORE = "EXPLORE"
    SEEK = "SEEK"                # 목표 보임 but 거리 미확정 → bearing 추종
    VISIT = "VISIT"              # 지도 완성 후 기억해둔 후보 방문·확인
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
        self.avoider = LocalAvoider(cfg)

        self.start_xy = (cfg.mission.start_x, cfg.mission.start_y)
        self.state = self.SPIN
        self.state_since = 0.0
        self.spin_accum = 0.0
        self._last_theta = None
        self._spin_return_state = self.WALL_FOLLOW \
            if (cfg.mission.map_first and cfg.explore.wall_follow) \
            else self.EXPLORE
        self.last_spin_t = 0.0
        self._initial_spin_done = False   # 최초 1회만 360도 (지도 시드)
        self._spin_phase = "left"         # 부분 스윕 상태
        self._force_full_spin = False     # frontier 소진 시 마지막 360도
        self._final_spin_done = False
        self._known_at_spin = 0           # 직전 스핀 시점의 기지(旣知) 셀 수

        self.target_est = None       # (x, y) 현재 추적 중인 목표 추정
        self._est_hist = []          # 최근 채택된 추정들 (안정성 게이트용)
        self._decoys = []            # YOLO가 기각한 디코이 위치들
        self.visited_targets = []    # 방문 완료한 목표 위치들
        self._seek_block_until = -1e9
        self.target_found = False    # '현재' 목표 확정 여부
        self._target_anchor = None
        self._target_seen_t = 0.0
        self._target_tracking_valid = False
        self.target_reached = False  # '모든' 목표 방문 완료 여부

        # 지도 우선 전략: 탐사 중 목표는 후보로만 기억 → 지도 완성 후 방문
        self._map_first_active = cfg.mission.map_first
        self._wf_active = cfg.mission.map_first and cfg.explore.wall_follow
        self.candidates = []         # [{"xy": (x,y), "w": 누적 가중치}]
        self._visit_phase = False
        self._visit_cand = None      # 현재 방문 중인 후보 좌표
        self._visit_gaze_t = None    # 후보 앞 응시 시작 시각
        self._visit_fail_t = None    # 후보 접근 계획 실패 시작 시각
        self._wf_anchor = None       # 벽 추종 루프 폐합 판정 기준점
        self._wf_path = 0.0
        self._wf_last = None

        # YOLO가 인식한 가구(탁자·의자 등) 구역 — 밑으로 기어들어가
        # 갇히는 것을 '예방': A* 페널티 + frontier 후순위. 갇힘 탈출
        # 스택은 이 레이어가 뚫렸을 때의 최후 수단.
        self.furniture_zones = []    # [(x, y)]

        # 구역 맴돌기 감지: 움직이고는 있는데 40초째 같은 방 안에서
        # 왔다갔다 → 벽 추종으로 이동 패턴을 깨고 나간다
        self._region_hist = []       # [(t, x, y)]
        self._region_check_t = -1e9
        self._region_escape_cd = -1e9
        self._wf_escape_until = None  # 벽 추종 '탈출 모드' 종료 시각

        # stuck 감지
        self._pose_hist = []         # [(t, x, y)]
        self._recover_phase = None   # ("backup"|"align"|"creep", 시작시각)
        self._recover_after = self.EXPLORE
        self._stuck_count = 0
        self._last_stuck_t = -1e9
        self._recover_escalate = False   # 반복 갇힘 → creep 탈출까지 격상
        self._escape_heading = None      # 탈출 방향 (월드 헤딩)
        self._creep_start = None

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
        self._last_scan_pose = None
        self._last_scan_time = -1e9
        self._last_step_t = None

    # ── 내부 유틸 ─────────────────────────────────────────────
    def _enter(self, state, now):
        if state == self.GOTO_TARGET and self._target_anchor is None:
            self._target_anchor = self.target_est
            self._target_seen_t = now
        if state == self.SPIN:
            # 정지 스캔 구간이 stuck 이력에 섞이면 스핀 직후 오탐이 난다
            self._pose_hist.clear()
            self._mark_spin()
        self.state = state
        self.state_since = now

    def _update_stuck(self, now, pose):
        self._pose_hist.append((now, pose[0], pose[1]))
        horizon = self.cfg.mission.stuck_time
        long_h = max(12.0, horizon)
        while self._pose_hist and self._pose_hist[0][0] < now - long_h - 0.5:
            self._pose_hist.pop(0)
        if not self._pose_hist:
            return False
        t_first = self._pose_hist[0][0]
        # 단기: 거의 정지 (제자리 회전만 반복)
        if now - t_first >= horizon:
            recent = [(t, x, y) for t, x, y in self._pose_hist
                      if t >= now - horizon]
            _, x0, y0 = recent[0]
            moved = max(math.hypot(x - x0, y - y0) for _, x, y in recent)
            if moved < self.cfg.mission.stuck_dist:
                return True
        # 장기: 좁은 반경 왕복/맴돌기 — 탁자 밑처럼 '움직이지만 갇힌'
        # 상태. 단기 게이트(8cm)는 30cm씩 오가는 진동이나 소형 원돌기를
        # 못 잡는다 (12s에 정상 주행이면 1m+는 나간다).
        # 사람 양보 대기 중에는 제자리 체류가 정상이므로 제외.
        if now - t_first >= long_h and self.planner.avoid_xy is None:
            xs = [x for _, x, _ in self._pose_hist]
            ys = [y for _, _, y in self._pose_hist]
            span = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
            if span < 0.7:
                return True
        return False

    @staticmethod
    def _sector_min(angles, ranges, phi, halfwidth=0.4):
        """로봇 프레임 방향 phi 섹터의 최소 LiDAR 거리."""
        import numpy as np
        a = np.asarray(angles)
        r = np.asarray(ranges)
        ok = np.isfinite(r) & (r > 0.05)
        if not ok.any():
            return 10.0
        a, r = a[ok], r[ok]
        sel = np.abs(wrap_angle(a - phi)) < halfwidth
        if not sel.any():
            return 10.0
        return float(r[sel].min())

    def _best_gap_heading(self, pose, angles, ranges, goal_xy=None):
        """'연속 빈 각도 구간(run)'의 중앙을 겨냥 — 탈출 방향 선정.

        섹터 최소거리 점수(구 방식)는 ]  [ 처럼 갇혔을 때 틈 양옆
        문설주가 min을 끌어내려 '먼 벽 방향'(=장애물 쪽)을 고르는 오류가
        있었다: 장애물로 직진 → 걸림 → 회전·후진 → 다시 장애물로 직진.
        free 빔의 연속 구간을 찾아 그 한가운데로 직진해야 한다.
        goal이 있으면 그 방향 구간에 보너스 — 좁은 문 '재진입'(복귀
        목표가 탁자 안) 시 무조건 트인 쪽으로 도망가는 것 방지."""
        import numpy as np
        a = np.asarray(angles)
        r = np.asarray(ranges)
        ok = np.isfinite(r) & (r > 0.05)
        a, r = a[ok], np.minimum(r[ok], 3.0)
        if a.size < 8:
            return pose[2]
        order = np.argsort(a)
        a, r = a[order], r[order]
        g_ang = None
        if goal_xy is not None:
            g_ang = math.atan2(goal_xy[1] - pose[1], goal_xy[0] - pose[0])
        free = r > 0.55
        if free.all():                       # 사방이 트임 — 목표/전방으로
            return g_ang if g_ang is not None else pose[2]
        if not free.any():                   # 전부 근접 — 최심 빔으로
            return wrap_angle(pose[2] + float(a[int(np.argmax(r))]))
        n = free.size
        off = int(np.argmin(free))           # blocked 빔에서 스캔 시작
        fr = np.roll(free, -off)             # (wrap-around run 처리)
        runs, s = [], None
        for i in range(n):
            if fr[i] and s is None:
                s = i
            elif not fr[i] and s is not None:
                runs.append((s, i))
                s = None
        if s is not None:
            runs.append((s, n))
        best, best_phi = -1e9, None
        beam_step = 2 * math.pi / n
        for s, e in runs:
            width = (e - s) * beam_step
            if width < 0.22:                 # 로봇이 못 지나갈 좁은 구간
                continue
            ii = (np.arange(s, e) + off) % n
            # The middle free ray is not necessarily a passable direction for
            # a finite-radius robot. Score swept-circle clearance at each ray;
            # nearby doorposts must constrain even a long-range center ray.
            radius = self.cfg.robot.robot_radius + .015
            for index in ii:
                phi = float(a[index])
                forward = r * np.cos(a - phi)
                lateral = r * np.sin(a - phi)
                corridor = (forward > 0) & (np.abs(lateral) < radius)
                depth = float(np.min(forward[corridor] - np.sqrt(
                    radius ** 2 - lateral[corridor] ** 2))) if corridor.any() else 3.0
                score = min(depth, 1.5) + 0.35 * min(width, 1.5) - 0.06 * abs(phi)
                if g_ang is not None and depth > 1.0:
                    score += 0.5 * math.cos(wrap_angle(pose[2] + phi - g_ang))
                if score > best:
                    best, best_phi = score, phi
        if best_phi is None:
            best_phi = float(a[int(np.argmax(r))])
        return wrap_angle(pose[2] + best_phi)

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
        # 좌우 잘림 블롭은 위치 추정 금지 — 잘린 물체는 종횡비·채움비·
        # 크기 일관성이 전부 '보이는 조각' 기준이라 무의미하다. 화면
        # 모서리에 하반신만 걸친 소화기가 "0.6m 사과"로 확정된 실사고:
        # bbox=(0,85,139,314), aspect 0.61, fill 0.62, LiDAR 크기비 1.3
        # — 게이트 전부 통과. SEEK가 중앙에 세우면 온전한 게이트로 재평가.
        if det.clipped:
            return
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
        # A 10cm floor apple must project near its known height. Letter-sized
        # red patches on walls can pass color/shape but imply a point below the
        # floor when their width is mistaken for an apple. No ground truth used.
        if det.bbox is not None and not det.v_clipped:
            camera = self.cfg.camera
            cy = (det.bbox[1] + det.bbox[3]) / 2
            focal = (camera.width / 2) / math.tan(camera.hfov / 2)
            depth = acc_d * math.cos(det.bearing - camera.mount_yaw)
            height = camera.height_above_floor - (cy - camera.height / 2) * depth / focal
            if abs(height - dcfg.target_center_height) > dcfg.height_tolerance:
                return
        if self.target_found and self._target_anchor is not None \
                and dist(accepted, self._target_anchor) > dcfg.target_association_radius:
            return  # do not drag one confirmed target toward another red object
        # 등록된 디코이(YOLO 기각 위치)·이미 방문한 목표 주변 추정은 버린다
        r_d = self.cfg.detection.decoy_radius
        if any(dist(accepted, dc) < r_d for dc in self._decoys):
            return
        r_v = self.cfg.mission.visited_radius
        if any(dist(accepted, vt) < r_v for vt in self.visited_targets):
            return
        self._target_tracking_valid = True

        # 지도 우선 모드: 모든 유효 추정을 '후보'로 기억 (원거리 포함 —
        # ±25% 오차라도 VISIT가 가서 확인할 예상 위치로는 충분)
        if self._map_first_active:
            near = acc_d is not None and acc_d <= dcfg.est_confirm_dist
            self._add_candidate(accepted, near)

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
        return True

    def note_furniture(self, xy, now=0.0, label=""):
        """컨트롤러의 YOLO 가구 탐지 보고 — 0.6m 내 기존 구역과 병합.

        구역은 TTL(75s)로 자연 소멸 — 오인식 한 방이 통로를 영구히
        비싸게 만드는 것을 방지. 재관측되면 갱신되어 유지된다."""
        for z in self.furniture_zones:
            if dist(xy, (z[0], z[1])) < 0.6:
                z[2] = now                   # 재관측 → TTL 갱신
                return
        self.furniture_zones.append([xy[0], xy[1], now])
        if len(self.furniture_zones) > 30:
            self.furniture_zones.pop(0)
        print(f"[mission] 가구 구역 기록 ({xy[0]:+.2f},{xy[1]:+.2f}) "
              f"[{label}] — 경로가 밑으로 파고들지 않게 회피")

    def _furniture_valid(self, now):
        self.furniture_zones = [z for z in self.furniture_zones
                                if now - z[2] < 75.0]
        return [(z[0], z[1]) for z in self.furniture_zones]

    def _add_candidate(self, xy, near):
        """목표 후보 등록/병합 — 가까운 관측일수록 위치 가중치 높게."""
        w = 2.0 if near else 1.0
        for c in self.candidates:
            if dist(xy, c["xy"]) < 0.7:
                a = w / (c["w"] + w)
                c["xy"] = (c["xy"][0] * (1 - a) + xy[0] * a,
                           c["xy"][1] * (1 - a) + xy[1] * a)
                c["w"] += w
                return
        self.candidates.append({"xy": tuple(xy), "w": w})
        print(f"[mission] 목표 후보 기억 ({xy[0]:+.2f},{xy[1]:+.2f}) "
              f"— 총 {len(self.candidates)}개 (지도 완성 후 방문)")

    def _prune_candidates(self):
        r_d = self.cfg.detection.decoy_radius
        r_v = self.cfg.mission.visited_radius
        self.candidates = [
            c for c in self.candidates
            if not any(dist(c["xy"], dc) < r_d for dc in self._decoys)
            and not any(dist(c["xy"], vt) < r_v
                        for vt in self.visited_targets)]

    def _pop_candidate(self, pose):
        """다음 방문 후보: 근접 관측(w>=2) 우선, 같은 급에선 가까운 순."""
        self._prune_candidates()
        if not self.candidates:
            return None
        best = max(self.candidates,
                   key=lambda c: (c["w"] >= 2.0, -dist(pose, c["xy"])))
        self.candidates.remove(best)
        return best["xy"]

    def _spin_worthwhile(self, now=None, pose=None):
        """지난 카메라 스윕 이후 지도가 유의미하게 자랐는가.

        안 자랐다 = 이미 훑은 구역을 재통과 중 — 여기서 또 도는 건
        순수 낭비다. 새 방에 들어서면 성장이 감지되어 스윕이 돌아온다."""
        if now is not None and pose is not None and self._last_scan_pose is not None:
            if now - self._last_scan_time < 16 or dist(pose, self._last_scan_pose) < 1.0:
                return False
        known_now = int((~self.grid.unknown_mask()).sum())
        return known_now - self._known_at_spin >= 250   # ≈0.6m² 신규

    def _mark_spin(self):
        self._known_at_spin = int((~self.grid.unknown_mask()).sum())

    def _mapping_state(self):
        """지도 우선 모드에서 '하던 일'로 돌아갈 상태."""
        if self._visit_phase:
            return self.VISIT
        if self._wf_active:
            return self.WALL_FOLLOW
        return self.EXPLORE

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
        # Keep a person's partial returns together, but never let track
        # proximity turn confirmed furniture/walls into a moving obstacle.
        if self.planner.avoid_xy is not None:
            ax, ay = self.planner.avoid_xy
            near_p = (ex[inb] - ax) ** 2 + (ey[inb] - ay) ** 2 \
                < self.cfg.plan.dyn_track_radius ** 2
            static = near_wall | (g.log[sy, sx] > self.cfg.plan.dyn_wall_logodds)
            dyn[sel] |= near_p & ~static
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
                dyn[:] = False  # The same mask feeds mapping and emergency avoidance.
                if now - getattr(self, "_person_seen_t", -1e9) > 2.0:
                    self.planner.avoid_xy = None
                    self.planner.avoid_vel = None
                    self._person_hist = []
                return
            cx = float(np.mean(pose[0] + r * np.cos(pose[2] + a)))
            cy = float(np.mean(pose[1] + r * np.sin(pose[2] + a)))
            # 유령 판별: 3초 넘게 사실상 정지한 '사람'은 문설주·가구 스침
            # 오분류다 (실제 보행자는 0.2m/s로 계속 움직인다). 위치 연속성
            # 규칙이 유령을 자가 유지시키므로 여기서 끊어야 한다 —
            # 유령이 살아 있으면 escape 정지·RECOVER 중단이 영구화된다.
            ph = getattr(self, "_phantom_hist", [])
            ph.append((now, cx, cy))
            ph = [p for p in ph if now - p[0] < 3.5]
            self._phantom_hist = ph
            if ph[-1][0] - ph[0][0] > 3.0:
                span = math.hypot(max(p[1] for p in ph) - min(p[1] for p in ph),
                                  max(p[2] for p in ph) - min(p[2] for p in ph))
                if span < 0.12:
                    dyn[:] = False
                    self.planner.avoid_xy = None
                    self.planner.avoid_vel = None
                    self._person_hist = []
                    # Retain the stationary evidence: clearing it rearms the
                    # same furniture as a moving person on the very next tick.
                    return
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
            self.explorer.observe_pose(pose)
        self.detector.process(camera_img)
        self._target_tracking_valid = False
        valid_estimate = False
        if self.detector.visible and self.state not in (self.RETURN, self.DONE):
            valid_estimate = bool(self._update_target_estimate(pose, angles, ranges))
        if valid_estimate:
            self._target_seen_t = now
        else:
            self._est_hist.clear()
            if not self.target_found:
                self.target_est = None
        # 지도 우선 모드의 탐사 단계: 초근접(즉시 잡는 게 싼 거리) 아니면
        # 확정하지 않는다 — 후보로 기억만 하고 지도 완성을 계속한다
        allow_confirm = True
        if self._map_first_active and not self._visit_phase \
                and self.target_est is not None \
                and dist(pose, self.target_est) \
                > self.cfg.mission.opportunistic_dist:
            allow_confirm = False
        if allow_confirm and valid_estimate and self.detector.confirmed \
                and not self.target_found \
                and self.target_est is not None and self._estimate_stable():
            # 최종 게이트: YOLO(활성 시)가 비사과 물체로 기각하지 않아야
            # 확정. 기각되면 그 위치를 디코이로 등록하고 추정 리셋 —
            # 빨간 음료캔 같은 색·기하 통과형 디코이 방어.
            if self.detector.yolo_confirm(camera_img, self.detector.last):
                self.target_found = True
                self._target_anchor = self.target_est
                self._confirm_img = camera_img       # 검증용 확정 프레임
                d0 = self.detector.last
                print(f"[mission] 목표 확정 est=({self.target_est[0]:+.2f},"
                      f"{self.target_est[1]:+.2f}) aspect={d0.aspect:.2f} "
                      f"fill={d0.fill:.2f} ang_w={d0.ang_width:.3f} "
                      f"bbox={d0.bbox}")
            else:
                print(f"[mission] YOLO 기각 → 디코이 등록 "
                      f"({self.target_est[0]:+.2f},{self.target_est[1]:+.2f})")
                self._decoys.append(self.target_est)
                self.target_est = None
                self._est_hist.clear()
                self._seek_block_until = now + 20.0

        # 목표 확정(위치 추정 성공) → 접근 전환 (복귀/완료 전이면)
        # RECOVER는 제외: 래치가 1틱 만에 복구를 선점하면 stuck 영구화
        if self.target_found and not self.target_reached \
                and self.state in (self.SPIN, self.EXPLORE, self.SEEK,
                                   self.WALL_FOLLOW, self.VISIT):
            self._enter(self.GOTO_TARGET, now)
        # 목표 보이지만 위치 미확정 → bearing 추종으로 접근해서 재측정
        # (YOLO가 디코이로 기각한 직후에는 쿨다운 — 캔을 향한 무한 추종 방지)
        elif not self.target_found and not self.target_reached \
                and not self._map_first_active \
                and self.detector.confirmed \
                and self.detector.visible and self._target_tracking_valid \
                and now > self._seek_block_until \
                and self.detector.last is not None \
                and not self._looking_at_visited(
                    pose, self.detector.last.bearing) \
                and self.state in (self.SPIN, self.EXPLORE):
            self._enter(self.SEEK, now)

        # 포기 시간 초과 → 복귀
        if cfg.mission.give_up_time > 0 and not self.target_reached \
                and self.state in (self.SPIN, self.EXPLORE, self.SEEK,
                                   self.WALL_FOLLOW, self.VISIT,
                                   self.GOTO_TARGET) \
                and now - self.start_time > cfg.mission.give_up_time:
            self._enter(self.RETURN, now)

        # 구역 맴돌기: 이동은 하는데 40s째 좁은 구역 안 — frontier와
        # 회피의 충돌로 같은 길만 왕복하는 패턴. 벽 추종으로 깨고 나간다.
        self._region_hist.append((now, pose[0], pose[1]))
        while self._region_hist and self._region_hist[0][0] < now - 42.0:
            self._region_hist.pop(0)
        if now - self._region_check_t >= 5.0 \
                and self.state in (self.EXPLORE, self.SEEK) \
                and self.planner.avoid_xy is None \
                and now > self._region_escape_cd:
            self._region_check_t = now
            h = self._region_hist
            if h and now - h[0][0] >= 40.0:
                xs_ = [p[1] for p in h]
                ys_ = [p[2] for p in h]
                span = math.hypot(max(xs_) - min(xs_), max(ys_) - min(ys_))
                path_len = sum(
                    math.hypot(h[i + 1][1] - h[i][1], h[i + 1][2] - h[i][2])
                    for i in range(len(h) - 1))
                if span < 2.2 and path_len > 4.0:
                    print(f"[mission] 구역 맴돌기 감지(40s 반경 "
                          f"{span:.1f}m, 이동 {path_len:.1f}m) — "
                          f"벽 추종으로 패턴 파괴")
                    self._region_escape_cd = now + 90.0
                    self._region_hist.clear()
                    self.explorer.fail_current()
                    self._wf_escape_until = now + 40.0
                    self._wf_anchor = None
                    self._enter(self.WALL_FOLLOW, now)

        # stuck → RECOVER (SPIN/DONE/RECOVER 제외)
        if self.state in (self.EXPLORE, self.SEEK, self.GOTO_TARGET,
                          self.WALL_FOLLOW, self.VISIT, self.RETURN) \
                and self._update_stuck(now, pose):
            if now - self._last_stuck_t > 60.0:
                self._stuck_count = 0        # 오래 잘 다녔으면 이력 리셋
            self._stuck_count += 1
            # 반복 갇힘 = 후진+회전으로 부족한 지형(탁자 밑 등) → creep 격상
            self._recover_escalate = self._stuck_count >= 2
            self._last_stuck_t = now
            self._recover_after = self.state
            if self.state == self.SEEK:
                # SEEK 중 갇힘 = 시선 방향이 물리적으로 막힘 — SEEK로
                # 복귀하면 같은 자리서 응시→갇힘 무한 루프 (타임아웃은
                # 재진입마다 리셋돼 영영 안 걸린다). 탐사로 돌리고 잠시
                # SEEK 금지 → 지도가 자라면 다른 각도로 재접근한다.
                self._recover_after = self.EXPLORE
                self._seek_block_until = max(self._seek_block_until,
                                             now + 15.0)
            self._recover_phase = ("backup", now)
            self._pose_hist.clear()
            if self._stuck_count >= 3 and self.state == self.EXPLORE:
                self.explorer.fail_current()
                self._stuck_count = 0
            self._enter(self.RECOVER, now)

        # 가구 구역을 경로 페널티로 전달 (사람 캡슐과 같은 채널, TTL 적용)
        fz = self._furniture_valid(now)
        self.planner.furniture_xy = fz

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
            self.WALL_FOLLOW: self._do_wall_follow,
            self.EXPLORE: self._do_explore,
            self.SEEK: self._do_seek,
            self.VISIT: self._do_visit,
            self.GOTO_TARGET: self._do_goto,
            self.RETURN: self._do_return,
            self.RECOVER: self._do_recover,
            self.DONE: self._do_done,
        }[self.state]
        v, w = handler(now, pose, angles, ranges)

        # 소프트 경로의 '좁은 구간' 근처에서만 회피 마진 완화 —
        # 계획이 허용한 틈을 회피가 도로 막는 모순 방지 (열린 구간은
        # 표준 마진·순항 속도 유지)
        if self.planner.last_plan_soft and any(
                dist(pose, s) < 0.6 for s in self.planner.soft_spots):
            self.avoider.relax_until = max(self.avoider.relax_until,
                                           now + 1.0)
        # 로봇이 표준 팽창 영역 '안'에 서 있으면(문틈·가구 틈 한가운데)
        # 마진 완화 — 표준 마진은 이 자리 자체를 부정하므로 그대로는
        # 문설주에 막혀 옴짝달싹 못 한다 (0.5s 스로틀: 팽창 마스크 비용).
        # 단, '경로 추종 중'일 때만 — 벽 추종처럼 계획 없는 주행에서
        # 상시 완화하면 벽에 붙어 그라인딩하는 사고가 난다.
        if now - getattr(self, "_pinch_check_t", -1e9) >= 0.5:
            self._pinch_check_t = now
            r_infl = self.cfg.robot.robot_radius \
                + self.cfg.plan.inflate_margin
            ix, iy = self.grid.world_to_grid(pose[0], pose[1])
            pinched = (0 <= ix < self.grid.n and 0 <= iy < self.grid.n
                       and bool(self.grid.inflated_mask(r_infl)[iy, ix]))
            if pinched and self.planner.waypoints \
                    and self.state in (self.EXPLORE, self.SEEK, self.VISIT,
                                       self.GOTO_TARGET, self.RETURN):
                self.avoider.relax_until = max(self.avoider.relax_until,
                                               now + 1.2)

        # 회피는 전 상태 공통 (후진 recovery 중에는 전방 회피 제외)
        if self.state not in (self.RECOVER, self.DONE):
            v, w, blocked_long = self.avoider.apply(
                v, w, angles, ranges, now, dynamic=dyn, pose=pose,
                person_xy=self.planner.avoid_xy,
                person_vel=self.planner.avoid_vel)
            # SPIN은 제자리 스캔이 목적 — 실제 탈출 상황이 아니면
            # 어떤 제안도 전진시키지 못하게 (오분류가 나선 전진 유발 방지)
            if self.state == self.SPIN \
                    and self.avoider.last_mode not in ("escape", "blind",
                                                       "sprint"):
                v = 0.0
            if blocked_long:
                # 오래 막힘: 지도에 반영됐을 것 → 강제 재계획
                self.planner.plan_to(pose, self.planner.goal or pose[:2],
                                     now, force=True) \
                    if self.planner.goal else None
                # frontier 포기는 3진 아웃 (12s 창) — 1.2s 막힘 한 번으로
                # 방금 들어간 방의 frontier를 블랙리스트하면 "방에 들어
                # 갔다가 움찔하고 바로 나가는" 동선이 된다. 재계획으로
                # 우회부터 시도하고, 반복해서 막힐 때만 포기한다.
                if self.state == self.EXPLORE:
                    self._bl_hist = [t for t in
                                     getattr(self, "_bl_hist", [])
                                     if now - t < 12.0]
                    self._bl_hist.append(now)
                    if len(self._bl_hist) >= 3:
                        self._bl_hist = []
                        self.explorer.fail_current()

        # 명령 평활화 — 채터로 인한 '문 앞 떨림' 제거.
        # 안전 비대칭: 감속·정지·후진은 즉시 반영, 가속만 제한.
        # 조향은 변화율 제한 (원형 로봇의 회전은 접촉과 무관 → 안전 영향 없음)
        dt_cmd = 0.064 if self._last_step_t is None \
            else max(1e-3, min(0.2, now - self._last_step_t))
        self._last_step_t = now
        dv_max = 0.35 * dt_cmd            # 가속 한계 (0→최고속 ~0.6s)
        dw_max = 6.0 * dt_cmd             # 조향 변화 한계 (풀스윙 ~0.5s)
        if v > 0.0:
            v = min(v, max(0.0, self._v_prev) + dv_max)
        w = max(self._w_prev - dw_max, min(self._w_prev + dw_max, w))
        if self.state == self.DONE:
            v, w = 0.0, 0.0
        self._v_prev, self._w_prev = v, w

        self._last_theta = pose[2]
        info = {
            "state": self.state,
            "start": self.start_xy,
            "distance_to_start": dist(pose, self.start_xy),
            "dyn_count": int(dyn.sum()) if dyn is not None else 0,
            "avoidance_mode": self.avoider.last_mode,
            "person_xy": self.planner.avoid_xy,
            "person_velocity": self.planner.avoid_vel,
            "detection_status": {"visible": self.detector.visible,
                                 "rejection": self.detector.rejection_reason,
                                 "position_valid": valid_estimate},
            "target_est": self.target_est,
            "goal": self.planner.goal,
            "waypoints": list(self.planner.waypoints),
            "crumbs": list(self.crumbs.crumbs),
            "found": self.target_found,
            "visited": len(self.visited_targets),
            "reached": self.target_reached,
            "success": (self.state == self.DONE
                        and len(self.visited_targets) >= cfg.mission.num_targets),
            "visited_targets": list(self.visited_targets),
            "candidates": [c["xy"] for c in self.candidates],
            "furniture": list(self.planner.furniture_xy),
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
        self._last_scan_pose = pose[:2]
        self._last_scan_time = now
        # 사람이 접근 중이면 스캔 중단 — 정지 스캔은 서서 치이는 자세다.
        # 지나간 뒤 주기 로직이 다시 스핀을 잡는다.
        if self.planner.avoid_xy is not None \
                and dist(pose, self.planner.avoid_xy) < 0.95:
            self.last_spin_t = now
            self._force_full_spin = self._force_full_spin \
                or not self._initial_spin_done
            self._initial_spin_done = True
            self._enter(self._spin_return_state, now)
            return 0.0, 0.0
        dth = wrap_angle(pose[2] - self._last_theta)
        full = self._force_full_spin or (
            not self._initial_spin_done and cfg.explore.initial_full_spin)
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
            self._initial_spin_done = True   # 초기 스윕도 여기로 끝난다
            self._enter(self._spin_return_state, now)
            return 0.0, 0.0
        return 0.0, -cfg.robot.max_w * 0.6

    def _do_wall_follow(self, now, pose, angles, ranges):
        """우수법 벽 추종 — 외곽을 일주하며 지도 골격을 완성한다.

        LiDAR P제어: 우측 벽과 wall_dist 유지, 전방 벽에서 좌회전,
        볼록 코너(벽 상실)에서 우회전으로 벽 재획득. 종료는 루프 폐합
        (경로 6m+ 이동 후 시작 앵커 0.6m 복귀) 또는 타임아웃 →
        frontier 탐사가 실내 잔여 미탐색 구역을 마무리한다."""
        cfg = self.cfg
        # 맴돌기-탈출 모드: 제한 시간 동안만 벽을 타고 구역을 벗어난 뒤
        # frontier 탐사로 복귀 (지도 우선 모드가 아니어도 사용)
        if self._wf_escape_until is not None \
                and now > self._wf_escape_until:
            self._wf_escape_until = None
            self._wf_active = False
            self._enter(self.EXPLORE, now)
            return 0.0, 0.0
        # 주기 카메라 스윕 (후보 발견은 카메라 몫 — 벽만 보면 못 찾는다)
        if cfg.explore.spin_period > 0 \
                and now - self.last_spin_t > cfg.explore.spin_period:
            self.last_spin_t = now
            if self._spin_worthwhile(now, pose):
                self.spin_accum = 0.0
                self._spin_phase = "left"
                self._spin_return_state = self.WALL_FOLLOW
                self._enter(self.SPIN, now)
                return 0.0, 0.0
        # 루프 폐합 / 타임아웃 판정
        if self._wf_anchor is None:
            self._wf_anchor = (pose[0], pose[1])
            self._wf_path = 0.0
            self._wf_last = (pose[0], pose[1])
        self._wf_path += dist(pose, self._wf_last)
        self._wf_last = (pose[0], pose[1])
        closed = self._wf_path > 6.0 \
            and dist(pose, self._wf_anchor) < 0.6
        if closed or now - self.state_since > cfg.explore.wall_timeout:
            why = "루프 폐합" if closed else "타임아웃"
            print(f"[mission] 벽 추종 종료({why}, {self._wf_path:.1f}m) "
                  f"→ frontier 잔여 탐사")
            self._wf_active = False
            self._enter(self.EXPLORE, now)
            return 0.0, 0.0
        # 우수법 P제어 (LocalAvoider 클램프가 뒤에서 최종 안전 보정)
        d_front = self._sector_min(angles, ranges, 0.0, halfwidth=0.45)
        d_right = self._sector_min(angles, ranges, -math.pi / 2,
                                   halfwidth=0.5)
        d_diag = self._sector_min(angles, ranges, -math.pi / 4,
                                  halfwidth=0.35)
        if d_front < 0.5:                    # 전방 벽 → 내측(좌) 회전
            return 0.04, 0.9
        if d_right > 1.3 and d_diag > 1.1:   # 볼록 코너: 벽 상실 → 우로
            return cfg.robot.max_v * 0.55, -0.75
        err = d_right - cfg.explore.wall_dist
        w = -1.8 * err                       # 멀면 우로(-), 가까우면 좌로(+)
        if d_diag < cfg.explore.wall_dist * 0.8:
            w += 0.6                         # 비스듬히 접근 중 → 좌로 보정
        w = max(-1.0, min(1.0, w))
        v = cfg.robot.max_v * max(0.4, min(1.0, (d_front - 0.3) / 0.6))
        if d_right < 0.25 or d_front < 0.55:
            v = min(v, 0.08)                 # 벽·모서리 초근접은 저속으로
        return v, w

    def _do_visit(self, now, pose, angles, ranges):
        """지도 완성 후 기억해둔 후보를 방문해 진짜 목표인지 확인.

        도착하면 후보 방향을 응시 — 확정(안정 추정+YOLO)은 step() 공통부가
        처리해 GOTO_TARGET으로 넘어간다. 응시 시간 내 확정 실패 = 오탐."""
        cfg = self.cfg
        if self._visit_cand is None:
            self._visit_cand = self._pop_candidate(pose)
            self._visit_gaze_t = None
            self._visit_fail_t = None
            if self._visit_cand is None:
                # 후보 소진: 목표가 남았으면 기존(목표 우선) 모드로 폴백
                if len(self.visited_targets) < cfg.mission.num_targets \
                        and self._map_first_active:
                    print("[mission] 후보 소진 — 목표 우선 탐사로 폴백")
                    self._map_first_active = False
                    self._final_spin_done = False
                    self.explorer.blacklist.clear()
                    self._enter(self.EXPLORE, now)
                else:
                    self._enter(self.RETURN, now)
                return 0.0, 0.0
            print(f"[mission] 후보 방문 → "
                  f"({self._visit_cand[0]:+.2f},{self._visit_cand[1]:+.2f}) "
                  f"남은 후보 {len(self.candidates)}개")
        cx, cy = self._visit_cand
        d = dist(pose, (cx, cy))
        # 실물 우선 근접 정지: 후보 좌표는 단안 오차(±25%)를 품고 있고
        # 사과는 LiDAR에 안 보여 회피가 못 막는다 — 카메라 각폭이 크면
        # (≈0.4m 이내) 좌표와 무관하게 그 자리서 응시로 전환.
        det = self.detector.last if self.detector.visible else None
        if det is not None and det.ang_width > 0.25:
            if self._visit_gaze_t is None:
                self._visit_gaze_t = now
            self._pose_hist.clear()          # 의도된 정지 — stuck 오탐 방지
            if now - self._visit_gaze_t > 7.0:
                print(f"[mission] 후보 ({cx:+.2f},{cy:+.2f}) 확정 실패 — "
                      f"오탐 처리, 다음 후보로")
                self._decoys.append((cx, cy))
                self._visit_cand = None
                return 0.0, 0.0
            b0 = det.bearing
            if abs(b0) > 0.12:
                return 0.0, max(-cfg.robot.max_w,
                                min(cfg.robot.max_w, 2.0 * b0))
            return 0.0, 0.0
        if d > cfg.mission.target_standoff + 0.35:
            # 접근: 후보 앞 standoff 지점으로
            ratio = max(0.0, (d - cfg.mission.target_standoff) / d)
            goal = (pose[0] + (cx - pose[0]) * ratio,
                    pose[1] + (cy - pose[1]) * ratio)
            ok = self.planner.plan_to(pose, goal, now)
            if not ok:
                if self._visit_fail_t is None:
                    self._visit_fail_t = now
                elif now - self._visit_fail_t > 4.0:
                    print("[mission] 후보 접근 불가 — 다음 후보로")
                    self._visit_cand = None
                return 0.0, 0.0
            self._visit_fail_t = None
            return self.planner.follow(pose)
        # 도착: 후보를 정면에 두고 응시 (확정은 공통부에서)
        b = wrap_angle(math.atan2(cy - pose[1], cx - pose[0]) - pose[2])
        self._pose_hist.clear()              # 의도된 정지 — stuck 오탐 방지
        if self._visit_gaze_t is None:
            self._visit_gaze_t = now
        if now - self._visit_gaze_t > 7.0:
            print(f"[mission] 후보 ({cx:+.2f},{cy:+.2f}) 확정 실패 — 오탐 "
                  f"처리, 다음 후보로")
            self._decoys.append((cx, cy))    # 재등록 방지
            self._visit_cand = None
            return 0.0, 0.0
        if abs(b) > 0.12:
            return 0.0, max(-cfg.robot.max_w,
                            min(cfg.robot.max_w, 2.0 * b))
        return 0.0, 0.0

    def _do_explore(self, now, pose, angles, ranges):
        cfg = self.cfg
        # 주기적 카메라 스캔 — 단, 지난 스윕 이후 지도가 새로 자랐을
        # 때만 (이미 훑은 구역 재통과 중의 회전은 순수 시간 낭비)
        if cfg.explore.spin_period > 0 \
                and now - self.last_spin_t > cfg.explore.spin_period:
            self.last_spin_t = now      # 게이트 불통과 시에도 주기 리셋
            if self._spin_worthwhile(now, pose):
                self.spin_accum = 0.0
                self._spin_phase = "left"
                self._spin_return_state = self.EXPLORE
                self._enter(self.SPIN, now)
                return 0.0, 0.0

        target = self.explorer.update(pose, person_xy=self.planner.avoid_xy,
                                      furniture=self.planner.furniture_xy, now=now)
        # frontier 도착 = 새 시야가 열린 순간 → 카메라 스윕 1회
        # (여기도 신규 지도 게이트 — 열린 게 없으면 그냥 다음으로)
        if self.explorer.just_reached:
            self.explorer.just_reached = False
            if self._spin_worthwhile(now, pose):
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
            # 지도 완성 시점: 기억해둔 후보 방문 단계로
            if self._map_first_active \
                    and len(self.visited_targets) < cfg.mission.num_targets:
                self._prune_candidates()
                if self.candidates:
                    print(f"[mission] 지도 완성 — 후보 "
                          f"{len(self.candidates)}개 방문 시작")
                    self._visit_phase = True
                    self._visit_cand = None
                    self._enter(self.VISIT, now)
                    return 0.0, 0.0
                print("[mission] 지도 완성 — 후보 없음, 목표 우선 모드 폴백")
                self._map_first_active = False
                self._final_spin_done = False
                self.explorer.blacklist.clear()
                return 0.0, 0.0
            self._enter(self.RETURN, now)
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
        if self.detector.visible and not self._target_tracking_valid:
            self._seek_block_until = now + 5.0
            self._enter(self.EXPLORE, now)
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
        if now - self._target_seen_t > cfg.detection.target_lost_timeout:
            print("[mission] 목표 재관측 시간 초과 — 오래된 추정 폐기, 탐색 재개")
            self.target_found = False
            self.target_est = self._target_anchor = None
            self._est_hist.clear()
            self.planner.waypoints = []
            self._seek_block_until = now + 8.0
            self._enter(self.EXPLORE, now)
            return 0.0, 0.0
        if self.target_est is None:      # 추정 소실 (이례적) → 탐색 복귀
            self.target_found = False
            self._enter(self.EXPLORE, now)
            return 0.0, 0.0
        goal = self._approach_goal(pose)
        # 최소 체류 0.3s: 확정 즉시 방문 판정되면 (추정이 이미 코앞)
        # 확정 프레임 저장·상태 표시가 전부 스킵되는 원턴 방문이 된다
        if dist(pose, goal) < cfg.mission.target_reach_tol \
                and now - self.state_since > 0.3:
            # 현재 목표 방문 완료 — 전부 채웠으면 복귀, 남았으면 탐색 재개
            self.visited_targets.append(self.target_est)
            k = len(self.visited_targets)
            print(f"[mission] 목표 {k}/{cfg.mission.num_targets} 방문 완료 "
                  f"({self.target_est[0]:+.2f},{self.target_est[1]:+.2f})")
            self.target_found = False
            self.target_est = None
            self._target_anchor = None
            self._est_hist.clear()
            self._seek_block_until = now + 8.0   # 방금 그 사과 응시 방지
            if k >= cfg.mission.num_targets:
                self.target_reached = True
                self._enter(self.RETURN, now)
            elif self._map_first_active:
                if self._visit_phase:
                    self._visit_cand = None      # 다음 후보로
                self._enter(self._mapping_state(), now)
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
                if self._map_first_active:
                    self._enter(self._mapping_state(), now)
                else:
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
        """갇힘 탈출: 후진 → 최대 여유 틈으로 정렬 → (반복 갇힘 시) 그
        틈을 향해 저속 전진(creep). 표준 회피가 정지시키는 좁은 틈도
        여기서는 빠져나가야 하므로 RECOVER는 LocalAvoider를 거치지
        않는다 — 대신 자체 섹터 거리 가드로 접촉을 막는다."""
        cfg = self.cfg
        # '실제로 움직이는' 사람이 근접해 있으면 복구 기동 금지 — RECOVER는
        # LocalAvoider를 우회하므로 동적 위협에는 눈이 먼다. 속도 조건이
        # 없으면 문설주 유령(정지 오분류)이 RECOVER를 영구 차단한다.
        vel = self.planner.avoid_vel
        moving = vel is not None and math.hypot(vel[0], vel[1]) > 0.08
        if self.planner.avoid_xy is not None and moving \
                and dist(pose, self.planner.avoid_xy) < 0.9:
            self._recover_phase = None
            self._enter(self._recover_after, now)
            return 0.0, 0.0
        phase, t0 = self._recover_phase
        esc = self._recover_escalate
        if phase == "backup":
            # 후방 가드 섹터를 넓게(±0.6rad) — 빗겨 선 문설주·코너에
            # 후진으로 박는 사고 방지 (좁으면 축 밖 장애물을 놓친다)
            rear = self._sector_min(angles, ranges, math.pi, halfwidth=0.6)
            t_back = 2.2 if esc else 1.2      # 격상 시 더 길게 물러남
            if now - t0 < t_back and rear > 0.16:
                return -0.08, 0.0
            # goal 보너스는 계획 기반 상태로 복귀할 때만 — WALL_FOLLOW로
            # 돌아갈 때의 planner.goal은 낡은 값이라 오도한다
            goal = self.planner.goal if self._recover_after in (
                self.EXPLORE, self.SEEK, self.VISIT,
                self.GOTO_TARGET, self.RETURN) else None
            self._escape_heading = self._best_gap_heading(
                pose, angles, ranges, goal_xy=goal)
            self._recover_phase = ("align", now)
            return 0.0, 0.0
        if phase == "align":
            err = wrap_angle(self._escape_heading - pose[2])
            if abs(err) > 0.18 and now - t0 < 3.0:
                w = max(-cfg.robot.max_w, min(cfg.robot.max_w, 2.2 * err))
                return 0.0, w
            if esc:
                self._recover_phase = ("creep", now)
                self._creep_start = (pose[0], pose[1])
                # 점 목표 추종: 고정 헤딩 유지는 횡편차를 보정하지 못해
                # 비스듬히 진입하면 문설주 쪽으로 흘러가 스친다
                self._creep_goal = (
                    pose[0] + 1.0 * math.cos(self._escape_heading),
                    pose[1] + 1.0 * math.sin(self._escape_heading))
                return 0.0, 0.0
            return self._finish_recover(now, pose)
        # creep: 틈 중앙을 향해 저속 전진 — 양옆이 막혀 있는 동안(문설주
        # 사이)은 멈추지 않고 장애물이 사라질 때까지 직진해야 한다.
        # 중간에 서면 반쯤 걸친 자리에서 도로 갇힌다.
        import numpy as np
        front = self._sector_min(angles, ranges, 0.0, halfwidth=0.35)
        creeped = dist(pose, self._creep_start)
        # 코리도식 측방 여유: 전방 0.45m 안 빔의 좌/우 최소 측방 오프셋
        # (넓은 각도 섹터 방식은 뒤쪽 벽까지 섞여 문틈 중앙 유지가 안 됨)
        aa = np.asarray(angles)
        rr_ = np.asarray(ranges)
        okb = np.isfinite(rr_) & (rr_ > 0.05)
        fwd = rr_[okb] * np.cos(aa[okb])
        lat = rr_[okb] * np.sin(aa[okb])
        # The narrow front sector misses a table leg beside the nose. Check the
        # full swept footprint before creeping, while leaving passable side gaps.
        radius = cfg.robot.robot_radius + .015
        corridor = (fwd > 0) & (np.abs(lat) < radius)
        contact_distance = float('inf')
        if corridor.any():
            # Distance until a translated circular footprint touches each hit.
            # A rectangular front guard falsely blocks passable doorway corners.
            contact_distance = float(np.min(
                fwd[corridor] - np.sqrt(radius ** 2 - lat[corridor] ** 2)))
        near = (fwd > 0.0) & (fwd < 0.45) & (np.abs(lat) < 0.4)
        lsel = near & (lat > 0)
        rsel = near & (lat < 0)
        l_lat = float(lat[lsel].min()) if lsel.any() else 0.4
        r_lat = float((-lat[rsel]).min()) if rsel.any() else 0.4
        flanked = min(l_lat, r_lat) < 0.30
        max_d = 1.0 if flanked else 0.55
        if front < 0.16 or contact_distance < .035 or creeped > max_d or now - t0 > 9.0:
            return self._finish_recover(now, pose)
        err = wrap_angle(math.atan2(self._creep_goal[1] - pose[1],
                                    self._creep_goal[0] - pose[0]) - pose[2])
        w = max(-0.8, min(0.8, 1.8 * err))
        if min(l_lat, r_lat) < 0.25:     # 틈 안: 중앙 유지가 헤딩보다 우선
            w += max(-0.5, min(0.5, 1.6 * (l_lat - r_lat)))
        v_c = 0.05 if min(l_lat, r_lat) < 0.2 else 0.07
        return v_c, max(-0.8, min(0.8, w))

    def _finish_recover(self, now, pose):
        self._recover_phase = None
        # 탈출 직후 잠시 안전 마진 완화 — 표준 마진이 같은 자리에 도로
        # 가두는 것을 방지 (소프트 경로 추종 허용)
        self.avoider.relax_until = max(self.avoider.relax_until, now + 3.0)
        # 벽 추종 중 '반복' 갇힘 = 벽 추종이 통하지 않는 협소 지형(우리
        # 같은) — 계획 기반 탐사로 넘긴다: 소프트 경로+저속이 틈을 뚫는다
        after = self._recover_after
        if self._recover_escalate and after == self.WALL_FOLLOW:
            print("[mission] 반복 갇힘 — 벽 추종 중단, 계획 탐사로 전환")
            self._wf_active = False
            after = self.EXPLORE
        if self.planner.goal:
            self.planner.plan_to(pose, self.planner.goal, now, force=True)
        self._enter(after, now)
        return 0.0, 0.0

    def _do_done(self, now, pose, angles, ranges):
        return 0.0, 0.0
