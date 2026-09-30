"""Webots 메인 컨트롤러 — sar 패키지의 미션 로직을 실로봇 루프에 연결.

실행 방법 (둘 중 하나):
1) Webots 월드에서 robot controller = "sar_controller" 로 두고 실행
   (Webots 환경설정의 Python command를 프로젝트 .venv 파이썬으로 지정)
2) extern 컨트롤러: 월드에서 controller = "<extern>" 으로 두고
   $ /Applications/Webots.app/Contents/MacOS/webots-controller \
       --stdout-redirect webots/controllers/sar_controller/sar_controller.py

당일 어댑테이션 포인트는 전부 sar/config.py 와 robot_io.py 에 있다.
"""
import math
import os
import sys
import time
import json

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 저장소 루트를 import 경로에 추가 — 'sar' 패키지가 보일 때까지 상위로
# 탐색 (webots/controllers/... 배치와 저장소 루트 직속 controllers/...
# 배치 모두 지원)
_ROOT = os.path.dirname(os.path.abspath(__file__))
for _ in range(6):
    if os.path.isdir(os.path.join(_ROOT, "sar")):
        break
    _ROOT = os.path.dirname(_ROOT)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from sar.config import default_config           # noqa: E402
from sar.detection import cv2                  # noqa: E402
from sar.mapping import OccupancyGrid           # noqa: E402
from sar.odometry import PoseEstimator          # noqa: E402
from sar.state_machine import Mission           # noqa: E402
from sar.viz import MapViz, save_rgb            # noqa: E402
from sar.runtime import RuntimeStats, write_json, save_grid  # noqa: E402
from robot_io import RobotIO                    # noqa: E402

SNAPSHOT_EVERY_S = 1.0        # wall seconds; 지도 PNG 저장 주기 (0이면 끔)
SNAPSHOT_DIR = os.environ.get("SAR_OUTPUT_DIR", os.path.join(_ROOT, "out"))
# YOLO 가구 스캔: 탁자·의자류를 미리 인식해 밑으로 파고드는 경로를 예방
FURNITURE_CLASSES = {"dining table", "chair", "couch", "bench", "bed"}
FURNITURE_SCAN_S = 3.0


def main():
    cfg = default_config()
    override = os.path.join(_ROOT, "config_override.json")
    if os.path.exists(override):
        cfg.apply_overrides(override)
        print(f"[sar] config override 적용: {override}")

    io = RobotIO(cfg)
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    write_json(os.path.join(SNAPSHOT_DIR, "live_status.json"),
               {"state": "STARTING", "updated_at": time.time()})
    try:
        if io.lidar is None or io.camera is None:
            raise RuntimeError("SAR requires both LiDAR and camera")
        run_mission(cfg, io)
    except Exception as e:
        io.drive(0.0, 0.0)
        write_json(os.path.join(SNAPSHOT_DIR, "live_status.json"),
                   {"state": "ERROR", "error": str(e), "updated_at": time.time()})
        io.step()  # flush the zero command to Webots before exiting
        raise
    finally:
        io.drive(0.0, 0.0)


def run_mission(cfg, io):
    dt = io.timestep / 1000.0
    start = (cfg.mission.start_x, cfg.mission.start_y, cfg.mission.start_theta)
    grid = OccupancyGrid(cfg, center_xy=start[:2])
    estimator = PoseEstimator(cfg, start)
    mission = Mission(cfg, grid)
    viz = MapViz(grid)
    if SNAPSHOT_EVERY_S > 0:
        os.makedirs(SNAPSHOT_DIR, exist_ok=True)

    print(f"[sar] timestep={io.timestep}ms lidar={io.lidar is not None} "
          f"camera={io.camera is not None}")
    mission.detector.yolo_warmup()
    stats = RuntimeStats()

    now = 0.0
    step_i = 0
    last_snap = -1e9
    last_cam = -1e9
    last_status = -1e9
    last_export = -1e9
    last_yolo = -1e9
    yolo_cache = []
    camera_frame = None
    camera_detection = None
    camera_time = None
    last_state = None
    done_logged = False
    found_saved = False
    last_visited = 0
    last_furn = -1e9
    last_match = -1e9
    previous_v = 0.0
    while True:
        tick_start = time.perf_counter()
        if io.step() == -1:
            break
        sensors_start = time.perf_counter()
        now += dt
        step_i += 1
        left, right = io.wheel_positions()
        pose = estimator.update(left, right, compass_raw=io.compass_raw())
        angles, ranges = io.lidar_scan()
        if angles is None:
            io.drive(0.0, 0.0)
            continue
        if now - last_match >= cfg.localization.period:
            pose = estimator.correct_with_scan(grid, angles, ranges, commanded_v=previous_v)
            last_match = now
        # 지도 갱신은 Mission.step 내부에서 (동적 빔 제외 후) 수행
        img = io.camera_image()
        yolo_start = time.perf_counter()
        yolo_due = img is not None and now - last_yolo >= cfg.detection.yolo_period
        if yolo_due:
            last_yolo = now
            yolo_cache = mission.detector.yolo_boxes(img)
        mission_start = time.perf_counter()
        v, w, info = mission.step(now, pose, angles, ranges, img)
        io.drive(v, w)
        previous_v = v
        output_start = time.perf_counter()
        if yolo_due:
            # Retain the exact frame used for these boxes, not a later image.
            camera_frame, camera_time = img, now
            camera_detection = mission.detector.last if mission.detector.visible else None

        # YOLO 가구 스캔 — 탁자류 방향의 LiDAR 최소거리로 위치를 잡아
        # '밑으로 파고들지 않을 구역'으로 등록 (갇힘의 예방 레이어)
        if yolo_due and now - last_furn >= FURNITURE_SCAN_S and img is not None \
                and getattr(mission.detector, "_yolo", None) is not None:
            last_furn = now
            try:
                import numpy as _np
                from sar.geometry import wrap_angle as _wrap
                h_, w_ = img.shape[:2]
                f_px = (w_ / 2) / math.tan(cfg.camera.hfov / 2)
                a_arr = _np.asarray(angles)
                r_arr = _np.asarray(ranges)
                for name, conf, (x0, y0, x1, y1) in yolo_cache:
                    if name not in FURNITURE_CLASSES or conf < 0.45:
                        continue
                    if y1 < h_ * 0.5:     # 화면 위쪽 절반뿐 → 원거리/벽면
                        continue
                    # bbox 각도 범위의 LiDAR 프로파일로 '투과성' 검증:
                    # 탁자·의자는 다리 사이로 빔이 통과해 최근접보다
                    # 0.4m+ 깊은 빔이 상당수 나온다. 문·벽·판형 가구는
                    # 전 빔이 같은 평면에 꽂힘 → 가구 구역 아님 (문을
                    # 가구로 등록해 통로를 회피하던 오인식 차단)
                    b_lo = math.atan2(w_ / 2.0 - x1, f_px) + cfg.camera.mount_yaw
                    b_hi = math.atan2(w_ / 2.0 - x0, f_px) + cfg.camera.mount_yaw
                    bc = 0.5 * (b_lo + b_hi)
                    hw = 0.5 * (b_hi - b_lo) + 0.05
                    sel = _np.abs(_wrap(a_arr - bc)) < hw
                    rr2 = r_arr[sel]
                    rr2 = rr2[_np.isfinite(rr2) & (rr2 > 0.15)]
                    if rr2.size < 5:
                        continue
                    d = float(rr2.min())
                    if d > 2.5:
                        continue
                    deep_frac = float((rr2 > d + 0.4).mean())
                    if deep_frac < 0.25:
                        continue          # 판형(문/벽/장) — 등록 금지
                    fx = pose[0] + d * math.cos(pose[2] + bc)
                    fy = pose[1] + d * math.sin(pose[2] + bc)
                    mission.note_furniture(
                        (fx, fy), now=now,
                        label=f"{name} {conf:.2f} 투과{deep_frac:.0%}")
            except Exception as e:
                print(f"[sar] 가구 스캔 실패: {e}")

        if info["state"] != last_state:
            last_state = info["state"]
            print(f"[sar] t={now:6.1f}s → {last_state} "
                  f"pose=({pose[0]:+.2f},{pose[1]:+.2f}) "
                  f"visited={info['visited']}/{cfg.mission.num_targets} "
                  f"home={info['distance_to_start']:.2f}m")
        # 목표 확정 순간의 카메라 프레임 저장 — "무엇을 목표로 봤는가"
        # 검증용 (디코이 오인 디버깅에 결정적)
        if info["found"] and not found_saved and img is not None:
            found_saved = True
            try:
                os.makedirs(SNAPSHOT_DIR, exist_ok=True)
                save_rgb(os.path.join(SNAPSHOT_DIR, "found_frame.png"), img)
                print(f"[sar] 목표 확정 프레임 저장 — est={info['target_est']}")
            except Exception as e:
                print(f"[sar] found_frame 저장 실패: {e}")
        # 방문 확정별 프레임 — "무엇을 사과로 봤는가" 사후 검증의 핵심
        if info.get("visited", 0) != last_visited:
            last_visited = info["visited"]
            frame = getattr(mission, "_confirm_img", None)
            if frame is not None:
                try:
                    save_rgb(os.path.join(
                        SNAPSHOT_DIR, f"found_{last_visited}.png"), frame)
                    print(f"[sar] 방문 {last_visited} 확정 프레임 저장")
                except Exception as e:
                    print(f"[sar] 방문 프레임 저장 실패: {e}")
        wall = time.perf_counter()
        if SNAPSHOT_EVERY_S > 0 and wall - last_snap >= SNAPSHOT_EVERY_S:
            last_snap = wall
            # 경량 렌더 — matplotlib figure는 제어 루프를 수백 ms 블록
            viz.save_fast(os.path.join(SNAPSHOT_DIR, "live_map.png"),
                          pose=pose, info=info)
        # Publish at most 4Hz wall time; inference runs at 4Hz simulation time.
        if wall - last_cam >= .25 and yolo_due and camera_frame is not None:
            last_cam = wall
            viz.save_camera(os.path.join(SNAPSHOT_DIR, "live_cam.png"),
                            camera_frame, det=camera_detection,
                            yolo_boxes=yolo_cache)
        stats.record(webots_wait=sensors_start - tick_start,
                     sensors=yolo_start - sensors_start,
                     yolo=mission_start - yolo_start,
                     mission=output_start - mission_start,
                     output=time.perf_counter() - output_start)
        if wall - last_status >= 1.0 or info["state"] == Mission.DONE and not done_logged:
            last_status = wall
            status = stats.snapshot(now)
            status.update(info)
            status.update(pose=list(pose), num_targets=cfg.mission.num_targets,
                          localization="encoder + compass + local LiDAR matching" if estimator.matcher else "encoder + compass",
                          scan_corrections=estimator.matcher.corrections if estimator.matcher else 0,
                          known_area_m2=float((~grid.unknown_mask()).sum()) * grid.res ** 2,
                          yolo_device=mission.detector.yolo_device,
                          yolo_ms=mission.detector.yolo_ms,
                          yolo_period_s=cfg.detection.yolo_period,
                          camera_sim_time=camera_time,
                          objects=[dict(name=n, confidence=c, bbox=b) for n, c, b in yolo_cache],
                          command=dict(v=v, w=w),
                          return_method="breadcrumbs" if mission._crumb_mode else "A*",
                          hsv_backend="OpenCV" if cv2 is not None else "NumPy")
            write_json(os.path.join(SNAPSHOT_DIR, "live_status.json"), status)
            with open(os.path.join(SNAPSHOT_DIR, "telemetry.jsonl"), "a", encoding="utf-8") as log:
                log.write(json.dumps(status, ensure_ascii=False) + "\n")
        if wall - last_export >= 5.0 or info["state"] == Mission.DONE and not done_logged:
            last_export = wall
            save_grid(os.path.join(SNAPSHOT_DIR, "map_data.npz"), grid, pose)
        if info["state"] == Mission.DONE and not done_logged:
            done_logged = True
            verdict = "SUCCESS" if info["success"] else "INCOMPLETE"
            print(f"[sar] MISSION {verdict} t={now:.1f}s "
                  f"visited={info['visited']}/{cfg.mission.num_targets} "
                  f"home={info['distance_to_start']:.3f}m — 정지 유지")
            io.drive(0.0, 0.0)


if __name__ == "__main__":
    main()
