"""목표 물체 색 기반 탐지 (OpenCV 가속, 미설치 시 NumPy 폴백).

- RGB → HSV 변환 후 config의 HSV 범위로 마스크
- 최소 픽셀 수 + N프레임 연속 확인으로 오탐 억제
- 출력 bearing: 로봇 프레임, +값 = 왼쪽(CCW). 화면 왼쪽 = +bearing
"""
from pathlib import Path
import time

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


def rgb_to_hsv(img):
    """img: (H,W,3) uint8 RGB → (H,W,3) float32, H∈[0,360), S,V∈[0,1]."""
    f = img.astype(np.float32) / 255.0
    if cv2 is not None:
        # Float input preserves H=degrees, S/V=0..1 (uint8 uses H=0..180).
        return cv2.cvtColor(f, cv2.COLOR_RGB2HSV)
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    diff = mx - mn + 1e-9
    h = np.zeros_like(mx)
    m = mx == r
    h[m] = (60 * ((g - b) / diff) % 360)[m]
    m = mx == g
    h[m] = (60 * ((b - r) / diff) + 120)[m]
    m = mx == b
    h[m] = (60 * ((r - g) / diff) + 240)[m]
    s = np.where(mx > 1e-9, diff / (mx + 1e-9), 0.0)
    return np.stack([h % 360, s, mx], axis=-1)


class Detection:
    __slots__ = ("bearing", "pixels", "cx_ratio", "area_rows", "px_width",
                 "ang_width", "clipped", "bbox", "aspect", "v_clipped",
                 "fill")

    def __init__(self, bearing, pixels, cx_ratio, area_rows, px_width,
                 ang_width, clipped, bbox=None, aspect=1.0, v_clipped=False,
                 fill=1.0):
        self.bearing = bearing        # rad, 로봇 프레임
        self.pixels = pixels
        self.cx_ratio = cx_ratio      # 0(왼쪽끝)~1(오른쪽끝)
        self.area_rows = area_rows    # 마스크 세로 범위 (근접도 참고용)
        self.px_width = px_width      # 블롭 가로 픽셀 수
        self.ang_width = ang_width    # 블롭이 차지하는 시야각 (rad)
        self.clipped = clipped        # 블롭이 화면 좌/우 끝에 잘렸는가
        self.bbox = bbox              # (x0, y0, x1, y1) — YOLO 검증용
        self.aspect = aspect          # 가로/세로 비 (확정 단계 게이트용)
        self.v_clipped = v_clipped    # 상/하단 잘림 (근접 사과)
        self.fill = fill              # 채움비 (사과=민무늬 ~0.75+, 캔=구멍)


class TargetDetector:
    def __init__(self, cfg):
        self.cfg = cfg
        self.consecutive = 0
        self.miss = 0
        self.confirmed = False
        self.last = None              # 마지막 Detection
        self.rejection_reason = 'no_frame'
        self._yolo = None
        self._yolo_failed = False
        self.yolo_device = "disabled"
        self.yolo_error = None
        self.yolo_ms = None
        self._yolo_frame = None
        self._yolo_result = []

    def yolo_warmup(self):
        """모델 로드+더미 추론을 미션 시작 전에 — 확정 순간 수 초 블록 방지."""
        if not self.cfg.detection.use_yolo:
            return
        if self._yolo_failed:
            raise RuntimeError(self.yolo_error or "YOLO is unavailable")
        try:
            from ultralytics import YOLO
            import torch
            device = self.cfg.detection.yolo_device
            if device == "auto":
                device = "cuda:0" if torch.cuda.is_available() else "cpu"
            model = Path(self.cfg.detection.yolo_model)
            if not model.is_absolute():
                root = Path(__file__).resolve().parents[1]
                candidates = [root / model, root / "controllers/sar_controller" / model]
                model = next((p for p in candidates if p.is_file()), model)
            if not model.is_file():
                raise FileNotFoundError(f"YOLO model missing: {model.resolve()}")
            self._yolo = YOLO(str(model))
            self._yolo.predict(source=np.zeros((self.cfg.camera.height,
                                               self.cfg.camera.width, 3), dtype=np.uint8),
                               device=device, conf=self.cfg.detection.yolo_conf,
                               verbose=False)
            self.yolo_device = str(self._yolo.predictor.device)
            print(f"[detection] YOLO 워밍업 완료 (device={self.yolo_device})")
        except Exception as e:
            self._yolo_failed = True
            self.yolo_error = str(e)
            raise RuntimeError(f"YOLO startup failed: {e}") from e

    def _predict(self, img_rgb):
        """Cache only the identical immutable frame, never a later camera frame."""
        if self._yolo_failed:
            raise RuntimeError(self.yolo_error or "YOLO is unavailable")
        if self._yolo is None:
            self.yolo_warmup()
        if img_rgb is self._yolo_frame:
            return self._yolo_result
        try:
            started = time.perf_counter()
            res = self._yolo.predict(source=np.ascontiguousarray(img_rgb[..., ::-1]),
                                     conf=self.cfg.detection.yolo_conf, verbose=False)[0]
            # One GPU→CPU transfer per frame, not several transfers per box.
            rows = res.boxes.data.cpu().numpy()
            self._yolo_result = [(res.names[int(row[5])], float(row[4]),
                                  row[:4].tolist()) for row in rows]
            self.yolo_ms = (time.perf_counter() - started) * 1000
            self._yolo_frame = img_rgb
            return self._yolo_result
        except Exception as e:
            self._yolo_failed = True
            self.yolo_error = str(e)
            raise RuntimeError(f"YOLO inference failed: {e}") from e

    def yolo_boxes(self, img_rgb):
        """YOLO 전체 탐지 박스 [(name, conf, xyxy)], 현재 프레임 전용."""
        if not self.cfg.detection.use_yolo or img_rgb is None:
            return []
        return [box for box in self._predict(img_rgb) if box[1] >= .15]

    def yolo_confirm(self, img_rgb, det):
        """확정 직전 1회 실행되는 YOLO 검증 게이트.

        후보와 강하게 겹치는 비사과 객체를 거부한다 (미탐지는 통과).
        비활성일 때만 YOLO 검증을 생략한다. 활성 상태의 실패는 오류.
        """
        dcfg = self.cfg.detection
        if not dcfg.use_yolo or det is None or det.bbox is None \
                or img_rgb is None:
            return True
        boxes = self._predict(img_rgb)
        # 베토 모드. 실측 근거: yolo11n은 Webots 저폴리 사과를 'apple'로
        # 인식하지 못한다(0.9m에서 미탐지) — 양성 확인을 요구하면 진짜
        # 사과가 기각된다. 대신 블롭과 강하게 겹치는 '비(非)사과' 물체
        # (캔→vase/tv, 병 등)가 잡히면 디코이로 기각한다.
        x0, y0, x1, y1 = det.bbox
        blob_area = max(1, (x1 - x0) * (y1 - y0))
        for name, conf, (bx0, by0, bx1, by1) in boxes:
            iw = min(x1, bx1) - max(x0, bx0)
            ih = min(y1, by1) - max(y0, by0)
            if iw <= 0 or ih <= 0 or iw * ih <= 0.4 * blob_area:
                continue
            if name == "apple":
                return True                   # 명시적 사과 확인 → 통과
            if name in dcfg.yolo_ambiguous_classes:
                continue  # Webots apple was measured as sports ball; HSV/geometry still required.
            object_area = max(1, (bx1 - bx0) * (by1 - by0))
            if iw * ih < .15 * object_area:
                continue  # A large background couch/table is not this small red blob.
            if conf >= 0.30:
                print(f"[detection] YOLO 베토: '{name}' {conf:.2f}")
                return False                  # 강한 비사과 물체 → 기각
        return True

    @staticmethod
    def _dominant_blob(mask):
        """행·열 히스토그램의 최장 연속 활성 구간으로 주 블롭만 남긴다."""
        def longest_run(active):
            best_s = best_e = cur_s = -1
            best_len = cur_len = 0
            for i, a in enumerate(active):
                if a:
                    if cur_len == 0:
                        cur_s = i
                    cur_len += 1
                    if cur_len > best_len:
                        best_len, best_s, best_e = cur_len, cur_s, i
                else:
                    cur_len = 0
            return best_s, best_e

        col = mask.sum(axis=0)
        if col.max() < 2:
            return mask
        x0, x1 = longest_run(col >= max(2, 0.15 * col.max()))
        if x0 < 0:
            return mask
        sub = mask[:, x0:x1 + 1]
        row = sub.sum(axis=1)
        y0, y1 = longest_run(row >= max(2, 0.15 * row.max()))
        if y0 < 0:
            return mask
        out = np.zeros_like(mask)
        out[y0:y1 + 1, x0:x1 + 1] = mask[y0:y1 + 1, x0:x1 + 1]
        return out

    def _mask(self, hsv):
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        mask = np.zeros(h.shape, dtype=bool)
        for h_lo, h_hi, s_lo, v_lo in self.cfg.resolve_hsv():
            mask |= (h >= h_lo) & (h <= h_hi) & (s >= s_lo) & (v >= v_lo)
        return mask

    def _candidate_masks(self, mask):
        if cv2 is None:
            yield self._dominant_blob(mask)
            return
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=8)
        candidates = [i for i in range(1, count)
                      if stats[i, cv2.CC_STAT_AREA] >= self.cfg.detection.min_pixels]
        # Examine distinct objects separately: a rejected red sign must not hide
        # a smaller apple elsewhere in the frame. Bound cost on noisy images.
        candidates.sort(key=lambda i: stats[i, cv2.CC_STAT_AREA], reverse=True)
        for i in candidates[:12]:
            yield labels == i

    def process(self, img_rgb):
        """img_rgb: (H,W,3) uint8. 반환: Detection 또는 None (이번 프레임)."""
        det = None
        self.rejection_reason = 'no_frame'
        if img_rgb is not None:
            hsv = rgb_to_hsv(img_rgb)
            mask = self._mask(hsv)
            self.rejection_reason = 'too_few_color_pixels_or_fragmented'
            h, w = img_rgb.shape[:2]
            for mask in self._candidate_masks(mask):
                n = int(mask.sum())
                if n >= self.cfg.detection.min_pixels \
                        and n <= self.cfg.detection.max_area_ratio * h * w:
                    ys, xs = mask.nonzero()
                    cx = float(xs.mean())
                    cx_ratio = cx / w
                    # 핀홀 투영: x_px = f*tan(bearing). 선형 근사는 단안 거리를
                    # 중심시야 +10%/가장자리 -17% 편향시킨다 (리뷰 지적 수정)
                    import math as _m
                    f_px = (w / 2) / _m.tan(self.cfg.camera.hfov / 2)
                    bearing = _m.atan2(w / 2 - cx, f_px) \
                        + self.cfg.camera.mount_yaw
                    ang_left = _m.atan2(w / 2 - float(xs.min()), f_px)
                    ang_right = _m.atan2(w / 2 - float(xs.max() + 1), f_px)
                    ang_width = ang_left - ang_right
                    px_width = int(xs.max() - xs.min() + 1)
                    clipped = bool(xs.min() == 0 or xs.max() == w - 1)
                    # 디코이 방어 (세로 기하):
                    # ① 카메라가 사과 높이에 있으므로 바닥의 사과는 화면
                    #    '상단'에 닿을 수 없다 — 상단 접촉 = 키 큰 물체
                    #    (소화기 등) → 기각. (하단 접촉은 근접 사과에서 정상)
                    # ② 세로로 안 잘린 블롭의 종횡비가 극단이면 기각
                    px_h = int(ys.max() - ys.min() + 1)
                    top_clipped = bool(ys.min() == 0)
                    v_clipped = top_clipped or bool(ys.max() == h - 1)
                    aspect = px_width / max(1, px_h)
                    # ③ 바닥 사과의 블롭 중심은 항상 수평선(화면 중앙 행)
                    #    근처다 (카메라가 사과 높이에 장착). 수평선보다 훨씬
                    #    위에 뜬 블롭 = 공중의 물체(표지판·벽걸이 등) → 기각.
                    #    아래쪽은 근접 사과에서 정상이므로 관대하게.
                    cy_blob = float(ys.mean())
                    above_horizon = cy_blob < h * self.cfg.detection.min_center_y_ratio
                    # ④ 채움비: 사과(구)는 bbox의 ~78%를 채운다. 표면에 글자·
                    #    무늬가 있는 물체(음료캔 등)는 마스크에 구멍이 남
                    fill = n / max(1, px_width * px_h)
                    # 탐지 단계는 느슨하게 — 가려진 사과(반달형)도 추적해야
                    # SEEK로 접근해 가림을 풀 수 있다. 엄격한 종횡비는
                    # '위치 확정' 단계(state_machine)에서 적용.
                    if top_clipped or above_horizon:
                        self.rejection_reason = 'top_clipped_or_above_horizon'
                        det = None
                    elif not v_clipped and not (0.35 <= aspect <= 2.4):
                        self.rejection_reason = 'aspect_ratio'
                        det = None
                    elif not v_clipped and not clipped and fill < 0.45:
                        self.rejection_reason = 'low_color_fill'
                        det = None
                    else:
                        det = Detection(bearing, n, cx_ratio,
                                        (int(ys.min()), int(ys.max())),
                                        px_width, ang_width, clipped,
                                        bbox=(int(xs.min()), int(ys.min()),
                                              int(xs.max()), int(ys.max())),
                                        aspect=aspect, v_clipped=v_clipped,
                                        fill=fill)
                if det is not None:
                    break

        if det is not None:
            self.rejection_reason = None
            self.consecutive += 1
            self.miss = 0
            self.last = det
            if self.consecutive >= self.cfg.detection.confirm_frames:
                self.confirmed = True
        else:
            self.consecutive = 0
            self.miss += 1
            if self.miss >= self.cfg.detection.lost_frames:
                self.confirmed = False
                self.last = None
        return det

    @property
    def visible(self):
        return self.consecutive > 0


def estimate_target_position(pose, bearing, angles, ranges, window=0.12):
    """탐지 bearing 방향의 LiDAR 거리로 목표 월드 좌표 추정.

    window(rad) 내 빔들의 최소값 근처 중앙값 사용 (배경 빔 섞임 방지).
    실패 시 None.

    주의: 낮은 물체(사과 등)는 LiDAR 스캔 평면 아래라 아예 안 잡히고
    이 함수는 '배경 벽' 거리를 반환하게 된다 — 반드시 blob 크기와의
    일관성 검사를 거친 뒤 사용할 것 (state_machine 참고).
    """
    import math

    angles = np.asarray(angles)
    ranges = np.asarray(ranges)
    sel = np.abs(((angles - bearing + math.pi) % (2 * math.pi)) - math.pi) < window
    r = ranges[sel]
    r = r[np.isfinite(r) & (r > 0.05)]
    if r.size == 0:
        return None
    rmin = r.min()
    near = r[r < rmin + 0.3]
    d = float(np.median(near))
    ang = pose[2] + bearing
    return (pose[0] + d * math.cos(ang), pose[1] + d * math.sin(ang), d)


def estimate_distance_mono(det, target_width):
    """단안 거리 추정: 알려진 목표 폭 + blob이 차지한 시야각 → 거리.

    LiDAR 평면보다 낮은 목표(사과)의 주 거리 추정 수단.
    d = (W/2) / tan(ang_width/2)
    """
    import math

    if det is None or det.ang_width <= 1e-6:
        return None
    return (target_width / 2) / math.tan(det.ang_width / 2)


def estimate_target_position_mono(pose, det, target_width,
                                  d_min=0.25, d_max=7.0):
    """단안 거리 기반 목표 월드 좌표. 실패/비상식 거리면 None.

    blob이 화면 끝에 잘린 프레임은 폭이 줄어 거리가 과대추정되므로 기각.
    """
    import math

    if det is not None and det.clipped:
        return None
    d = estimate_distance_mono(det, target_width)
    if d is None or not (d_min <= d <= d_max):
        return None
    ang = pose[2] + det.bearing
    return (pose[0] + d * math.cos(ang), pose[1] + d * math.sin(ang), d)
