"""YOLO-first 목표 탐지 + HSV 색상 검증.

- YOLO11n COCO apple(class 47) 박스를 먼저 탐지
- 각 apple 박스 안에서만 HSV 빨간색 블롭을 검사해 red apple 확정
- 최소 픽셀 수 + N프레임 연속 확인으로 오탐 억제
- use_yolo=False인 오프라인/단위 테스트에서는 기존 HSV 단독 모드 지원
- 출력 bearing: 로봇 프레임, +값 = 왼쪽(CCW). 화면 왼쪽 = +bearing
"""
from pathlib import Path

import numpy as np


def rgb_to_hsv(img):
    """img: (H,W,3) uint8 RGB → (H,W,3) float32, H∈[0,360), S,V∈[0,1]."""
    f = img.astype(np.float32) / 255.0
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    mx = f.max(axis=-1)
    mn = f.min(axis=-1)
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
        self._yolo = None
        self._yolo_failed = False
        self._last_yolo_boxes = []
        self._last_table_boxes = []

    def _yolo_classes(self):
        classes = [self.cfg.detection.yolo_class_id]
        if self.cfg.table.enabled:
            classes.append(self.cfg.table.yolo_class_id)
        return classes

    def _model_source(self):
        """상대 모델 경로는 저장소 루트 기준으로 찾고, 없으면 자동 다운로드."""
        configured = Path(self.cfg.detection.yolo_model)
        if configured.is_absolute():
            return str(configured)
        repo_path = Path(__file__).resolve().parents[1] / configured
        if repo_path.exists():
            return str(repo_path)
        # Ultralytics 공식 가중치 이름이면 basename만 넘겨 자동 다운로드.
        return configured.name

    def _ensure_yolo(self):
        if self._yolo is not None:
            return True
        if self._yolo_failed or not self.cfg.detection.use_yolo:
            return False
        try:
            from ultralytics import YOLO
            self._yolo = YOLO(self._model_source())
            return True
        except Exception as e:
            print(f"[detection] YOLO 사용 불가({e}) — apple 탐지 비활성")
            self._yolo_failed = True
            return False

    def yolo_warmup(self):
        """모델 로드+더미 추론을 미션 시작 전에 수행."""
        if not self.cfg.detection.use_yolo or self._yolo_failed:
            return
        if not self._ensure_yolo():
            return
        try:
            self._yolo.predict(source=np.zeros((64, 64, 3), dtype=np.uint8),
                               conf=self.cfg.detection.yolo_conf,
                               iou=self.cfg.detection.yolo_iou,
                               classes=self._yolo_classes(),
                               verbose=False)
            print("[detection] YOLO 워밍업 완료 — COCO apple(47) 우선 탐지")
        except Exception as e:
            print(f"[detection] YOLO 워밍업 실패({e}) — apple 탐지 비활성")
            self._yolo_failed = True
            self._yolo = None

    def _predict_apples(self, img_rgb):
        """한 번의 YOLO 추론으로 apple과 dining table을 분리한다."""
        self._last_yolo_boxes = []
        self._last_table_boxes = []
        if img_rgb is None or not self._ensure_yolo():
            return []
        dcfg = self.cfg.detection
        try:
            result = self._yolo.predict(
                source=img_rgb[..., ::-1],       # RGB -> BGR
                conf=dcfg.yolo_conf,
                iou=dcfg.yolo_iou,
                classes=self._yolo_classes(),
                verbose=False,
            )[0]
        except Exception as e:
            print(f"[detection] YOLO 추론 실패({e}) — 이번 프레임 무시")
            return []

        apple_boxes = []
        table_boxes = []
        for box in result.boxes:
            # Ultralytics는 길이 1인 torch.Tensor를 돌려준다. numpy 기반
            # 테스트 더블과도 호환되도록 첫 원소를 명시적으로 꺼낸다.
            cls_id = int(box.cls[0])
            name = result.names[cls_id]
            name_lower = name.lower()
            is_apple = cls_id == dcfg.yolo_class_id or name_lower == "apple"
            is_table = self.cfg.table.enabled and (
                cls_id == self.cfg.table.yolo_class_id
                or name_lower in ("dining table", "table"))
            # classes 필터가 무시되는 커스텀 모델에도 안전하게 재확인.
            if not is_apple and not is_table:
                continue
            conf = float(box.conf[0])
            xyxy = [float(v) for v in box.xyxy[0]]
            item = (name, conf, xyxy)
            if is_apple:
                apple_boxes.append(item)
            elif conf >= self.cfg.table.yolo_conf:
                table_boxes.append(item)
        self._last_table_boxes = table_boxes
        self._last_yolo_boxes = apple_boxes + table_boxes
        return apple_boxes

    def yolo_boxes(self, img_rgb):
        """라이브 뷰용: process()에서 계산한 apple/table 박스."""
        return list(self._last_yolo_boxes)

    def table_boxes(self):
        """현재 프레임의 확신도 기준을 통과한 dining table 박스."""
        return list(self._last_table_boxes)

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

    def _detection_from_mask(self, img_rgb, mask):
        """빨간색 마스크 하나를 기하 검증하고 Detection으로 변환."""
        h, w = img_rgb.shape[:2]
        # 최대 밀집 블롭만 추출 — 화면 곳곳의 반사 노이즈 몇 픽셀이
        # bbox를 부풀려 채움비·중심 통계를 무너뜨리는 것 방지
        mask = self._dominant_blob(mask)
        n = int(mask.sum())
        if n < self.cfg.detection.min_pixels \
                or n > self.cfg.detection.max_area_ratio * h * w:
            return None

        ys, xs = mask.nonzero()
        cx = float(xs.mean())
        cx_ratio = cx / w
        # 핀홀 투영: x_px = f*tan(bearing).
        import math as _m
        f_px = (w / 2) / _m.tan(self.cfg.camera.hfov / 2)
        bearing = _m.atan2(w / 2 - cx, f_px) + self.cfg.camera.mount_yaw
        ang_left = _m.atan2(w / 2 - float(xs.min()), f_px)
        ang_right = _m.atan2(w / 2 - float(xs.max() + 1), f_px)
        ang_width = ang_left - ang_right
        px_width = int(xs.max() - xs.min() + 1)
        clipped = bool(xs.min() == 0 or xs.max() == w - 1)

        # 바닥 사과에 맞춘 세로 기하/형태 검증.
        px_h = int(ys.max() - ys.min() + 1)
        top_clipped = bool(ys.min() == 0)
        v_clipped = top_clipped or bool(ys.max() == h - 1)
        aspect = px_width / max(1, px_h)
        cy_blob = float(ys.mean())
        above_horizon = cy_blob < h * 0.35
        fill = n / max(1, px_width * px_h)
        if top_clipped or above_horizon:
            return None
        if not v_clipped and not (0.35 <= aspect <= 2.4):
            return None
        if not v_clipped and not clipped and fill < 0.45:
            return None
        return Detection(
            bearing, n, cx_ratio, (int(ys.min()), int(ys.max())),
            px_width, ang_width, clipped,
            bbox=(int(xs.min()), int(ys.min()),
                  int(xs.max()), int(ys.max())),
            aspect=aspect, v_clipped=v_clipped, fill=fill,
        )

    def process(self, img_rgb):
        """YOLO apple -> HSV red 순서로 한 프레임을 검사."""
        det = None
        if img_rgb is not None:
            if self.cfg.detection.use_yolo:
                # 순서가 중요하다: apple 박스가 하나도 없으면 HSV 계산조차
                # 하지 않는다. 즉 빨간색 자체는 탐색 후보를 만들 수 없다.
                apple_boxes = self._predict_apples(img_rgb)
                red_mask = None
                if apple_boxes:
                    red_mask = self._mask(rgb_to_hsv(img_rgb))
                h, w = img_rgb.shape[:2]
                best = None
                best_score = -1.0
                for _name, conf, (bx0, by0, bx1, by1) \
                        in apple_boxes:
                    x0 = max(0, min(w, int(np.floor(bx0))))
                    y0 = max(0, min(h, int(np.floor(by0))))
                    x1 = max(0, min(w, int(np.ceil(bx1))))
                    y1 = max(0, min(h, int(np.ceil(by1))))
                    if x1 <= x0 or y1 <= y0:
                        continue
                    roi = red_mask[y0:y1, x0:x1]
                    red_pixels = int(roi.sum())
                    red_ratio = red_pixels / max(1, roi.size)
                    if red_pixels < self.cfg.detection.min_pixels \
                            or red_ratio < self.cfg.detection.yolo_red_min_ratio:
                        continue
                    candidate_mask = np.zeros_like(red_mask)
                    candidate_mask[y0:y1, x0:x1] = roi
                    candidate = self._detection_from_mask(
                        img_rgb, candidate_mask)
                    if candidate is None:
                        continue
                    score = conf * candidate.pixels
                    if score > best_score:
                        best, best_score = candidate, score
                det = best
            else:
                # 단위/오프라인 테스트 및 긴급 폴백용 HSV-only 모드.
                self._last_yolo_boxes = []
                self._last_table_boxes = []
                red_mask = self._mask(rgb_to_hsv(img_rgb))
                det = self._detection_from_mask(img_rgb, red_mask)

        if det is not None:
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
