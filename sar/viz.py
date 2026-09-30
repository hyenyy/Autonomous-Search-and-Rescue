"""지도/경로/상태 시각화 (matplotlib, 없으면 조용히 비활성).

- save(path): 현재 상황 PNG 저장 (headless 안전, Agg 백엔드)
- 오프라인 sim과 Webots 컨트롤러 양쪽에서 사용
"""
import math
import io

import numpy as np
from PIL import Image, ImageDraw
from .runtime import atomic_write


def save_rgb(path, rgb):
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(rgb)).save(buf, format='PNG', compress_level=1)
    return atomic_write(path, buf.getvalue())

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.image as mpimage
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:  # matplotlib 미설치 환경
    HAVE_MPL = False


class MapViz:
    def __init__(self, grid):
        self.grid = grid
        self.enabled = HAVE_MPL

    def save_fast(self, path, pose=None, info=None):
        """경량 실시간 렌더(~10ms) — matplotlib figure는 스냅샷마다
        제어 루프를 수백 ms 블록해 주기적 스터터를 만든다. 픽셀 합성으로 대체."""
        g = self.grid
        img = np.full((g.n, g.n, 3), 165, dtype=np.uint8)   # 미지 = 회색
        img[g.free_mask()] = (245, 245, 245)
        img[g.occupied_mask()] = (25, 25, 25)

        def dot(x, y, color, r=2):
            ix, iy = g.world_to_grid(x, y)
            if 0 <= ix < g.n and 0 <= iy < g.n:
                img[max(0, iy - r):iy + r + 1,
                    max(0, ix - r):ix + r + 1] = color

        if info:
            if info.get("start") is not None:
                dot(*info["start"], (0, 210, 230), 4)
            for visited in info.get("visited_targets") or []:
                dot(*visited, (220, 40, 40), 4)
            for c in info.get("crumbs") or []:
                dot(c[0], c[1], (255, 165, 0), 1)
            for p in info.get("waypoints") or []:
                dot(p[0], p[1], (80, 130, 255), 1)
            for c in info.get("candidates") or []:
                dot(c[0], c[1], (200, 0, 220), 2)   # 보라: 목표 후보
            for f in info.get("furniture") or []:
                dot(f[0], f[1], (150, 95, 40), 2)   # 갈색: 가구 회피 구역
            if info.get("goal"):
                dot(info["goal"][0], info["goal"][1], (0, 80, 255), 3)
            if info.get("target_est"):
                dot(info["target_est"][0], info["target_est"][1],
                    (255, 210, 0), 4)  # yellow = unvisited estimated position
        if pose is not None:
            dot(pose[0], pose[1], (0, 190, 0), 3)
            tip = (pose[0] + 0.28 * math.cos(pose[2]),
                   pose[1] + 0.28 * math.sin(pose[2]))
            dot(tip[0], tip[1], (0, 120, 0), 1)
        save_rgb(path, img[::-1])   # origin='lower' 뒤집기

    @staticmethod
    def save_camera(path, img_rgb, det=None, yolo_boxes=()):
        """카메라 라이브 뷰: HSV 블롭(노랑)·YOLO 박스(초록) 테두리 합성."""
        if img_rgb is None:
            return
        out = img_rgb.copy()
        hh, ww = out.shape[:2]

        def rect(x0, y0, x1, y1, color):
            x0 = max(0, min(ww - 1, int(x0)))
            x1 = max(0, min(ww - 1, int(x1)))
            y0 = max(0, min(hh - 1, int(y0)))
            y1 = max(0, min(hh - 1, int(y1)))
            out[y0:y0 + 2, x0:x1 + 1] = color
            out[max(0, y1 - 1):y1 + 1, x0:x1 + 1] = color
            out[y0:y1 + 1, x0:x0 + 2] = color
            out[y0:y1 + 1, max(0, x1 - 1):x1 + 1] = color

        for (_name, _conf, (a, b, c, d)) in yolo_boxes:
            rect(a, b, c, d, (0, 255, 0))
        if det is not None and det.bbox is not None:
            rect(*det.bbox, (255, 255, 0))
        canvas = Image.fromarray(out)
        draw = ImageDraw.Draw(canvas)
        for name, confidence, (x0, y0, _, _) in yolo_boxes:
            text = f'{name} {confidence:.2f}'
            xy = (max(0, int(x0)), max(0, int(y0) - 13))
            draw.rectangle(draw.textbbox(xy, text), fill=(0, 0, 0))
            draw.text(xy, text, fill=(0, 255, 0))
        buf = io.BytesIO()
        canvas.save(buf, format='PNG', compress_level=1)
        atomic_write(path, buf.getvalue())

    def save(self, path, pose=None, info=None, true_pose=None, world_extras=None):
        if not self.enabled:
            return
        g = self.grid
        fig, ax = plt.subplots(figsize=(7, 7), dpi=90)
        # 지도: unknown 회색, free 흰색, occupied 검정
        img = 0.5 - 0.5 * g.free_mask() + 0.5 * g.occupied_mask()
        extent = (g.origin_x, g.origin_x + g.n * g.res,
                  g.origin_y, g.origin_y + g.n * g.res)
        ax.imshow(1 - img, cmap="gray", origin="lower", extent=extent,
                  vmin=0, vmax=1)
        if info:
            wps = info.get("waypoints") or []
            if wps:
                ax.plot([p[0] for p in wps], [p[1] for p in wps],
                        "-", color="tab:blue", lw=1.5, label="path")
            crumbs = info.get("crumbs") or []
            if crumbs:
                ax.plot([p[0] for p in crumbs], [p[1] for p in crumbs],
                        ".", color="tab:orange", ms=2, label="crumbs")
            if info.get("goal"):
                ax.plot(*info["goal"], "x", color="tab:blue", ms=10)
            if info.get("target_est"):
                ax.plot(*info["target_est"], "*", color="red", ms=14,
                        label="target est")
        if pose is not None:
            ax.plot(pose[0], pose[1], "o", color="tab:green", ms=8)
            ax.arrow(pose[0], pose[1],
                     0.25 * math.cos(pose[2]), 0.25 * math.sin(pose[2]),
                     head_width=0.08, color="tab:green")
        if true_pose is not None:
            ax.plot(true_pose[0], true_pose[1], "+", color="purple", ms=10,
                    label="true pose")
        if world_extras:
            for (x, y, marker, color, label) in world_extras:
                ax.plot(x, y, marker, color=color, ms=10, label=label)
        state = info.get("state") if info else ""
        ax.set_title(f"{state}")
        ax.legend(loc="upper right", fontsize=7)
        ax.set_aspect("equal")
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
