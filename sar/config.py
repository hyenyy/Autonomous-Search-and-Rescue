"""All tunables in one place.

당일 적응 절차: 실제 로봇/월드 사양이 나오면 이 파일의 값만 고친다.
코드 로직은 건드리지 않는 것이 원칙. 어떤 값을 어떻게 확인하는지는
docs/DAY_OF_CHECKLIST.md 참고.
"""
from dataclasses import dataclass, field
import json
import math


@dataclass
class RobotConfig:
    # TurtleBot3 Burger defaults — 당일 실제 로봇 사양으로 교체
    wheel_radius: float = 0.033          # m
    wheel_base: float = 0.160            # m, 좌우 바퀴 간격
    max_wheel_speed: float = 6.6         # rad/s (TB3 한계 6.67 직전)
    robot_radius: float = 0.11           # m, 충돌 판정/지도 팽창용 (여유 포함)
    max_v: float = 0.21                  # m/s (TB3 burger 실제 최고 0.22)
    max_w: float = 1.6                   # rad/s


@dataclass
class LidarConfig:
    # 주최측 예제(tb3_teleop_sensors)로 확인된 LDS-01 매핑:
    #   ranges[0]=후방, ranges[90]=좌, ranges[180]=전방, ranges[270]=우
    # → index0 = pi(후방), index 증가 = 시계방향(ccw=False)
    index0_angle: float = math.pi        # rad
    ccw: bool = False                    # index 증가 방향이 CCW면 True
    max_range: float = 3.4               # m, 이 이상은 미측정 취급
    min_range: float = 0.12              # m, 이 이하는 노이즈 취급
    subsample: int = 2                   # 매핑 시 빔 서브샘플링 간격


@dataclass
class CameraConfig:
    # 주최측 월드 기준: 640x480, FOV 1.0472. robot_io가 실제 디바이스
    # 값으로 자동 갱신하므로 여기 값은 폴백.
    hfov: float = 1.0472                 # rad, 수평 화각
    width: int = 640
    height: int = 480
    # 카메라가 로봇 정면(x축) 기준 틀어진 각도 (보통 0)
    mount_yaw: float = 0.0


# 주최측 사과 프로토 baseColor 기준 HSV 프리셋 (h_lo, h_hi, s_lo, v_lo)
# s_lo는 apartment 실측으로 캘리브레이션: 순색 사과는 S 0.65+(p5)로
# 렌더링되고, 디코이(라디에이터 노브 등 재질색)는 S≤0.54(p95) —
# 0.60이 깨끗한 분리선. green 사과는 텍스처라 넓게 — 당일 보정 필수.
COLOR_PRESETS = {
    "red": ((345.0, 360.0, 0.60, 0.25), (0.0, 18.0, 0.60, 0.25)),
    "orange": ((25.0, 55.0, 0.55, 0.25),),
    "purple": ((250.0, 295.0, 0.45, 0.15),),
    "green": ((60.0, 150.0, 0.28, 0.08),),
}


@dataclass
class DetectionConfig:
    target_color: str = "red"            # 당일 지정되는 목표 색
    # HSV 범위 목록: (h_lo, h_hi, s_lo, v_lo), H는 0~360.
    # target_color 프리셋이 있으면 그것이 우선 (config.resolve_hsv 참고)
    hsv_ranges: tuple = ()
    min_pixels: int = 60                 # 이 미만이면 무시
    # 프레임 점유율 상한: 사과는 최근접 유효거리(0.3m)에서도 화면의
    # ~12%가 한계 — 이보다 큰 블롭은 빨간 가구/러그 등 디코이
    max_area_ratio: float = 0.2
    confirm_frames: int = 4              # 연속 N프레임 탐지 시 확정 (오탐 방지)
    lost_frames: int = 15                # 연속 미탐지 시 추적 해제
    target_width: float = 0.10           # m, 목표 폭 (사과 지름 ≈ 0.1m)
    # blob 크기와 LiDAR 거리의 일관성 허용 비율 (오추정 거부용)
    size_ratio_lo: float = 0.65
    size_ratio_hi: float = 1.6
    # 위치 확정 안정성 게이트: 최근 K개 추정이 서로 이 반경 안이어야 확정
    est_stable_count: int = 3
    est_stable_radius: float = 0.4
    # 이 거리 안에서의 추정만 '확정'에 사용 (원거리 단안은 픽셀 양자화로
    # ±25% 오차 — 멀면 SEEK로 접근해서 다시 잰다)
    est_confirm_dist: float = 2.2
    # YOLO-first 탐지: COCO apple(47) 박스를 먼저 찾고, 그 박스 안에서
    # 위 HSV 빨간색 범위를 만족하는 픽셀 블롭이 있어야 최종 후보가 된다.
    # 강사 예제와 동일한 YOLO11n/COCO ID/conf/IoU 기본값.
    use_yolo: bool = False
    yolo_model: str = "models/YOLO/yolo11n.pt"
    yolo_class_id: int = 47              # COCO: apple
    yolo_conf: float = 0.10
    yolo_iou: float = 0.50
    # YOLO 박스 중 최소 이 비율이 빨간색이어야 red apple로 인정.
    yolo_red_min_ratio: float = 0.05


@dataclass
class TableConfig:
    """RGB 의미 인식 + 2D LiDAR 다리 결합 기반 테이블 회피."""
    enabled: bool = True
    yolo_class_id: int = 60              # COCO: dining table
    yolo_conf: float = 0.35
    # 너무 작게 보이는 먼 테이블은 다리-박스 대응이 불안정하므로 무시.
    min_box_width_ratio: float = 0.08
    max_lidar_range: float = 2.8
    # 가장 가까운 다리보다 이 거리 안쪽인 LiDAR 점만 같은 테이블 후보.
    leg_depth_band: float = 0.45
    min_leg_gap: float = 0.22
    max_leg_gap: float = 2.8
    barrier_thickness: float = 0.10
    confirm_frames: int = 3
    track_resolution: float = 0.40
    track_timeout: float = 1.0
    # RECOVER 후진 중 이 거리보다 가까운 물체가 뒤에 있으면 즉시 중단.
    recover_rear_stop: float = 0.30


@dataclass
class MapConfig:
    resolution: float = 0.05             # m/cell
    # 그리드는 '시작점' 중심이므로, 시작점이 월드 구석이면 월드 전체
    # 지름만큼 필요하다. apartment(~8x13m) 구석 시작까지 커버되게 크게.
    half_size: float = 16.0              # m, 시작점 기준 ±half_size 커버
    update_period: float = 0.15          # s, 지도 갱신 주기 (Mission 내부)
    l_occ: float = 0.9                   # 점유 관측 log-odds 증가량
    l_free: float = 0.4                  # 빈공간 관측 log-odds 감소량
    l_clamp: float = 5.0
    occ_threshold: float = 1.2           # log-odds > 이 값 → 장애물
    free_threshold: float = -0.5         # log-odds < 이 값 → 빈공간


@dataclass
class PlanConfig:
    # 팽창 여유: 0.10은 좁은 가구 틈 진입 불가 문제 — 코너 클리핑은
    # A* 대각 금지+정밀 LOS로 따로 해결됐으므로 0.07이면 안전
    inflate_margin: float = 0.07         # m, robot_radius 에 더하는 여유
    unknown_cost: float = 2.5            # A*에서 미지 셀 통과 비용 배율
    goal_tolerance: float = 0.15         # m
    replan_period: float = 2.5           # s (사람 보이면 0.8s로 자동 단축)
    lookahead: float = 0.35              # m, pure pursuit 전방 주시 거리
    k_heading: float = 2.2               # 헤딩 오차 → 각속도 게인
    # 반응형 회피 — 전방 판정은 '각도 부채꼴'이 아니라 '충돌 코리도'
    # (로봇 진행 폭 안의 물체만). 부채꼴 방식은 문설주를 정면 장애물로
    # 오인해 문 통과마다 감속·정지를 유발한다.
    corridor_margin: float = 0.05        # m, 코리도 반폭 = robot_radius + 이 값
    front_halfwidth: float = 0.75        # rad, (구) 섹터 반각 — 좌우 여유 계산용
    slow_dist: float = 0.45              # m, 이하부터 감속
    stop_dist: float = 0.20              # m, 이하 정지 (코리도 판정이라
                                         # 경로상 장애물에만 반응 — 안전)
    danger_dist: float = 0.20            # m, 전방향 비상 탈출 발동 거리
    person_wait: float = 4.0             # s, 동적 장애물 양보 대기 후 재계획
    static_wait: float = 1.2             # s, '정적' 막힘의 재계획 대기
                                         # (우회 판단 지연의 주범이던 4s 분리)
    # 동적 장애물(사람) 전용 — 사람이 로봇보다 빠르다는 전제의 조기 대응
    dyn_escape_dist: float = 0.50        # m, 이내면 적극 이탈
    dyn_yield_dist: float = 0.95         # m, 접근 중이면 정지 양보
    # 동적 분류의 free 요구 수준: 벽은 빔이 통과 못 해 free 증거가 안 쌓이니
    # -1.2면 '갓 그려지는 벽'(log≈0~-0.5)을 확실히 배제하면서, 확실히
    # 비었던 공간을 걷는 사람은 잡는다. (-0.5는 신축 벽을 동적으로 오분류
    # → 벽으로 탈출 기동하는 회귀를 일으켰음)
    dyn_free_logodds: float = -1.2
    dyn_wall_logodds: float = 1.2        # 이웃이 이보다 높으면(=점유 확정)
                                         # '벽 근처'로 보고 동적 분류 제외.
                                         # 사람 잔상 오염은 지도 제외로 이미
                                         # 방지되므로 낮게 잡아도 안전.
    dyn_track_radius: float = 0.45       # m, 직전 사람 위치 주변은 셀 상태와
                                         # 무관하게 동적으로 간주 (연속 추적)
    # 사람이 지나가고 멀어지기 시작하면 이 시간 동안 양보를 끄고 전속 횡단
    dyn_commit_time: float = 4.0
    dyn_commit_escape: float = 0.35      # 커밋 중 탈출 발동 거리 (완화)


@dataclass
class ExploreConfig:
    min_cluster: int = 6                 # frontier 클러스터 최소 셀 수
    reach_tolerance: float = 0.35        # m, frontier 도달 판정
    spin_period: float = 16.0            # s, 주기적 카메라 스캔 간격 (0=끄기)
    # 주기 스캔은 전방 ±spin_arc/2 스윕만 (LiDAR는 360도라 회전 불필요 —
    # 회전은 60도 카메라용). 스윕은 frontier 도착 시에도 1회 수행.
    # 360도 풀스핀은 최초 1회 + frontier 소진 시(복귀 전 최종 확인)만.
    spin_arc: float = 4.19               # rad (±120도)
    blacklist_radius: float = 0.4        # m, 실패한 frontier 주변 제외 반경


@dataclass
class EnergyConfig:
    """주행 거리 기반 배터리 추정과 안전 귀환 설정.

    Webots 기본 로봇에 실제 배터리 센서가 없어도 시연 가능한 보수적
    에너지 모델이다. 실제 센서가 제공되면 initial_percent 대신 센서 값을
    EnergyManager에 주입하도록 교체하면 된다.
    """
    enabled: bool = True
    initial_percent: float = 100.0
    percent_per_meter: float = 1.0
    percent_per_radian: float = 0.03
    reserve_percent: float = 15.0
    # breadcrumb 귀환 거리는 실제 우회·회피를 고려해 이 배수만큼 여유를 둔다.
    return_multiplier: float = 1.25


@dataclass
class MissionConfig:
    start_x: float = 0.0                 # 시작 포즈 (대회에서 제공되는 값)
    start_y: float = 0.0
    start_theta: float = 0.0
    num_targets: int = 1                 # 찾아야 할 목표 개수 (대회: 사과 2개)
    visited_radius: float = 0.9          # m, 방문한 목표 주변 재확정 금지 반경
    return_tolerance: float = 0.25       # m, 복귀 성공 판정
    target_standoff: float = 0.40        # m, 목표 앞 정지 거리
    target_reach_tol: float = 0.25       # m
    stuck_time: float = 5.0              # s, 이 시간 동안
    stuck_dist: float = 0.08             # m, 이만큼도 못 가면 → recovery
    crumb_spacing: float = 0.25          # m, breadcrumb 기록 간격
    use_astar_return: bool = True        # False면 무조건 breadcrumb 복귀
    give_up_time: float = 0.0            # s, 0=무제한. 초과 시 목표 포기하고 복귀


@dataclass
class Config:
    robot: RobotConfig = field(default_factory=RobotConfig)
    lidar: LidarConfig = field(default_factory=LidarConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    table: TableConfig = field(default_factory=TableConfig)
    map: MapConfig = field(default_factory=MapConfig)
    plan: PlanConfig = field(default_factory=PlanConfig)
    explore: ExploreConfig = field(default_factory=ExploreConfig)
    energy: EnergyConfig = field(default_factory=EnergyConfig)
    mission: MissionConfig = field(default_factory=MissionConfig)

    def apply_overrides(self, path):
        """JSON 파일로 부분 오버라이드: {"robot": {"max_v": 0.2}, ...}"""
        with open(path) as f:
            data = json.load(f)
        for section, values in data.items():
            obj = getattr(self, section)
            for k, v in values.items():
                if not hasattr(obj, k):
                    raise KeyError(f"unknown config key: {section}.{k}")
                setattr(obj, k, v)
        return self

    def resolve_hsv(self):
        """탐지에 실제로 쓸 HSV 범위: 명시된 hsv_ranges > 색 프리셋."""
        if self.detection.hsv_ranges:
            return self.detection.hsv_ranges
        return COLOR_PRESETS[self.detection.target_color]


def default_config() -> Config:
    return Config()
