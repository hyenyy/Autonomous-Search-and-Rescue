# 대회 당일 체크리스트

목표: **코드 로직은 건드리지 않고** `sar/config.py`(또는 `config_override.json`)와
`robot_io.py`만 손봐서 실환경에 이식한다.

## 계획안 준수 매트릭스 (평가 기준에서 벗어나지 않기)

계획안(부산대_TECH_WEEK_해커톤_계획안.pdf)의 조건 ↔ 구현 대응:

| 계획안 조건/기술 | 구현 | 당일 확인 |
|---|---|---|
| 사전 지도 없음 | LiDAR+Odometry로 Occupancy Grid 실시간 생성 | — |
| 현재 위치 알 수 없음 | 시작 포즈에서 자체 추정만 사용 (절대 위치 센서/GPS/Supervisor 미사용) | Supervisor 노드 사용 금지 재확인 |
| 시작 지점(포즈)은 제공 | `mission.start_*`로 주입 | 제공값 입력 |
| 목표 위치 미지 + 시각 특징 제공 | frontier 탐색 + 색 탐지(단안 거리) | 목표 색·크기 입력 |
| 정적/동적 충돌 금지 | 팽창 A* + 정적 클램프 + 동적 양보/탈출 | E2E 1회 확인 |
| Mapping: NumPy Occupancy Grid | `sar/mapping.py` 그대로 | — |
| Localization: Scan Matching | **주의**: 현재 주력은 compass 헤딩 융합. compass 사용이 금지되면 `PoseEstimator`가 자동으로 순수 odometry 폴백 — 규칙 확인 필수. scan matching 삽입 지점은 `PoseEstimator` 한 곳 | compass 허용 여부 질문 |
| Detection: 규칙 기반 CV | HSV blob (`sar/detection.py`) | HSV 보정 |
| Global: Dijkstra/A* | A* (`sar/planning.py`) | — |
| Local: DWA 등 Critic 기반 | 경량 critic 절충(여유·진행·안전) 반응형 planner — 발표 시 이렇게 설명 | — |
| Python 3.10 기준 | 코드 3.10 호환 문법만 사용 (개발은 3.12) | 현장 파이썬 버전 확인 |

**미션 판정 흐름(계획안)**: 탐색 → 대상 발견 → 대상(목적지)까지 이동 → 시작 지점 복귀.
스윕 게이트(`sim_offline.is_success`)가 정확히 이 4단계 + 무충돌을 검사한다.

## 0. 시작 직후 (5분)

- [ ] 대회 월드/규칙 파일 받기. **채점 기준 확인**: 충돌 = 실격인가 감점인가?
      복귀 판정 반경? 목표 "발견" 판정 방식? 제한 시간?
- [ ] 목표 사과 **색** 확인 → `detection.target_color` = red/orange/purple/green
- [ ] **시작 포즈 (x, y, θ)** 제공값 → `mission.start_x/y/theta`
- [ ] 월드 크기 확인 → `map.half_size` ≥ (시작점에서 가장 먼 구석까지 거리)

## 1. 로봇 확인 (10분) — upstream 예제로 이미 확인된 값

| 항목 | 확인된 값 (upstream 기준) | config 반영 |
|---|---|---|
| 로봇 | TurtleBot3 Burger | `robot.*` 기본값 그대로 |
| 모터 | "left/right wheel motor" | robot_io 자동 탐색 |
| LiDAR | "LDS-01", 360빔 | `ranges[0]=후방, [90]=좌, [180]=전방, [270]=우` → `index0_angle=π, ccw=False` |
| 카메라 | 640×480, FOV 1.0472 | robot_io가 **자동 캘리브레이션** |
| 컴퍼스 | 있음 | 자동 사용 (시작 헤딩으로 오프셋 보정) |
| basicTimeStep | 64ms | 자동 (`getBasicTimeStep`) |

- [ ] **LiDAR 방향 검증 (필수 30초)**: 로봇 전방에 벽/손을 두고
      `ranges[180]`이 짧아지는지 확인. 아니라면 `lidar.index0_angle`/`ccw` 수정.
- [ ] 카메라가 로봇 전방을 보는지 Webots 오버레이로 확인.

## 2. 탐지 캘리브레이션 (10분)

- [ ] 목표 사과를 카메라에 비추고 `out/live_map.png` 또는 로그로 탐지 확인.
- [ ] 안 잡히면 HSV 완화: `s_lo`, `v_lo`를 낮춘다 (조명이 어두우면 V부터).
- [ ] 다른 색 사과(교란물)가 같이 잡히면 H 범위를 좁힌다.
- [ ] **디코이 자동 방어 확인** (apartment 검증됨): 화면 상단에 닿는 블롭
      기각(소화기 등 키 큰 빨간 물체), 프레임 20% 초과 블롭 기각(벽·러그),
      종횡비 게이트(캔·표지판). 목표가 책상 '위'에 올려지는 규칙이면
      상단 접촉 기각이 오작동할 수 있음 — 그때만 `detection.py`의
      top_clipped 기각을 완화.
      주최측 baseColor: red=(1,0,0) H≈0 / orange=(1,0.73,0) H≈44 /
      purple=(0.56,0,1) H≈274 / green=텍스처(넓게 60~150).
- [ ] `detection.target_width`: 사과 지름 ≈ **0.10m** (scale 바뀌면 수정)
      — 단안 거리 추정의 기준이므로 중요.

## 3. 첫 주행 (15분)

- [ ] SPIN → EXPLORE 전환 확인, `out/live_map.png`에서 지도가 상식적으로
      쌓이는지 확인 (벽이 이중으로 생기면 odometry/LiDAR 방향 문제).
- [ ] 목표 발견 시 GOTO_TARGET 전환 + est 위치가 실제와 비슷한지.
- [ ] 복귀까지 end-to-end 1회 완주.

## 4. 흔한 증상 → 원인

| 증상 | 원인 후보 |
|---|---|
| 지도가 회전하며 뭉개짐 | LiDAR `ccw`/`index0_angle` 반대 |
| 제자리 회전만 함 | 모터 좌우 뒤바뀜 (NAME_HINTS) |
| 벽이 이중 | odometry 드리프트 — 컴퍼스 연결 확인 |
| 목표 위치 추정이 항상 멀리 | `target_width` 과대 (blob 대비) |
| 좁은 문을 못 지나감 | `plan.inflate_margin` 축소 (0.10→0.06) |
| 사람 앞에서 영원히 대기 | `dyn_yield_dist` 축소 or `dyn_commit_time` 확대 |
| A*가 자주 실패 | `map.half_size` 부족 (목표가 지도 밖) — 크게 |

## 5. 제출 전 (마지막 15분)

- [ ] 파라미터 동결. 마지막 완주 성공 세팅으로 롤백할 수 있게 커밋.
- [ ] `viz` 스냅샷 끄기(`SNAPSHOT_EVERY_S = 0`) — 성능 여유 확보.
- [ ] 최종 실행 1회 확인 후 손 떼기.
