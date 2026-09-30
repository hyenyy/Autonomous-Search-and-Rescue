# Windows / GPU 실행 환경

이 PC에서 확인한 환경은 Webots R2025a, NVIDIA RTX 4060 Ti, Python 3.12, PyTorch 2.11.0+cu128, Ultralytics 8.4.166입니다. Webots 시스템 정보에서 NVIDIA OpenGL 렌더러를 확인했고, YOLO 추론의 CUDA 사용도 별도로 확인했습니다.

프로젝트의 `.venv`에 해당 GPU와 드라이버에 맞는 CUDA PyTorch 빌드를 먼저 설치한 다음 `requirements.txt`를 설치합니다. CUDA 빌드는 [PyTorch 공식 설치 안내](https://pytorch.org/get-started/locally/)에서 선택합니다.

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
.venv\Scripts\python.exe tools/verify_gpu.py
```

`config_override.json`의 `detection.use_yolo`가 `true`, `detection.yolo_device`가 `cuda:0`이어야 합니다. 가중치는 `controllers/sar_controller/yolo11n.pt` 등에 별도로 준비합니다. 모델 가중치와 가상환경은 Git에 포함하지 않습니다.

`controllers/sar_controller/runtime.ini`의 Python 경로는 이 PC의 절대 경로입니다. 다른 PC에서는 해당 PC의 프로젝트 `.venv/Scripts/python.exe`로 변경하고 Webots의 Python command도 맞춰야 합니다. 독립 검증 월드는 `tools/create_validation_world.py`로 생성합니다.

3D 렌더링 GPU 선택과 PyTorch CUDA 장치 선택은 별개입니다. 빠른 실행 배속은 GPU뿐 아니라 물리 연산, 센서, 경로 계획, 이미지 저장에도 영향을 받습니다.
