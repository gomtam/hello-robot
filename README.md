# 사람 인식 인사 로봇

이 프로젝트는 카메라로 사람을 인식하고, 사람이 감지되면 로봇이 자동으로 인사하는 프로그램입니다. YOLOv5n 객체 인식 모델을 사용하여 사람뿐만 아니라 다양한 객체를 인식할 수 있습니다.

<img src="https://github.com/gomtam/image/blob/main/250514/robot%20(2).jpg" width="400"> <img src="https://github.com/gomtam/image/blob/main/250514/robot%20(1).jpg" width="400">



## 주요 기능

- YOLOv5n 모델을 사용한 실시간 객체 인식 (50% 이상 신뢰도)
- 사람 감지 시 자동 인사 기능 (5초 이상 부재 후 재등장 시)
- 다양한 로봇 모션 제어 (차렷, 앉기, 일어서기, 좌우 회전 등)
- 시리얼 포트 자동 감지 및 연결
- 객체 인식 정보 실시간 표시 (유형, 수량, 신뢰도)
- 사용자 친화적인 GUI 인터페이스

<img src="https://github.com/gomtam/image/blob/main/250514/KakaoTalk_20250514_164538559.png" width="400">

## 개발 환경 및 필요 사항

- Python 3.8 이상
- PyTorch 및 Ultralytics YOLO
- OpenCV
- PyQt5
- PySerial
- 웹캠 또는 USB 카메라
- 시리얼 통신이 가능한 휴머노이드 로봇

## 설치 방법

1. 저장소 복제:
```
git clone https://github.com/your-username/human-detection-robot.git
cd human-detection-robot
```

2. 필요한 패키지 설치:
```
pip install -r requirements.txt
```

3. UI 파일을 위한 리소스 폴더 생성:
```
mkdir -p res
```

4. 필요한 UI 파일을 res 폴더에 복사:
```
cp /path/to/findComPort.ui res/
```

5. YOLOv5n 모델 확인:
```
# models 폴더에 yolov5n.pt 파일이 있는지 확인
mkdir -p models
# 모델이 없는 경우 다운로드
```

6. 프로그램 실행:
```
python main.py
```
### 프로그램 시연 영상<br>
![시연 영상](https://github.com/gomtam/image/blob/main/250514/robot_demo.gif)


## 사용 방법

1. 프로그램을 실행하면 자동으로 카메라가 활성화되고 YOLOv5n 모델이 로드됩니다.
2. 시리얼 포트가 자동으로 감지되며, 필요한 경우 "포트 새로고침" 버튼으로 재검색할 수 있습니다.
3. 카메라에 사람이 50% 이상의 신뢰도로 감지되면 로봇이 자동으로 인사합니다.
4. 사람이 사라진 후 5초 이상 지난 다음 다시 나타나면 다시 인사합니다.
5. UI의 버튼을 통해 차렷, 앉기, 일어서기 등 다양한 모션을 수동으로 제어할 수 있습니다.
6. 오른쪽 패널에서 감지된 모든 객체의 정보를 실시간으로 확인할 수 있습니다.
7. "카메라 끄기" 버튼으로 카메라를 일시 중지할 수 있습니다.

## 기술적 세부 사항

- **객체 인식**: YOLOv5n 모델을 사용하여 80개 이상의 다양한 객체 클래스 인식
- **신뢰도 임계값**: 50% 이상의 신뢰도를 가진 객체만 인식 (코드에서 조정 가능)
- **로봇 통신 프로토콜**: 시리얼 통신을 통한 고유 패킷 포맷으로 로봇 제어
- **모션 ID**: 차렷(1), 앉기(115), 일어서기(116), 오른쪽 돌기(29), 왼쪽 돌기(28), 인사(19)

## 주의 사항

- 로봇 연결 전 시리얼 포트가 자동으로 감지되지 않을 경우 "포트 새로고침" 버튼을 사용하세요.
- 모델 변경이 필요한 경우 코드 레벨에서 `DEFAULT_MODEL_PATH` 변수를 수정하세요.
- 객체 인식 신뢰도 임계값은 `confidence_threshold` 변수를 통해 코드에서 조정할 수 있습니다.
- 로봇의 모션 ID는 모델에 따라 다를 수 있으니 필요시 코드를 수정하세요. 
