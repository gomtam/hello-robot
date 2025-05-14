import cv2
import time
import os
import sys
import torch
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QComboBox, QFrame
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import Qt, QTimer
from serial_port_selector import SerialPortSelector
import serial
from serial.tools.list_ports import comports
from ultralytics import YOLO

# YOLOv5n 모델 경로
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "yolov5n.pt")

class HumanoidControlApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO 객체 인식 로봇")
        self.setGeometry(100, 100, 1200, 800)
        
        # 변수 초기화
        self.motion_ready = False
        self.is_person_detected = False
        self.last_greeting_time = 0
        self.greeting_cooldown = 5  # 인사 간격 (초)
        self.detected_objects = []  # 감지된 객체 목록
        self.last_person_disappear_time = 0  # 마지막으로 사람이 사라진 시간
        self.person_absence_threshold = 5  # 사람이 사라진 후 다시 인식되면 인사할 시간 간격(초)
        self.confidence_threshold = 0.5  # 객체 인식 신뢰도 임계값 (0.5 = 50%)
        
        # YOLO 모델 설정
        self.model = None
        self.model_path = DEFAULT_MODEL_PATH
        self.class_names = []
        
        # 카메라 설정
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "카메라 오류", "카메라를 열 수 없습니다.")
            sys.exit(1)
        
        # 시리얼 포트 자동 설정
        self.available_ports = []
        self.serial_port = None
        
        # UI 구성
        self.setup_ui()
        
        # 시리얼 포트 자동 감지
        self.auto_detect_port()
        
        # YOLO 모델 로드
        self.load_model()
        
        # 타이머 설정 (카메라 프레임 업데이트용)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # 30ms 간격 (약 33fps)
    
    def setup_ui(self):
        # 중앙 위젯 설정
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 메인 레이아웃
        main_layout = QHBoxLayout()
        
        # 왼쪽 영역 (카메라 뷰)
        left_layout = QVBoxLayout()
        
        # 카메라 뷰
        self.camera_label = QLabel()
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setMinimumSize(640, 480)
        left_layout.addWidget(self.camera_label)
        
        # 상태 정보
        self.status_label = QLabel("상태: 모델 로딩 중...")
        self.status_label.setFont(QFont("Arial", 10))
        left_layout.addWidget(self.status_label)
        
        # 모델 정보 레이아웃
        model_layout = QHBoxLayout()
        
        # 모델 정보 레이블
        self.model_info_label = QLabel(f"모델: {os.path.basename(self.model_path)}")
        model_layout.addWidget(self.model_info_label)
        
        left_layout.addLayout(model_layout)
        
        # 시리얼 포트 정보 레이아웃
        port_layout = QHBoxLayout()
        
        # 포트 정보 레이블
        self.port_label = QLabel("포트: 자동 감지 중...")
        port_layout.addWidget(self.port_label)
        
        # 포트 새로고침 버튼
        self.refresh_port_button = QPushButton("포트 새로고침")
        self.refresh_port_button.clicked.connect(self.auto_detect_port)
        port_layout.addWidget(self.refresh_port_button)
        
        # 수동 인사 버튼 
        self.greet_button = QPushButton("인사하기")
        self.greet_button.clicked.connect(lambda: self.exeHumanoidMotion(19))  # 인사 모션 ID 19
        port_layout.addWidget(self.greet_button)
        
        # 카메라 활성화/비활성화 버튼
        self.camera_toggle_button = QPushButton("카메라 끄기")
        self.camera_toggle_button.clicked.connect(self.toggle_camera)
        port_layout.addWidget(self.camera_toggle_button)
        
        left_layout.addLayout(port_layout)
        
        # 추가 모션 버튼 레이아웃
        motion_layout = QHBoxLayout()
        
        # 모션 버튼들
        motion_buttons = [
            ("차렷", 1),
            ("앉기", 115),
            ("일어서기", 116),
            ("오른쪽 돌기", 29),
            ("왼쪽 돌기", 28)
        ]
        
        for text, motion_id in motion_buttons:
            button = QPushButton(text)
            button.clicked.connect(lambda checked, id=motion_id: self.exeHumanoidMotion(id))
            motion_layout.addWidget(button)
        
        left_layout.addLayout(motion_layout)
        
        # 왼쪽 영역을 메인 레이아웃에 추가
        main_layout.addLayout(left_layout, 7)  # 비율 7
        
        # 오른쪽 영역 (객체 인식 정보)
        right_widget = QFrame()
        right_widget.setFrameShape(QFrame.StyledPanel)
        right_layout = QVBoxLayout()
        
        # 객체 인식 제목
        object_title = QLabel("인식된 객체")
        object_title.setFont(QFont("Arial", 14, QFont.Bold))
        object_title.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(object_title)
        
        # 객체 인식 목록
        self.object_list = QLabel("인식된 객체가 없습니다")
        self.object_list.setFont(QFont("Arial", 12))
        self.object_list.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.object_list.setWordWrap(True)
        self.object_list.setMinimumWidth(300)
        right_layout.addWidget(self.object_list)
        
        # 객체 수 표시
        self.object_count = QLabel("총 객체 수: 0")
        self.object_count.setFont(QFont("Arial", 12, QFont.Bold))
        right_layout.addWidget(self.object_count)
        
        # 마지막 인사 시간 정보
        self.greeting_info = QLabel("마지막 인사: 없음")
        self.greeting_info.setFont(QFont("Arial", 11))
        right_layout.addWidget(self.greeting_info)
        
        # 사람 부재 시간 정보
        self.absence_info = QLabel("사람 부재 시간: 0초")
        self.absence_info.setFont(QFont("Arial", 11))
        right_layout.addWidget(self.absence_info)
        
        # 여백 추가
        right_layout.addStretch()
        
        # 오른쪽 위젯에 레이아웃 적용
        right_widget.setLayout(right_layout)
        
        # 오른쪽 영역을 메인 레이아웃에 추가
        main_layout.addWidget(right_widget, 3)  # 비율 3
        
        # 레이아웃 적용
        central_widget.setLayout(main_layout)
    
    def auto_detect_port(self):
        """시리얼 포트 자동 감지"""
        try:
            self.available_ports = [port.device for port in comports()]
            
            if not self.available_ports:
                self.port_label.setText("포트: 연결된 포트 없음")
                self.motion_ready = False
                return
            
            # 일반적으로 첫 번째 포트를 선택
            selected_port = self.available_ports[0]
            self.port_label.setText(f"포트: {selected_port} (자동)")
            self.motion_ready = True
            
            self.status_label.setText(f"상태: 포트 {selected_port}에 자동 연결됨")
        except Exception as e:
            self.port_label.setText("포트: 오류 발생")
            self.status_label.setText(f"상태: 포트 감지 오류 - {str(e)}")
            self.motion_ready = False
    
        # select_model 함수는 더 이상 사용하지 않음 (코드 레벨에서만 모델 변경)
    
    def load_model(self):
        """YOLO 모델 로드"""
        try:
            if not os.path.exists(self.model_path):
                QMessageBox.warning(self, "모델 로드 오류", f"모델 파일을 찾을 수 없습니다: {self.model_path}")
                return
                
            self.model = YOLO(self.model_path)
            self.status_label.setText(f"상태: YOLO 모델 로드 완료 ({os.path.basename(self.model_path)}, 신뢰도 임계값: {self.confidence_threshold*100:.0f}%)")
            # 모델이 YOLOv8인 경우 클래스 이름 가져오기
            self.class_names = self.model.names if hasattr(self.model, 'names') else []
        except Exception as e:
            self.model = None
            QMessageBox.warning(self, "모델 로드 오류", f"모델을 로드하는 중 오류가 발생했습니다: {str(e)}")
            self.status_label.setText("상태: YOLO 모델 로드 실패")
    
    def toggle_camera(self):
        if self.timer.isActive():
            self.timer.stop()
            self.camera_toggle_button.setText("카메라 켜기")
        else:
            self.timer.start(30)
            self.camera_toggle_button.setText("카메라 끄기")
    
    def update_frame(self):
        if not self.model:
            return  # 모델이 로드되지 않은 경우 처리하지 않음
            
        ret, frame = self.cap.read()
        if not ret:
            return
        
        # 프레임 뒤집기 (거울 모드)
        frame = cv2.flip(frame, 1)
        
        # 감지된 객체 목록 초기화
        self.detected_objects = []
        
        # 현재 시간
        current_time = time.time()
        
        # 사람 감지 여부
        person_detected = False
        
        # YOLO 객체 감지
        try:
            results = self.model(frame)
            
            for result in results:
                boxes = result.boxes
                for i, box in enumerate(boxes):
                    # 바운딩 박스 좌표
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # 클래스 ID와 신뢰도
                    cls_id = int(box.cls[0].item())
                    conf = box.conf[0].item() * 100
                    
                    # 신뢰도가 임계값 미만인 객체는 무시
                    if conf < (self.confidence_threshold * 100):
                        continue
                    
                    # 클래스 이름
                    if cls_id in self.model.names:
                        cls_name = self.model.names[cls_id]
                    else:
                        cls_name = f"Class {cls_id}"
                    
                    # 바운딩 박스와 텍스트 그리기
                    color = (0, 255, 0)  # 기본 색상 (녹색)
                    
                    # 사람(person) 클래스인 경우
                    if cls_name.lower() == 'person':
                        color = (0, 0, 255)  # 빨간색
                        person_detected = True
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{cls_name} {conf:.1f}%", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    
                    # 객체 목록에 추가
                    self.detected_objects.append({"type": cls_name, "confidence": conf})
            
        except Exception as e:
            print(f"YOLO 감지 오류: {str(e)}")
        
        # 사람 감지 상태 변경 처리
        if person_detected:
            # 이전에 사람이 감지되지 않았고, 현재 처음 감지된 경우
            if not self.is_person_detected:
                self.is_person_detected = True
                self.status_label.setText("상태: 사람 감지됨")
                
                # 사람이 사라진 후 지정된 시간(예: 5초) 이상 지났으면 인사
                time_since_disappear = current_time - self.last_person_disappear_time
                if (self.last_person_disappear_time == 0 or 
                    time_since_disappear > self.person_absence_threshold) and self.motion_ready:
                    self.exeHumanoidMotion(19)  # 인사 모션 ID 19
                    self.last_greeting_time = current_time
                    self.greeting_info.setText(f"마지막 인사: {time.strftime('%H:%M:%S')}")
        else:
            # 사람이 감지되지 않는 경우
            if self.is_person_detected:
                # 방금 사라진 경우, 시간 기록
                self.is_person_detected = False
                self.last_person_disappear_time = current_time
                self.status_label.setText("상태: 대기 중")
        
        # 사람 부재 시간 계산 및 표시
        if not self.is_person_detected and self.last_person_disappear_time > 0:
            absence_time = current_time - self.last_person_disappear_time
            self.absence_info.setText(f"사람 부재 시간: {absence_time:.1f}초")
        else:
            self.absence_info.setText("사람 부재 시간: 0초")
        
        # 객체 정보 업데이트
        self.update_object_info()
        
        # OpenCV 프레임을 QPixmap으로 변환
        h, w, c = frame.shape
        bytes_per_line = 3 * w
        q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(q_img)
        self.camera_label.setPixmap(pixmap)
    
    def update_object_info(self):
        """객체 정보 레이블 업데이트"""
        if not self.detected_objects:
            self.object_list.setText("인식된 객체가 없습니다")
            self.object_count.setText("총 객체 수: 0")
            return
        
        # 동일한 유형의 객체는 그룹화하여 카운트
        object_types = {}
        for obj in self.detected_objects:
            obj_type = obj["type"]
            if obj_type in object_types:
                object_types[obj_type]["count"] += 1
                object_types[obj_type]["confidence"] = max(object_types[obj_type]["confidence"], obj["confidence"])
            else:
                object_types[obj_type] = {"count": 1, "confidence": obj["confidence"]}
        
        # 객체 정보 문자열 생성
        object_info = ""
        for i, (obj_type, data) in enumerate(sorted(object_types.items()), 1):
            if data["count"] > 1:
                object_info += f"{i}. {obj_type} x{data['count']} (신뢰도: {data['confidence']:.1f}%)\n"
            else:
                object_info += f"{i}. {obj_type} (신뢰도: {data['confidence']:.1f}%)\n"
        
        self.object_list.setText(object_info)
        self.object_count.setText(f"총 객체 수: {len(self.detected_objects)}")
    
    def exeHumanoidMotion(self, motion_id):
        if not self.motion_ready:
            QMessageBox.warning(self, "모션 오류", "로봇이 준비되지 않았습니다. 포트를 자동으로 감지 중입니다.")
            self.auto_detect_port()
            return

        # 모션 제어 기본 패킷 생성
        packet_buff = [0xff, 0xff, 0x4c, 0x53,  # 헤더
                       0x00, 0x00, 0x00, 0x00,  # Destination ADD, Source ADD
                       0x30, 0x0c, 0x03,        # 0x30 실행 명령어 0x0c 모션실행 0x03 파라메타 길이
                       motion_id, 0x00, 0x64,   # 모션 ID, 모션 반복, 모션 속도 지정
                       0x00]                    # 체크섬

        # 체크섬 계산 (바이트 범위 제한 추가)
        checksum = 0
        for i in range(6, 14):
            checksum += packet_buff[i]
        packet_buff[14] = checksum % 256  # 256으로 나눈 나머지를 사용하여 0-255 범위로 제한

        # 모든 값이 바이트 범위 내에 있는지 확인
        for i in range(len(packet_buff)):
            if not (0 <= packet_buff[i] <= 255):
                QMessageBox.warning(self, "패킷 오류", f"패킷의 {i}번째 값 {packet_buff[i]}이(가) 바이트 범위를 벗어났습니다.")
                return

        # 시리얼 포트 열기
        try:
            port_name = self.port_label.text().replace("포트: ", "").split(" ")[0]
            if "연결된 포트 없음" in port_name or "오류" in port_name:
                raise serial.SerialException("포트가 선택되지 않았습니다.")
            
            # 바이트 배열로 변환
            byte_packet = bytearray(packet_buff)
            
            ser = serial.Serial(port_name, 115200, timeout=1)
            if ser.is_open:
                # 패킷 전송
                ser.write(byte_packet)
                self.status_label.setText(f"상태: 모션 실행 중 (ID: {motion_id})")
            else:
                raise serial.SerialException(f"시리얼 포트 {port_name}를 열 수 없습니다.")
                
        except serial.SerialException as e:
            # 경고 메시지 박스 표시
            QMessageBox.warning(self, "시리얼 포트 오류", str(e))
            # 포트 재감지 시도
            self.auto_detect_port()
        except ValueError as e:
            # 바이트 변환 오류 처리
            QMessageBox.warning(self, "패킷 오류", f"패킷 변환 중 오류 발생: {str(e)}")
        finally:
            if 'ser' in locals() and ser.is_open:
                ser.close()
    
    def closeEvent(self, event):
        # 종료 시 자원 해제
        if self.cap.isOpened():
            self.cap.release()
        event.accept()

if __name__ == "__main__":
    # PyQt 애플리케이션 초기화
    app = QApplication(sys.argv)
    
    # 메인 윈도우 생성
    window = HumanoidControlApp()
    window.show()
    
    # 애플리케이션 실행
    sys.exit(app.exec_())
