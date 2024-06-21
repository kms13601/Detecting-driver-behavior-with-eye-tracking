import cv2
import dlib
from GazeTracking.gaze_tracking import GazeTracking
from utils import cal_rec, eyes_ext, distraction_dt

# GazeTracking 초기화
gaze = GazeTracking()

# 얼굴 검출기 초기화
detector = dlib.get_frontal_face_detector()

# 랜드마크 검출기 초기화
sp_path = 'shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(sp_path)

# 변수 설정
EYE_AR_THRESH = 0.15       # 졸음 임계값
EYE_AR_CONSEC_FRAMES = 36  # 졸음을 인식하는 시간

COUNTER = 0     # 졸음 카운터
d_COUNTER = 0   # 딴짓 카운터
DROWSY_COUNT = 0  # 누적 졸음 횟수
DISTRACTION_FRAMES = 15  # 딴짓 임계값
text = ""          # 출력되는 텍스트
eye_open = True    # 눈을 떴는지 감았는지 체크

# 비디오 캡처 객체 생성
video_path = 'modify here'
cap = cv2.VideoCapture(video_path)

ret, frame = cap.read()

if not ret:
    print("Error: Could not read frame from video.")
    cap.release()
    exit()

# 얼굴 검출
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
rects = detector(gray, 0)

# 첫 프레임에서 눈 영역의 크기를 계산
if len(rects) > 0:
    output_size = cal_rec(gray, rects, predictor)
else:
    print("Error: Could not detect face in the first frame.")
    cap.release()
    exit()

# 비디오 저장을 위한 초기화
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('modify here', fourcc, cap.get(cv2.CAP_PROP_FPS), output_size)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 졸음 감지
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    if len(rects) > 0:
        ear, x_min_extended, y_min_extended, x_max_extended, y_max_extended = eyes_ext(predictor, gray, rects)
        cropped_frame = frame[y_min_extended:y_max_extended, x_min_extended:x_max_extended]

        if ear < EYE_AR_THRESH:     # 눈이 감기는 프레임 수 세기
            eye_open = False
            COUNTER += 1
            if COUNTER >= EYE_AR_CONSEC_FRAMES: # 일정 프레임 동안 눈을 감는다면
                DROWSY_COUNT += 1               # 누적 졸음 +1
                COUNTER = 0
        else:
            eye_open = True
            COUNTER = 0

    # 딴짓 감지
    frame, text, d_COUNTER = distraction_dt(gaze, text, frame, eye_open, d_COUNTER, DISTRACTION_FRAMES)

    # 화면에 텍스트 추가
    cv2.putText(frame, text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(cropped_frame, text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # 졸음 누적 횟수 표시
    cv2.putText(frame, f"Total drowsy: {DROWSY_COUNT}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(cropped_frame, f"Total drowsy: {DROWSY_COUNT}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # 총 3회 누적 시 "차를 멈추세요" 메시지 표시
    if DROWSY_COUNT >= 3:
        cv2.putText(frame, "Stop the car immediately!", (frame.shape[1] // 2 - 150, frame.shape[0] // 2), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                                                                                                (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(cropped_frame, "Stop the car immediately!", (60, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    cropped_frame = cv2.resize(cropped_frame, output_size)
    out.write(cropped_frame)

    cv2.imshow("Drowsiness and Distraction Detection", frame)
    cv2.imshow("Crop", cropped_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 리소스 해제
cap.release()
out.release()
cv2.destroyAllWindows()
