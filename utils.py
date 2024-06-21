import cv2
import dlib
import numpy as np
from GazeTracking.gaze_tracking import GazeTracking

# 눈 가로세로비(Eye Aspect Ratio) 계산 함수
def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)

# 눈 중심 좌표 계산 함수
def eye_center(eye):
    return np.mean(eye, axis=0).astype(int)

def cal_rec(gray, rects, predictor) :

    shape = predictor(gray, rects[0])
    shape = np.array([[p.x, p.y] for p in shape.parts()])

    # 왼쪽 눈과 오른쪽 눈의 좌표 추출
    left_eye = shape[36:42]
    right_eye = shape[42:48]

    # 좌표로 사각형 그리기
    x_min = min(left_eye[:, 0].min(), right_eye[:, 0].min())
    y_min = min(left_eye[:, 1].min(), right_eye[:, 1].min())
    x_max = max(left_eye[:, 0].max(), right_eye[:, 0].max())
    y_max = max(left_eye[:, 1].max(), right_eye[:, 1].max())

    # 상하좌우로 여백 설정
    x_min_extended = max(x_min - 200, 0)
    y_min_extended = max(y_min - 200, 0)
    x_max_extended = x_max + 80
    y_max_extended = y_max + 100

    # 사각형 크기 계산
    crop_width = x_max_extended - x_min_extended
    crop_height = y_max_extended - y_min_extended

    return crop_width, crop_height

def eyes_ext(predictor, gray, rects):

    shape = predictor(gray, rects[0])
    shape = np.array([[p.x, p.y] for p in shape.parts()])

    # 왼쪽 눈과 오른쪽 눈의 좌표 추출
    left_eye = shape[36:42]
    right_eye = shape[42:48]

    # 눈 가로세로비 계산
    left_ear = eye_aspect_ratio(left_eye)
    right_ear = eye_aspect_ratio(right_eye)

    # 두 눈의 평균 눈 가로세로비
    ear = (left_ear + right_ear) / 2.0

    # 눈 중심 좌표 계산
    left_eye_center = eye_center(left_eye)
    right_eye_center = eye_center(right_eye)

    # 좌표로 사각형 그리기
    x_min = min(left_eye[:, 0].min(), right_eye[:, 0].min())
    y_min = min(left_eye[:, 1].min(), right_eye[:, 1].min())
    x_max = max(left_eye[:, 0].max(), right_eye[:, 0].max())
    y_max = max(left_eye[:, 1].max(), right_eye[:, 1].max())

    # 상하좌우로 여백 설정
    x_min_extended = max(x_min - 200, 0)
    y_min_extended = max(y_min - 200, 0)
    x_max_extended = x_max + 80
    y_max_extended = y_max + 100

    return ear, x_min_extended, y_min_extended, x_max_extended, y_max_extended


def distraction_dt(gaze, text, frame, eye_open, COUNTER, DISTRACTION_FRAMES):
    # GazeTracking을 통한 시선 이탈 감지
    gaze.refresh(frame)
    frame = gaze.annotated_frame()

    if gaze.is_center():    # 시선이 중앙일 때
        text = ""
        COUNTER = 0
    elif not gaze.is_center() and eye_open:  # 시선이 중앙이 아니고 눈을 뜨고 있을 때
        COUNTER += 1
        if COUNTER >= DISTRACTION_FRAMES:
            text = "Stop looking aside!"
            COUNTER = 0

    return frame, text, COUNTER
