# Introduction
눈동자를 트래킹해 운전자의 행동(졸음 운전 or 스마트폰)을 감지합니다.

Detect driver behavior (drowsy driving or smartphone) by tracking eye movements.

![20240621_122948](https://github.com/kms13601/Detecting-driver-behavior-with-eye-tracking/assets/150672183/eeebf6bd-fedf-4f4b-b4fc-ef8bf2ecc307)

# Installation
Clone this project:

    git clone https://github.com/kms13601/DriverGaze.git
    cd DrivaerGaze
    git clone https://github.com/antoinelame/GazeTracking.git

# How to use
main.py 에서 video_path, out을 수정해서 사용하세요

In main.py, modify video_path, out to use

    video_path = 'modify here'
    out = cv2.VideoWriter('modify here', fourcc, cap.get(cv2.CAP_PROP_FPS), output_size)
