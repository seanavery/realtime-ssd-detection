import time
import cv2

from Processor import Processor
from Visualizer import Visualizer

def stream_camera():
    pipeline = (
        "nvarguscamerasrc wbmode=1 !"
        "nvvidconv flip-method=2 ! "
        "videoconvert ! video/x-raw, format=(string)BGR !"
        "appsink"
    )
    video_capture = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if video_capture.isOpened():
        while True:
            window = cv2.namedWindow("Camera", cv2.WINDOW_AUTOSIZE)
            _, frame = video_capture.read()
            if frame is not None:
                pre = time.time()
                boxes, confs, clss = processor.detect(frame)
                # frame = processor.detect(frame)
            prev = time.time()
            frame = vis.draw(frame, boxes, confs, clss)
            height, width = frame.shape[:2]
            print('height', height)
            print('width', width)
            cv2.imshow("Camera", frame)
            keyCode = cv2.waitKey(1) & 0xFF
            if keyCode == 27:
                print('hit escape')
                break
        video_capture.release()
        cv2.destroyAllWindows()
    else:
        print("could not open camera")

if __name__ == "__main__":
    # processor = SSD()
    vis = Visualizer((0, 255, 0))
    processor = Processor()
    stream_camera()
