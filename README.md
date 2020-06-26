# Realtime SSD Object Detection

> Asynchronous TensorRT execution for realtime inference

## Materials

1. Jetson hardware with camera attachment. See the previous article for details on how to build.

## Procedure

### 0. Convert Onnx Model to TensorRT Engine

### 1. GStreamer OpenCV Camera Capture

We can write a simple python script to feed a GStreamer pipeline into a `cv2::VideoCapture` object.

```
import time
import cv2

def stream_camera():
    pipeline = (
        "nvarguscamerasrc wbmode=2 !"
        "nvvidconv flip-method=2 ! "
        "videoconvert ! video/x-raw, format=(string)BGR !"
        "appsink"
    )
    video_capture = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    print('video_capture', video_capture)
    if video_capture.isOpened():
        while True:
            window = cv2.namedWindow("Camera", cv2.WINDOW_AUTOSIZE)
            return_value, frame = video_capture.read()
            cv2.imshow("Camera", frame)
            show_time_end = time.time()
            keyCode = cv2.waitKey(1) & 0xFF
            if keyCode == 27:
                break
        video_capture.release()
        cv2.destroyAllWindows()
    else:
        print("could not open camera")

if __name__ == "__main__":
    stream_camera()
```

### 2. Initialize TensorRT Engine

```
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np

class SSD():
    def __init__(self):
        # initialize logger for debugging
        trt_logger = trt.Logger(trt.Logger.INFO)
        
        # load libnvinfer plugins
        # https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/
        trt.init_libnvinfer_plugins(trt_logger, '')
        
        # instantiate TensorRT engine
        trt_model = 'models/ssd-mobilenet-v2-coco.trt'
        with open(trt_model, 'rb') as f, trt.Runtime(trt_logger) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
        
        # create context
        context = engine.create_execution_context()
        stream = cuda.Stream()

        # memory allocations for input/output layers
        # binding Input
        # binding NMS
        # binding NMS_1
        self.inputs = []
        self.outputs = []
        self.bindings = []
        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding))
            host_mem = cuda.pagelocked_empty(size, np.float32)   
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            if engine.binding_is_input(binding):
                self.inputs.append({ 'host': host_mem, 'cuda': cuda_mem })
            else:
                self.outputs.append({ 'host': host_mem, 'cuda': cuda_mem })
```

### 3. Preprocess Image

### 4. Inference

### 5. Postprocess Image

### 6. Draw Boxes and Visualize Results

## Conclusion
