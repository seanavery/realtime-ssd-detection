# Real-time SSD Object Detection

> Gstreamer, Opencv, Tensort camera pipeline

## Materials

1. Jetson hardware with camera CSI attachment. See the [previous article](https://seanavery.github.io/jetson-nano-box/#/) for details on how to build.
2. Need to have OpenCV installed. See [instructions](https://jkjung-avt.github.io/opencv-on-nano/) from  JK Jung's blog.

## Procedure

The goal for this blog post, is to write the "hello world" of computer vision. In other workds, get some object detection neural net running and visualize results.

### 0. GStreamer OpenCV Camera Capture

We can write a simple python script to feed a GStreamer pipeline into a `cv2::VideoCapture` object. First [nvarguscamerasrc]() is used for camera bringup and to set the `whitebalance` mode to auto. Then the frame is flipped 180 degrees and formatted as BGR for Opencv [VideoCapture]() ingestion. 

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

As you can see from the gif above, the whitebalance and auto exposure works quite well on the Sony IMX219. I have a lamp off to the side in a somewhat dark room, purposefully less than ideal. The white balance and expose adjust over a few second timeframe. 

### 2. Initialize TensorRT Engine

For now we will skip the process of converting the [model](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md#coco-trained-models) from Tensorflow's Model Zoo into TensorRT. All we need to do is load up the `trt` engine and initialize the GPU memory using [pycuda](https://documen.tician.de/pycuda/). 

The model has been loaded with [graph-surgeon](https://github.com/jkjung-avt/tensorrt_demos/blob/master/ssd/build_engine.py) plugins to make the model compatible with TensorRT layers specification. You need to first regsiter the plugins with `libnvinfer`. The next step is to load up the trt file and allocate `GPU` and `Host` memory for input/output layers.

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

The SSD model is based on the [Coco Dataset](https://cocodataset.org/) which expects incoming images to be of shape `(300, 300, 3)`. First we need to resize the incoming (1920, 1080, 3), then we returned the flattened image of size (270000). Keep in mind that these numpy functions are not yet hardware optimized.

```
def pre_process(self, frame):
    frame = cv2.resize(frame, (300, 300))
    return frame.ravel()
```

### 4. Inference

def infer(self, frame):


### 5. Postprocess Image

### 6. Draw Boxes and Visualize Results

## Conclusion
