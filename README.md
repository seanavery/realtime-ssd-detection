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

This SSD model uses [mobilenet-v2]() and is trained on the [Coco Dataset](https://cocodataset.org/) which expects incoming images to be of shape `(300, 300, 3)`. First we need to resize the incoming `(1920, 1080, 3)`, Lastley, return the flattened image of size (270000).  We also need to normalize th input matrix to values between `[0, -1]`. Keep in mind that these numpy functions are not yet hardware optimized.

```
def pre_process(self, frame):
    # convert to 300 * 300
    frame = cv2.resize(frame, (300, 300))

    # normalize tensor
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame.transpose((2, 0, 1)).astype(np.float32)
    frame *= (2.0/255.0)
    frame -= 1.0

    return frame
```

### 4. Inference

First, copy the flattend image as a 1d tensor into cuda memory using the `memcpy_htod_async` function. Then execute inference with a batch size of 1 since we are feeding in one frame at a time. Lastly, we will need to copy the output tensor back from the GPU to the Host so we can use the CPU to post-process.

```
def infer(self, frame):
    np.copyto(self.host_inputs[0], frame)

    # copy buffer into cuda, serialize via stream
    cuda.memcpy_htod_async(
        self.cuda_inputs[0], self.host_inputs[0], self.stream)
    pre_end = time.time()

    # execute inference async
    self.context.execute_async(
        batch_size=1,
        bindings=self.bindings, # input/output buffer addresses
        stream_handle=self.stream.handle)
    cuda.memcpy_dtoh_async(
        self.host_outputs[1], self.cuda_outputs[1], self.stream)
    cuda.memcpy_dtoh_async(
        self.host_outputs[0], self.cuda_outputs[0], self.stream)

    self.stream.synchronize()
    output = self.host_outputs[0]
```

### 5. Postprocess Image

The output tensor is flat with results seperated by 7 `(cls, conf, x1, y1, x2, y2)`. We simply ned to iterate through the tensor and filter out boxes that are less then a defined confidence threshold `[0, 1]`.

```
def post_process(self, frame, output, confidence_threshold):
    img_h, img_w, _ = frame.shape
    boxes, confs, clss = [], [], []
    for prefix in range(0, len(output), 7):
        confidence = float(output[prefix+2])
        if confidence < confidence_threshold:
            continue
        x1 = int(output[prefix+3] * img_w)
        y1 = int(output[prefix+4] * img_h)
        x2 = int(output[prefix+5] * img_w)
        y2 = int(output[prefix+6] * img_h)
        cls = int(output[prefix+1])
        boxes.append((x1, y1, x2,  y2))
        confs.append(confidence)
        clss.append(cls)
    return boxes, confs, clss
```

### 6. Draw Boxes and Visualize Results

First we need to generate a unique color for every one of the 80 coco classes. First, generate a unique hsv value for every class by varying the hue. Then convert to bgr values for use in OpenCV. To draw boxes with translucent box, we first need to draw a filled rectangle on an overlay copy of the image, and then apply the overlay with a `alpha` opacity value.

```
class Visualizer():
    def __init__(self):
        self.color_list = self.gen_colors()

    def gen_colors(self):
        # generate random hues
        hsvs = []
        for x in range(len(COCO_CLASSES_LIST)):
            hsvs.append([float(x) / len(COCO_CLASSES_LIST), 1., 0.7])
        random.seed(3344)
        random.shuffle(hsvs)
        
        # convert hsv to rgb values
        rgbs = []
        for hsv in hsvs:
            (h, s, v) = hsv
            rgb = colorsys.hsv_to_rgb(h, s, v)
            rgbs.append(rgb)

        # convert to bgr and (0-255) range
        bgrs = []
        for rgb in rgbs:
            bgr = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
            bgrs.append(bgr)

        return bgrs

    def draw(self, frame, boxes, confs, clss):
        overlay = frame.copy()
        for bb, cf, cl in zip(boxes, confs, clss):
            x_min, y_min, x_max, y_max = bb[0], bb[1], bb[2], bb[3]
            cls = COCO_CLASSES_LIST[cl]
            color = self.color_list[cl]
            print('cls', cls) 
            if cls == 'dining table' or cls == 'suitcase':
                print("here")
                continue
            print('color', color)
            cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), color, -1)
            cv2.putText(frame, cls, (x_min + 20, y_min + 20), fontScale=1, fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(255, 255, 255))
        
        alpha = 0.4

        return cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0)
```

## Conclusion
