import math
import os

import cv2
import numpy
from ultralytics import YOLO

from edge_yolo_demo.cameras.image import Frame, encode
from edge_yolo_demo.models.model_output import Trackable

# Directory path to external model files
EXTERN_FILES_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "assets", "Ultralytics"
)

# COCO dataset class names used by YOLOv8 model (80 classes total)
# These correspond to the standard COCO object detection dataset categories
ULTRALYTICS_CLASSNAMES = [
    "person",
    "bicycle",
    "car",
    "motorbike",
    "aeroplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "sofa",
    "pottedplant",
    "bed",
    "diningtable",
    "toilet",
    "tvmonitor",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


class UltralyticsYolov8n:
    """
    YOLOv8 Nano object detection model implementation using Ultralytics.

    This class provides a wrapper around the Ultralytics YOLOv8 nano model
    for real-time object detection. It focuses specifically on person detection,
    processing input frames and returning both trackable objects and annotated
    output frames.

    The implementation includes:
    - Automatic image preprocessing (resize to 640x480)
    - Person detection with bounding box visualization
    - Confidence score processing
    - Trackable object creation with spatial coordinates
    - Frame annotation with detection results

    Note:
        Currently configured to detect only "person" class objects from the
        full COCO dataset. Other detected objects are ignored.
    """

    def __init__(self) -> None:
        """
        Initialize the YOLOv8 nano model.

        Loads the pre-trained YOLOv8 nano model from the assets directory
        and sets up the model configuration for inference.

        Raises:
            FileNotFoundError: If the yolov8n.pt model file cannot be found
            RuntimeError: If the model fails to initialize
        """
        super().__init__()
        # Load the YOLOv8 nano model with verbose output disabled
        self.model = YOLO(os.path.join(EXTERN_FILES_DIR, "yolov8n.pt"), verbose=False)
        self.model_name = "Ultralytics YoloV8n - edge-yolo-demo"

    def perform_inference(self, original_image: Frame) -> tuple[list[Trackable], Frame]:
        """
        Perform object detection inference on the input frame.

        This method processes an input frame through the YOLOv8 model to detect
        objects (specifically persons), draws bounding boxes and labels on the
        image, and creates trackable objects with spatial information.

        Processing steps:
        1. Decode the input frame from JPEG to OpenCV format
        2. Resize image to 640x480 for model input
        3. Run YOLOv8 inference
        4. Filter results for "person" class only
        5. Draw bounding boxes and labels on the image
        6. Create Trackable objects with spatial coordinates
        7. Encode the annotated image back to Frame format

        Args:
            original_image (Frame): Input frame containing image data and metadata

        Returns:
            tuple[list[Trackable], Frame]: A tuple containing:
                - List of Trackable objects representing detected persons
                - Annotated Frame with bounding boxes and labels drawn

        Note:
            - Only detects "person" class objects, other detections are filtered out
            - Confidence scores are rounded to 2 decimal places
            - Each detected person gets a unique identifier (person0, person1, etc.)
            - Coordinate system is centered on the frame with field of view context
        """
        # Decode JPEG frame data to OpenCV image format
        data = numpy.array(original_image.data)
        img = cv2.imdecode(buf=data, flags=cv2.IMREAD_COLOR)

        # Resize image to standard input size for YOLOv8 model
        img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_LINEAR)

        # Perform inference with streaming results for memory efficiency
        results = self.model(img, stream=True, verbose=False)

        # Initialize tracking variables
        trackables: list[Trackable] = []
        count = 0

        # Process detection results
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Extract class information
                cls = int(box.cls[0])
                class_name = ULTRALYTICS_CLASSNAMES[cls]

                # Filter for person detections only
                if class_name == "person":
                    # Extract bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    # Draw bounding box rectangle (magenta color)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                    # Set up text properties for labels
                    org = [x1, y1]
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    fontScale = 1
                    color = (255, 0, 0)  # Blue color for text
                    thickness = 2

                    # Create unique identifier for this detection
                    unique_name = f"{ULTRALYTICS_CLASSNAMES[cls]}{count}"

                    # Draw model name at top of frame
                    cv2.putText(img, self.model_name, [0, 25], font, 0.5, color, 1)

                    # Draw unique identifier label on bounding box
                    cv2.putText(
                        img, unique_name, org, font, fontScale, color, thickness
                    )

                    # Calculate center point of detection
                    midpoint = ((x1 + x2) / 2, (y1 - y2) / 2)

                    # Convert to field-of-view centered coordinate system
                    centre_x = midpoint[0] - original_image.field_of_view_w / 2
                    centre_y = midpoint[1] - original_image.field_of_view_h / 2

                    # Round confidence score to 2 decimal places
                    confidence = math.ceil((box.conf[0] * 100)) / 100

                    # Create trackable object with detection information
                    t = Trackable(
                        x=int(centre_x),
                        y=int(centre_y),
                        frame_pixels_wide=img.shape[1],
                        frame_pixels_high=img.shape[0],
                        field_of_view_w=original_image.field_of_view_w,
                        field_of_view_h=original_image.field_of_view_h,
                        confidence=confidence,
                        class_name=class_name,
                        unique_name=unique_name,
                    )
                    count += 1
                    trackables.append(t)

        # Return both trackable objects and annotated frame
        return (
            trackables,
            encode(img, original_image.field_of_view_w, original_image.field_of_view_h),
        )
