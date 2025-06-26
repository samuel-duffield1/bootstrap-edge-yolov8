import os
import time

from loguru import logger

from edge_yolo_demo.cameras.ovc_camera import OcvCamera
from edge_yolo_demo.displays.web_display import WebDisplay
from edge_yolo_demo.models.ultralytics_model import UltralyticsYolov8n

LOCAL_CAMERA_SOURCE = int(os.environ.get("LOCAL_CAMERA_SOURCE", "0"))
CAMERA_FOV_H = int(os.environ.get("CAMERA_FOV_H", "90"))
CAMERA_FOV_W = int(os.environ.get("CAMERA_FOV_W", "90"))
PIXEL_WINDOW_X = 100


def main():
    camera = OcvCamera(LOCAL_CAMERA_SOURCE, CAMERA_FOV_H, CAMERA_FOV_W)
    model = UltralyticsYolov8n()
    display = WebDisplay()

    while True:
        image = camera.get_frame()
        if image is not None:
            trackables, output_image = model.perform_inference(image)
            if trackables is not None:
                for trackable in trackables:
                    logger.info(f"Trackable: {trackable.to_dict()}")
            if output_image is not None:
                display.send_image(output_image)


if __name__ == "__main__":
    print("Starting edge-yolo-demo...")
    while True:
        try:
            main()
        except Exception as e:
            print(f"An error occurred: {e}")
        time.sleep(5)  # Wait before retrying
        print("Restarting the application...")
