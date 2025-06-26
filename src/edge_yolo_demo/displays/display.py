from abc import ABC, abstractmethod

from edge_yolo_demo.cameras.image import Frame


class VideoDisplay(ABC):
    def __init__(self) -> None:
        return

    @abstractmethod
    def send_image(self, image: Frame):
        pass
