import threading

import cv2

from edge_yolo_demo.cameras.camera import CameraException, ICamera
from edge_yolo_demo.cameras.image import Frame, ImageFormat


class OcvCamera(ICamera):
    """
    OpenCV-based camera implementation for real-time frame capture.

    This class provides a concrete implementation of the ICamera interface using
    OpenCV's VideoCapture functionality. It continuously captures frames in a
    background thread and provides the latest frame on demand. The camera
    automatically encodes frames to JPEG format for efficient storage and transmission.

    The class is designed for use with standard USB cameras, webcams, or other
    video capture devices supported by OpenCV.
    """

    def __init__(
        self, capture_device: int, field_of_view_h: int, field_of_view_w: int
    ) -> None:
        """
        Initialize the OpenCV camera with specified device and field of view.

        Args:
            capture_device (int): Camera device index (typically 0 for default camera,
                                1 for second camera, etc.)
            field_of_view_h (int): Vertical field of view in degrees
            field_of_view_w (int): Horizontal field of view in degrees

        Raises:
            CameraException: If the camera device cannot be opened or initialized
        """
        # Initialize the parent ICamera class
        super().__init__(
            field_of_view_w=field_of_view_w, field_of_view_h=field_of_view_h
        )

        # Camera device configuration
        self.capture_Device = capture_device
        self.vid = cv2.VideoCapture(self.capture_Device)

        # Frame storage and threading
        self.latest_frame: Frame = None
        self.capture_thread = threading.Thread(
            target=self._constant_frame_capture,
            daemon=True,
            name="OcvCamera capture thread",
        )
        # Start continuous frame capture in background
        self.capture_thread.start()

    def __del__(self):
        """
        Destructor to properly release camera resources.

        Ensures that the OpenCV VideoCapture object is properly released
        when the camera instance is destroyed, preventing resource leaks
        and allowing other applications to access the camera device.
        """
        if hasattr(self, "vid") and self.vid is not None:
            self.vid.release()

    def get_frame(self) -> Frame:
        """
        Get the most recently captured frame from the camera.

        This method returns the latest frame that was captured by the background
        thread. The frame is already encoded in JPEG format for efficiency.

        Returns:
            Frame: The most recent frame captured by the camera, or None if
                   no frame has been captured yet.

        Note:
            This method is non-blocking and returns immediately with the latest
            available frame. The frame capture happens continuously in a
            background thread.
        """
        return self.latest_frame

    def _constant_frame_capture(self):
        """
        Continuously capture frames from the camera in a background thread.

        This private method runs in an infinite loop, constantly capturing frames
        from the OpenCV VideoCapture object. Each captured frame is immediately
        encoded to JPEG format and stored as the latest frame available for retrieval.

        The method runs in a daemon thread, so it will automatically terminate
        when the main program exits.

        Raises:
            CameraException: If frame capture fails (e.g., camera disconnected,
                           hardware error, or device not available).
        """
        while True:
            # Capture frame from camera
            ret, frame = self.vid.read()
            if not ret:
                error = CameraException("Unable to capture frame")
                raise error

            # Get frame dimensions
            height, width, _ = frame.shape

            # Set JPEG encoding parameters (90% quality)
            encode_param = [
                int(cv2.IMWRITE_JPEG_QUALITY),
                90,
            ]
            # Encode frame to JPEG format
            _, jpeg_image = cv2.imencode(".jpg", frame, encode_param)

            # Store the encoded frame with metadata
            self.latest_frame = Frame(
                width=width,
                height=height,
                format=ImageFormat.JPEG,
                data=jpeg_image,
                field_of_view_w=self.field_of_view_w,
                field_of_view_h=self.field_of_view_h,
            )

    def change_exposure_time(self, exposure_time: int):
        """
        Change the camera's exposure time setting.

        Currently not implemented for OpenCV cameras. This method serves as a
        placeholder for future implementation of exposure control functionality.

        Args:
            exposure_time (int): Desired exposure time value

        Note:
            This method currently does nothing (pass). Implementation would
            depend on the specific camera hardware and OpenCV's camera property
            support for the connected device.
        """
        pass

    def change_gain(self, gain: float):
        """
        Change the camera's gain (sensitivity) setting.

        Currently not implemented for OpenCV cameras. This method serves as a
        placeholder for future implementation of gain control functionality.

        Args:
            gain (float): Desired gain value

        Note:
            This method currently does nothing (pass). Implementation would
            depend on the specific camera hardware and OpenCV's camera property
            support for the connected device.
        """
        pass
