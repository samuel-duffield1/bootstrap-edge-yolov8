from abc import ABC, abstractmethod

from edge_yolo_demo.cameras.image import Frame


class CameraException(Exception):
    """
    Custom exception class for camera-related errors.

    This exception is raised when camera operations fail, such as connection issues,
    hardware errors, or invalid parameter settings. It inherits from the base Exception
    class and provides a way to handle camera-specific errors in the application.
    """

    def __init__(self, *args: object) -> None:
        """
        Initialize the CameraException with optional error messages.

        Args:
            *args: Variable number of arguments to pass to the base Exception class.
                   Typically includes error messages or error codes.
        """
        super().__init__(*args)


class ICamera(ABC):
    """
    Abstract base class defining the interface for camera implementations.

    This interface provides a standardized way to interact with different types of cameras
    in the YOLO detection system. It defines the essential methods that all camera
    implementations must provide, including frame capture, field of view configuration,
    and camera parameter adjustments.

    All concrete camera classes should inherit from this interface and implement
    the abstract methods according to their specific hardware requirements.
    """

    def __init__(self, field_of_view_h: int = 90, field_of_view_w: int = 90) -> None:
        """
        Initialize the camera interface with field of view parameters.

        Args:
            field_of_view_h (int, optional): Vertical field of view in degrees. Defaults to 90.
            field_of_view_w (int, optional): Horizontal field of view in degrees. Defaults to 90.
        """
        # Camera field of view dimensions in degrees
        self.field_of_view_h = field_of_view_h
        self.field_of_view_w = field_of_view_w

    def get_field_of_view(self) -> tuple[int, int]:
        """
        Get the camera's field of view dimensions.

        Returns:
            tuple[int, int]: A tuple containing (width, height) field of view in degrees.
                           The first element is the horizontal FOV, the second is the vertical FOV.
        """
        return self.field_of_view_w, self.field_of_view_h

    @abstractmethod
    def get_frame(self) -> Frame:
        """
        Capture and return a single frame from the camera.

        This method must be implemented by concrete camera classes to provide
        frame capture functionality specific to their hardware.

        Returns:
            Frame: A Frame object containing the captured image data and metadata.

        Raises:
            CameraException: If frame capture fails due to hardware or connection issues.
            NotImplementedError: If called on the abstract base class directly.
        """
        raise NotImplementedError

    @abstractmethod
    def change_exposure_time(self, exposure_time: int):
        """
        Change the camera's exposure time setting.

        This method adjusts how long the camera sensor is exposed to light for each frame.
        Longer exposure times can improve image quality in low-light conditions but may
        introduce motion blur.

        Args:
            exposure_time (int): Exposure time value. Units and valid range depend on
                               the specific camera implementation.

        Raises:
            CameraException: If the exposure time cannot be set or is out of valid range.
            NotImplementedError: If called on the abstract base class directly.
        """
        raise NotImplementedError

    @abstractmethod
    def change_gain(self, gain: float):
        """
        Change the camera's gain (sensitivity) setting.

        This method adjusts the amplification applied to the camera sensor's signal.
        Higher gain values can improve visibility in low-light conditions but may
        introduce noise in the image.

        Args:
            gain (float): Gain value to set. Valid range depends on the specific
                         camera implementation (e.g., 1.0-4.0 for some cameras).

        Raises:
            CameraException: If the gain cannot be set or is out of valid range.
            NotImplementedError: If called on the abstract base class directly.
        """
        raise NotImplementedError
