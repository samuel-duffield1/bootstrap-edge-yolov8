class Trackable:
    """
    Represents a trackable object detected by YOLO model with spatial and classification information.

    This class encapsulates the essential properties of an object detected in a frame,
    including its position, dimensions, field of view context, confidence score,
    and classification details. It's designed to facilitate object tracking and
    analysis in computer vision applications.
    """

    def __init__(
        self,
        x: int,
        y: int,
        frame_pixels_wide: int,
        frame_pixels_high: int,
        field_of_view_w: int,
        field_of_view_h: int,
        confidence: float = 0.0,
        class_name: str = "unknown",
        unique_name: str = "none",
    ) -> None:
        """
        Initialize a Trackable object with position, frame, and classification data.

        Args:
            x (int): X-coordinate of the object's center in pixels
            y (int): Y-coordinate of the object's center in pixels
            frame_pixels_wide (int): Width of the detection frame in pixels
            frame_pixels_high (int): Height of the detection frame in pixels
            field_of_view_w (int): Width of the camera's field of view in degrees
            field_of_view_h (int): Height of the camera's field of view in degrees
            confidence (float, optional): Model confidence score for the detection (0.0-1.0). Defaults to 0.0.
            class_name (str, optional): Classification label for the detected object. Defaults to "unknown".
            unique_name (str, optional): Unique identifier for tracking purposes. Defaults to "none".
        """

        # Object position coordinates in the frame
        self.x = x
        self.y = y

        # Model confidence in the detection (0.0 to 1.0)
        self.confidence = confidence

        # Frame dimensions in pixels
        self.pixels_wide = frame_pixels_wide
        self.pixels_high = frame_pixels_high

        # Camera field of view dimensions in degrees
        self.field_of_view_w = field_of_view_w
        self.field_of_view_h = field_of_view_h

        # Object classification and identification
        self.class_name = class_name
        self.unique_name = unique_name

    def to_dict(self) -> dict:
        """
        Convert the Trackable object to a dictionary representation.

        This method serializes all object properties into a dictionary format,
        which is useful for JSON serialization, logging, or data export purposes.

        Returns:
            dict: Dictionary containing all trackable object properties with their current values
        """
        # Create dictionary with all object properties
        trackable_data = {
            "x": self.x,
            "y": self.y,
            "confidence": self.confidence,
            "pixels_wide": self.pixels_wide,
            "pixels_high": self.pixels_high,
            "field_of_view_w": self.field_of_view_w,
            "field_of_view_h": self.field_of_view_h,
            "class_name": self.class_name,
            "unique_name": self.unique_name,
        }
        return trackable_data
