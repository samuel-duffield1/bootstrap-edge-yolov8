from enum import Enum

import cv2


class ImageFormat(Enum):
    """
    Enumeration defining supported image formats for frame data.

    This enum specifies the different formats that image data can be stored in
    within the Frame class. Each format represents a different encoding or
    compression method for the image data.
    """

    NONE = 0  # No specific format or uninitialized
    RAW = 1  # Raw uncompressed image data
    JPEG = 2  # JPEG compressed image format


class Frame:
    """
    Represents a single image frame with metadata for computer vision processing.

    This class encapsulates image data along with essential metadata including
    dimensions, format, and field of view information. It's designed to work
    with camera systems in YOLO object detection pipelines, providing a
    standardized way to handle image data throughout the processing chain.
    """

    def __init__(
        self,
        width: int,
        height: int,
        format: ImageFormat,
        data: bytes,
        field_of_view_w: int,
        field_of_view_h: int,
    ):
        """
        Initialize a Frame with image data and metadata.

        Args:
            width (int): Width of the image in pixels
            height (int): Height of the image in pixels
            format (ImageFormat): Format/encoding of the image data (RAW, JPEG, etc.)
            data (bytes): Raw image data in the specified format
            field_of_view_w (int): Horizontal field of view in degrees
            field_of_view_h (int): Vertical field of view in degrees
        """
        # Image dimensions in pixels
        self.width = width
        self.height = height

        # Image format and data
        self.format = format
        self.data = data

        # Camera field of view information in degrees
        self.field_of_view_w = field_of_view_w
        self.field_of_view_h = field_of_view_h

    def copy(self):
        """
        Create a deep copy of the Frame object.

        This method creates a new Frame instance with the same metadata
        and a copy of the image data. Useful for creating independent
        copies that can be modified without affecting the original frame.

        Returns:
            Frame: A new Frame object with copied data and metadata
        """
        # Create new frame with same metadata but empty data list
        copied_frame = Frame(
            self.width,
            self.height,
            self.format,
            [],
            self.field_of_view_w,
            self.field_of_view_h,
        )
        # Copy the image data to the new frame
        copied_frame.data[:] = self.data
        return copied_frame


def encode(frame, fov_w: int, fov_h: int) -> Frame:
    """
    Encode an OpenCV frame to JPEG format and wrap it in a Frame object.

    This function takes a raw OpenCV image array and converts it to JPEG format
    with a specified quality level (90%). The encoded image is then wrapped in
    a Frame object with the provided field of view information.

    Args:
        frame: OpenCV image array (numpy array) with shape (height, width, channels)
        fov_w (int): Horizontal field of view in degrees
        fov_h (int): Vertical field of view in degrees

    Returns:
        Frame: A Frame object containing the JPEG-encoded image data and metadata

    Note:
        The JPEG quality is set to 90% for a good balance between file size
        and image quality. The input frame should be a standard OpenCV BGR image.
    """
    # Extract dimensions from the OpenCV frame
    height, width, _ = frame.shape

    # Set JPEG encoding parameters (90% quality)
    encode_param = [
        int(cv2.IMWRITE_JPEG_QUALITY),
        90,
    ]

    # Encode the frame to JPEG format
    _, jpeg_image = cv2.imencode(".jpg", frame, encode_param)

    # Create and return Frame object with encoded data
    encoded_frame = Frame(
        width=width,
        height=height,
        format=ImageFormat.JPEG,
        data=jpeg_image,
        field_of_view_w=fov_w,
        field_of_view_h=fov_h,
    )
    return encoded_frame
