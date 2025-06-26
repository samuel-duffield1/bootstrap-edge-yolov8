import logging
import threading
from queue import Queue

import cv2
import flask
from loguru import logger

from edge_yolo_demo.cameras.image import Frame, ImageFormat
from edge_yolo_demo.displays.display import VideoDisplay

# MJPEG streaming protocol constants
PREFIX = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"  # MJPEG frame boundary prefix
SUFFIX = b"\r\n"  # MJPEG frame boundary suffix

# Minimal HTML page for displaying the video stream
INDEX = """\
<!doctype html>
<html lang="en">
<head>
  <link rel="icon" href="data:;base64,iVBORw0KGgo=">
</head>
<body><img src="/stream" /></body>
</html>
"""


class WebDisplay(VideoDisplay):
    """
    Web-based video display using Flask and MJPEG streaming.

    This class provides a web interface for displaying video frames in real-time
    through a web browser. It implements the VideoDisplay interface and serves
    frames using the MJPEG (Motion JPEG) streaming protocol over HTTP.

    The display runs a Flask web server in a background thread and maintains
    a queue of frames to stream to connected clients. Multiple clients can
    connect simultaneously to view the same stream.

    Features:
    - Real-time MJPEG streaming over HTTP
    - Non-blocking frame queuing with automatic overflow handling
    - Supports both JPEG and RAW image formats
    - Automatic JPEG encoding for RAW frames
    - Multi-client support
    """

    def __init__(self, port: int = 8787):
        """
        Initialize the web display server.

        Creates a Flask web application, sets up URL routes, and starts the
        web server in a background thread. The server will be accessible
        at http://localhost:<port>/ once initialized.

        Args:
            port (int, optional): Port number for the web server. Defaults to 8787.
        """
        # Server configuration
        self.port = port

        # Frame queue for streaming (max 100 frames to prevent memory issues)
        self.image_queue = Queue(100)

        # Flask application setup
        self.app = flask.Flask("model-output-cam")
        self.app.add_url_rule("/", view_func=self._index)
        self.app.add_url_rule("/stream", view_func=self._stream)

        # Disable Flask's default logging to reduce console noise
        log = logging.getLogger("werkzeug")
        log.disabled = True

        # Start the web server in a background daemon thread
        self.run_thread = threading.Thread(
            target=self._run_thread, name="Flask Web Display Thread", daemon=True
        )
        self.run_thread.start()

    def send_image(self, image: Frame):
        """
        Queue a frame for web streaming.

        Adds a frame to the internal queue for streaming to web clients.
        Uses non-blocking queue insertion to prevent the calling thread
        from being blocked if the queue is full.

        Args:
            image (Frame): The frame to be streamed to web clients

        Note:
            If the queue is full (100 frames), the frame will be dropped
            and a debug message will be logged. This prevents memory
            buildup when clients are slow or disconnected.
        """
        try:
            # Non-blocking queue insertion to prevent blocking the caller
            self.image_queue.put(image, block=False)
        except:
            logger.debug("Unable to queue image, looks as if nobody is connected.")

    def _run_thread(self):
        """
        Run the Flask web server in a background thread.

        Starts the Flask application server with the following configuration:
        - Host: 0.0.0.0 (accessible from any network interface)
        - Threaded: True (allows multiple concurrent requests)
        - Reloader: False (prevents auto-restarting in production)
        """
        self.app.run(host="0.0.0.0", port=self.port, threaded=True, use_reloader=False)

    def _index(self):
        """
        Serve the main HTML page.

        Returns the simple HTML page that displays the video stream.
        This page contains a single image element that sources from /stream.

        Returns:
            str: HTML content for the main page
        """
        return INDEX

    def _stream(self):
        """
        Serve the MJPEG video stream.

        Creates a Flask Response object that streams MJPEG data to the client.
        The stream uses the multipart/x-mixed-replace content type which
        allows continuous frame updates in the browser.

        Returns:
            flask.Response: Streaming response with MJPEG data
        """
        return flask.Response(
            self._get_frame(), mimetype="multipart/x-mixed-replace; boundary=frame"
        )

    def _get_frame(self):
        """
        Generate MJPEG frame data for streaming.

        This generator function continuously yields MJPEG frame data from the
        image queue. It handles both JPEG and RAW image formats, automatically
        encoding RAW frames to JPEG for web compatibility.

        The function formats each frame with the proper MJPEG boundaries
        required for the multipart/x-mixed-replace streaming protocol.

        Yields:
            bytes: MJPEG frame data including boundaries and headers

        Note:
            This is a generator function that runs indefinitely, yielding
            frames as they become available in the queue. Each frame is
            formatted with MJPEG protocol boundaries.
        """
        while True:
            # Get the next frame from the queue (blocks until available)
            frame = self.image_queue.get()
            data = None

            if frame.format == ImageFormat.JPEG:
                # Frame is already in JPEG format, use directly
                data = frame.data
            elif frame.format == ImageFormat.RAW:
                # Convert RAW frame to JPEG format
                encode_param = [
                    int(cv2.IMWRITE_JPEG_QUALITY),
                    90,
                ]
                _, jpeg_frame = cv2.imencode(".jpg", frame, encode_param)
                data = jpeg_frame

            # Yield frame with MJPEG boundaries for streaming
            yield b"".join([PREFIX, bytes(data), SUFFIX])
