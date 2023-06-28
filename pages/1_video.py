"""Object detection demo with MobileNet SSD.
This model and code are based on
https://github.com/robmarkcole/object-detection-app
"""

import logging
import queue
from pathlib import Path
from typing import List, NamedTuple

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

from utils import get_ice_servers

logger = logging.getLogger(__name__)

MODEL_URL = None
MODEL_LOCAL_PATH = "model_xgb_alphabet.xgb"
PROTOTXT_URL = None
PROTOTXT_LOCAL_PATH = None

class Detection(NamedTuple):
    class_id: int
    label: str
    score: float
    box: np.ndarray

# NOTE: The callback will be called in another thread,
#       so use a queue here for thread-safety to pass the data
#       from inside to outside the callback.
# TODO: A general-purpose shared state object may be more useful.
result_queue: "queue.Queue[List[Detection]]" = queue.Queue()


def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    # image = frame.to_ndarray(format="bgr24")

    # Render bounding boxes and captions

    # result_queue.put(detections)

    return av.VideoFrame.from_ndarray(frame)


webrtc_ctx = webrtc_streamer(
    key="object-detection",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={"iceServers": get_ice_servers()},
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# if st.checkbox("Show the detected labels", value=True):
#     if webrtc_ctx.state.playing:
#         labels_placeholder = st.empty()
#         # NOTE: The video transformation with object detection and
#         # this loop displaying the result labels are running
#         # in different threads asynchronously.
#         # Then the rendered video frames and the labels displayed here
#         # are not strictly synchronized.
#         while True:
#             result = result_queue.get()
#             labels_placeholder.table(result)
