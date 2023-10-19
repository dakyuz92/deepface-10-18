import cv2
import os
import logging

# logging.basicConfig(level=logging.DEBUG)

from skimage import metrics as skimage_metrics


def is_similar_frame(f1, f2, resize_to=(64, 64), thresh=0.5, return_score=False):
    thresh = float(os.getenv("FRAME_SIMILARITY_THRESH", thresh))

    if f1 is None or f2 is None:
        return False

    if isinstance(f1, str) and os.path.exists(f1):
        try:
            f1 = cv2.imread(f1)
        except Exception as ex:
            logging.exception(ex, exc_info=True)
            return False

    if isinstance(f2, str) and os.path.exists(f2):
        try:
            f2 = cv2.imread(f2)
        except Exception as ex:
            logging.exception(ex, exc_info=True)
            return False

    if resize_to:
        f1 = cv2.resize(f1, resize_to)
        f2 = cv2.resize(f2, resize_to)

    if len(f1.shape) == 3:
        f1 = f1[:, :, 0]

    if len(f2.shape) == 3:
        f2 = f2[:, :, 0]

    score = skimage_metrics.structural_similarity(f1, f2, multichannel=False)

    if return_score:
        return score

    if score >= thresh:
        return True

    return False


def get_interest_frames_from_video(
    video_path,
    frame_similarity_threshold=0.5,
    similarity_context_n_frames=3,
    skip_n_frames=0.5,
    out