�ǥ.)�z�Z�YI���OW����ͺ�u�.H6�A�v��Np�l�
p��I�~Պ�Փ7+��VlK��pM�1xa�/^��+g��:�E����Vu/Y���v�&��P1�Y�l��hD�+X�1g��IW�vi]k!�^��lE�_�Ԥ���@Al�����سa��I_yx�]Պ��J�D�@�E צ�	p5B${XĖT��\�`X�S�)�EB���a��_�Q�!��eB�� �Ww�K�Q֮UTթ��e]%�@��rL�.��P����_Rڕ�J��pA���}T�����P��%����tQ��9�!�Ag�`Ae�!�[����1����C���Q$���X��d^���W�Y�Y��A�y���=��\Q��q`��S�$�l�a>I��5�C�
ht��}�Pm2�%��,��jY���-"҉��-���߉�Y���D�:Ac&�(N�l#b!bc�>ۙ�)�­L�K���h�}@F-n�ǖ�
�y�
�_.u��8H,��'(��-�'\�2B�"*��[۹ޜ�Z5��%�� �WuT|#�u��I��b9z��\�J=œ+lBӘI�pD�DD���~�cDƖ��,�F��) �t���VA��Yy�!f�B�_���5��D�EJ���^6r�G��[��ZGb�ƍ�(���h�k���I�3Y�T�{t�L&��	Md��q�+�@X�	��*�P��q��?N1$d�1#3�������M]�i���� ��Wu�Y�ch���H�&�ϑm�]qZ,aO:��fԋM�\�I�ET��ϺUΉ)�[�}B+��3��R>M��_B��dVf�mH#:"fNbf��}-�b$h�I�Wj��%j�%{��Sa���I�a�-�c��L!DD^���T�}���i gHDM+$te��q�rF�G�0�A:e�I&Cj� ��C��'�'eM#��W>`�TR"}��&����m�ԅX§�Fy�'�է1x�C�a&p~PC�@�L�JDP_
'���9�+ ;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             import os
#import cv2
#import pydload
import logging
import numpy as np
import onnxruntime
#from progressbar import progressbar

from .detector_utils import preprocess_image
from .video_utils import get_interest_frames_from_video


def dummy(x):
    return x


FILE_URLS = {
    "default": {
        "checkpoint": "https://github.com/notAI-tech/NudeNet/releases/download/v0/detector_v2_default_checkpoint.onnx",
        "classes": "https://github.com/notAI-tech/NudeNet/releases/download/v0/detector_v2_default_classes",
    },
    "base": {
        "checkpoint": "https://github.com/notAI-tech/NudeNet/releases/download/v0/detector_v2_base_checkpoint.onnx",
        "classes": "https://github.com/notAI-tech/NudeNet/releases/download/v0/detector_v2_base_classes",
    },
}


class Detector:
    detection_model = None
    classes = None

    def __init__(self, model_name="default"):
        """
        model = Detector()
        """
        checkpoint_url = FILE_URLS[model_name]["checkpoint"]
        classes_url = FILE_URLS[model_name]["classes"]

        home = os.path.expanduser("~")
        model_folder = os.path.join(home, f".NudeNet/")
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)

        checkpoint_name = os.path.basename(checkpoint_url)
        checkpoint_path = os.path.join(model_folder, checkpoint_name)
        classes_path = os.path.join(model_folder, "classes")

        if not os.path.exists(checkpoint_path):
            print("Downloading the checkpoint to", checkpoint_path)
           # pydload.dload(checkpoint_url, save_to_path=checkpoint_path, max_time=None)

        if not os.path.exists(classes_path):
            print("Downloading the classes list to", classes_path)
            #pydload.dload(classes_url, save_to_path=classes_path, max_time=None)

        self.detection_model = onnxruntime.InferenceSession(checkpoint_path)

        self.classes = [c.strip() for c in open(classes_path).readlines() if c.strip()]

    def detect_video(
        self, video_path, mode="default", min_prob=0.6, batch_size=2, show_progress=True
    ):
        frame_indices, frames, fps, video_length = get_interest_frames_from_video(
            video_path
        )
        logging.debug(
            f"VIDEO_PATH: {video_path}, FPS: {fps}, Important frame indices: {frame_indices}, Video length: {video_length}"
        )
        if mode == "fast":
            frames = [
                preprocess_image(frame, min_side=480, max_side=800) for frame in frames
            ]
        else:
            frames = [preprocess_image(frame) for frame in frames]

        scale = frames[0][1]
        frames = [frame[0] for frame in frames]
        all_results = {
            "metadata": {
                "fps": fps,
                "video_length": video_length,
                "video_path": video_path,
            },
            "preds": {},
        }

        #progress_func = progressbar

        if not show_progress:
            progress_func = dummy

        for _ in progress_