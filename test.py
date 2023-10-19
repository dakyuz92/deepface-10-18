import sys

import cv2
import numpy as np
from PIL.Image import Image
import torch
import deepface
from Lib import os
from deepface import DeepFace

# ------------------------------------------------------------------------------------------------------------------------------------------------
# Face detection using UltraFace-640 onnx model
face_detector_onnx = "../ultraface/models/version-RFB-640.onnx"


# Start from ORT 1.10, ORT requires explicitly setting the providers parameter if you want to use execution providers
# other than the default CPU provider (as opposed to the previous behavior of providers getting set/registered by default
# based on the build flags) when instantiating InferenceSession.
# For example, if NVIDIA GPU is available and ORT Python package is built with CUDA, then call API as following:
# ort.InferenceSession(path/to/model, providers=['CUDAExecutionProvider'])
# face_detector = ort.InferenceSession(face_detector_onnx)


# scale current rectangle to box
def scale(box):
    width = box[2] - box[0]
    height = box[3] - box[1]
    maximum = max(width, height)
    dx = int((maximum - width) / 2)
    dy = int((maximum - height) / 2)

    bboxes = [box[0] - dx, box[1] - dy, box[2] + dx, box[3] + dy]
    return bboxes


# crop image
def cropImage(image, box):
    num = image[box[1]:box[3], box[0]:box[2]]
    return num


# face detection method


# ------------------------------------------------------------------------------------------------------------------------------------------------
# Face gender classification using GoogleNet onnx model
gender_classifier_onnx = "models/gender_googlenet.onnx"

# Start from ORT 1.10, ORT requires explicitly setting the providers parameter if you want to use execution providers
# other than the default CPU provider (as opposed to the previous behavior of providers getting set/registered by default
# based on the build flags) when instantiating InferenceSession.
# For example, if NVIDIA GPU is available and ORT Python package is built with CUDA, then call API as following:
# ort.InferenceSession(path/to/model, providers=['CUDAExecutionProvider'])


# gender classification method
# ------------------------------------------------------------------------------------------------------------------------------------------------
# Face age classification using GoogleNet onnx model
age_classifier_onnx = "models/age_googlenet.onnx"

# Start from ORT 1.10, ORT requires explicitly setting the providers parameter if you want to use execution providers
# other than the default CPU provider (as opposed to the previous behavior of providers getting set/registered by default
# based on the build flags) when instantiating InferenceSession.
# For example, if NVIDIA GPU is available and ORT Python package is built with CUDA, then call API as following:
# ort.InferenceSession(path/to/model, providers=['CUDAExecutionProvider'])
# age_classifier = ort.InferenceSession(age_classifier_onnx)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

# age classification method
import pickle
from pathlib import Path
# sys.path.append('..')
#import piexif



if __name__ == '__main__':
    print(torch.cuda_path)
    print(torch.cuda_version)
    print(torch.cuda.get_device_name(0))
    entries = os.listdir(
        'C:/Users/Dylan/Videos/DeepFaceLab/DeepFaceLab_NVIDIA_RTX3000_series_build_11_20_2021/DeepFaceLab_NVIDIA_RTX3000_series/workspace/data_dst/aligned/')
    for entry in entries:
        img_path = (
                'C:/Users/Dylan/Videos/DeepFaceLab/DeepFaceLab_NVIDIA_RTX3000_series_build_11_20_2021/DeepFaceLab_NVIDIA_RTX3000_series/workspace/data_dst/aligned/' + str(entry))
      #  try    # img_path = 'F:\\tscomps\\' + str(entry) + "\\" + str(ent)
        try:
            obj = deepface.DeepFace.analyze(img_path)
            file1 = open("C:\\Users\\Dylan\\Desktop\\teams5.txt", "a")  # append mode
            #print(obj[0]['age'])
            if obj.__len__() == 0:
                tags = {"NoFace"}
            elif obj.__len__() > 1:
                tags = [img_path,"MultiFace"]
                file1.write(img_path + "," + "MultiFace" + "\n")
            else:
                tags = [img_path,obj[0]['age'], obj[0]['gender'], obj[0]['race']]
                file1.write(img_path + "," + tags[1]+","+tags[2]+","+tags[3] + "\n")

                print(str(tags))
                file1.close()
        except:
            file1 = open("C:\\Users\\Dylan\\Desktop\\teams5.txt", "a")  # append mode
            file1.write(img_path + "," + "NoFace" + "\n")
            file1.close()

        # exif_ifd = {piexif.ExifIFD.MakerNote: data}
        # exif_dict = {"0th": {}, "Exif": exif_ifd, "1st": {}, "thumbnail": None, "GPS": {}}
        #img = Image.new('RGB', (500, 500), 'green')
        #  exif_dat = piexif.dump(exif_dict)
        #img.save('image.jpg', exif=exif_dat)
        #img = Image.open('image.jpg')
        # raw = img.getexif()[piexif.ExifIFD.MakerNote]
        #tags = pickle.loads(raw)
       # except:
        #    print("EXCEPT")

            # file1 = open("C:\\Users\\Dylan\\Desktop\\teams.txt", "a")  # append mode
            # file1.write(str(ent)+","+str(obj) + "\n")
            # file1.close()
            # file1 = open("C:\\Users\\Dylan\\Desktop\\teams.txt", "a")  # append mode
            # file1.write(str(ent)+","+"{}" + "\n")
            # ile1.close()
def man():

    from PIL import Image

with open("C:\\Users\\Dylan\\Desktop\\teams.txt") as f:
    line = f.readline()
    while line:
        print(str(line))
        if str(line).__contains__("\'dominant_gender\': \'Man\'"):
            src_file = Path(
                'C:/Users/Dylan/Videos/DeepFaceLab/DeepFaceLab_NVIDIA_RTX3000_series_build_11_20_2021/DeepFaceLab_NVIDIA_RTX3000_series/workspace/data_dst/aligned/' +
                line.split(",")[0])
            dst_file = Path(
                'C:/Users/Dylan/Videos/DeepFaceLab/DeepFaceLab_NVIDIA_RTX3000_series_build_11_20_2021/DeepFaceLab_NVIDIA_RTX3000_series/workspace/data_dst/M/' +
                line.split(",")[0])
            src_file.rename(dst_file)
        line = f.readline()



def movePath():
    entries = os.listdir(
        'C:/Users/Dylan/Videos/DeepFaceLab/DeepFaceLab_NVIDIA_RTX3000_series_build_11_20_2021/DeepFaceLab_NVIDIA_RTX3000_series/workspace/data_dst/aligned/')
    for entry in entries:
        img_path = (
                    'C:/Users/Dylan/Videos/DeepFaceLab/DeepFaceLab_NVIDIA_RTX3000_series_build_11_20_2021/DeepFaceLab_NVIDIA_RTX3000_series/workspace/data_dst/aligned/' + str(entry))
        try:
            obj = DeepFace.analyze(img_path)
            print(str(obj))
            # s= str(obj).split("\'Woman\': ")[1].split(",")[0]
            # print(str(int))

            file1 = open("C:\\Users\\Dylan\\Desktop\\teams.txt", "a")  # append mode
            file1.write(str(entry) + "," + str(obj) + "\n")
            file1.close()
        except:
            file1 = open("C:\\Users\\Dylan\\Desktop\\teams.txt", "a")  # append mode
            file1.write(str(entry) + "," + "{}" + "\n")
            file1.close()

    # for a in range(47):
    # ageClassifier("C:\\Users\\Dylan\\Pictures\\aaPreviews\\43.jpg")
# print(str(obj))
# ------------------------------------------------------------------------------------------------------------------------------------------------
# Main void
