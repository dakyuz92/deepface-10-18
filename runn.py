import os

import json
import onnxruntime as ort
from deepface import DeepFace

import numpy as np

import age_gender
import emotion_ferplus.src.ferplus
import setup
from nudenet import NudeDetector

import cv2
import age_gender.levi_googlenet as ss
import age_gender.rothe_vgg as ss2

face_detector_onnx = "C:\\Users\\Dylan\\.NudeNet/version-RFB-640.onnx"
face_detector = ort.InferenceSession(face_detector_onnx)

age_classifier_onnx = "C:\\Users\\Dylan\\.NudeNet/age_googlenet.onnx"

# Start from ORT 1.10, ORT requires explicitly setting the providers parameter if you want to use execution providers
# other than the default CPU provider (as opposed to the previous behavior of providers getting set/registered by default
# based on the build flags) when instantiating InferenceSession.
# For example, if NVIDIA GPU is available and ORT Python package is built with CUDA, then call API as following:
# ort.InferenceSession(path/to/model, providers=['CUDAExecutionProvider'])
age_classifier = ort.InferenceSession(age_classifier_onnx)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']


def predict(width, height, confidences, boxes, prob_threshold, iou_threshold=0.5, top_k=-1):
    """
    Select boxes that contain human faces
    Args:
        width: original image width
        height: original image height
        confidences (N, 2): confidence array
        boxes (N, 4): boxes array in corner-form
        iou_threshold: intersection over union threshold.
        top_k: keep top_k results. If k <= 0, keep all the results.
    Returns:
        boxes (k, 4): an array of boxes kept
        labels (k): an array of labels for each boxes kept
        probs (k): an array of probabilities for each boxes being in corresponding labels
    """
    boxes = boxes[0]
    confidences = confidences[0]
    # print(boxes)
    # print(confidences)

    picked_box_probs = []
    picked_labels = []
    for class_index in range(1, confidences.shape[1]):
        # print(confidences.shape[1])
        probs = confidences[:, class_index]
        # print(probs)
        mask = probs > prob_threshold
        probs = probs[mask]

        if probs.shape[0] == 0:
            continue
        subset_boxes = boxes[mask, :]
        # print(subset_boxes)
        box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
        box_probs = hard_nms(box_probs,
                             iou_threshold=iou_threshold,
                             top_k=top_k,
                             )
        picked_box_probs.append(box_probs)
        picked_labels.extend([class_index] * box_probs.shape[0])
    if not picked_box_probs:
        return np.array([]), np.array([]), np.array([])
    picked_box_probs = np.concatenate(picked_box_probs)
    picked_box_probs[:, 0] *= width
    picked_box_probs[:, 1] *= height
    picked_box_probs[:, 2] *= width
    picked_box_probs[:, 3] *= height
    return picked_box_probs[:, :4].astype(np.int32), np.array(picked_labels), picked_box_probs[:, 4]


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
    import array as arr

# creating an array with integer type
a = arr.array('i', [1, 2, 3])
    num = image[box[1]:box[3], box[0]:box[2]]
    return num


# face detection method
def faceDetector(orig_image, threshold=0.7):
  #  gray = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
   # image = cv2.resize(gray, (640, 480))
    image_mean = np.array([127, 127, 127])
    #image = (image - image_mean) / 128
 #   image = np.transpose(image, [2, 0, 1])
  #  image = np.expand_dims(image, axis=0)
   # image = image.astype(np.float32)
    image=0
    input_name = face_detector.get_inputs()[0].name
    confidences, boxes = face_detector.run(None, {input_name: image})
    boxes, labels, probs = predict(orig_image.shape[1], orig_image.shape[0], confidences, boxes, threshold)
    return boxes, labels, probs


# ------------------------------------------------------------------------------------------------------------------------------------------------
# Face gender classification using GoogleNet onnx model
gender_classifier_onnx = "C:\\Users\\Dylan\\.NudeNet/gender_googlenet.onnx"

# Start from ORT 1.10, ORT requires explicitly setting the providers parameter if you want to use execution providers
# other than the default CPU provider (as opposed to the previous behavior of providers getting set/registered by default
# based on the build flags) when instantiating InferenceSession.
# For example, if NVIDIA GPU is available and ORT Python package is built with CUDA, then call API as following:
# ort.InferenceSession(path/to/model, providers=['CUDAExecutionProvider'])
gender_classifier = ort.InferenceSession(gender_classifier_onnx)
genderList = ['Male', 'Female']


# gender classification method


# ------------------------------------------------------------------------------------------------------------------------------------------------
# Face age classification using GoogleNet onnx model
age_classifier_onnx = "C:\\Users\\Dylan\\.NudeNet/age_googlenet.onnx"

# Start from ORT 1.10, ORT requires explicitly setting the providers parameter if you want to use execution providers
# other than the default CPU provider (as opposed to the previous behavior of providers getting set/registered by default
# based on the build flags) when instantiating InferenceSession.
# For example, if NVIDIA GPU is available and ORT Python package is built with CUDA, then call API as following:
# ort.InferenceSession(path/to/model, providers=['CUDAExecutionProvider'])
age_classifier = ort.InferenceSession(age_classifier_onnx)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']


def transform(point, center, scale, resolution):
    pt = np.array([point[0], point[1], 1.0])
    h = 200.0 * scale
    m = np.eye(3)
    m[0, 0] = resolution / h
    m[1, 1] = resolution / h
    m[0, 2] = resolution * (-center[0] / h + 0.5)
    m[1, 2] = resolution * (-center[1] / h + 0.5)
    m = np.linalg.inv(m)
    return np.matmul(m, pt)[0:2]


def ageClassifier(orig_image):
  #  image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
   # image = cv2.resize(image, (224, 224))
    image=0
    image_mean = np.array([104, 117, 123])
    image = image - image_mean
    image = np.transpose(image, [2, 0, 1])
    image = np.expand_dims(image, axis=0)
    image = image.astype(np.float32)

    input_name = age_classifier.get_inputs()[0].name
    ages = age_classifier.run(None, {input_name: image})
    age = ageList[ages[0].argmax()]
    return age


def crop2(image, box, scale, resolution=224.0):
    h = np.float32(box[3] - box[1])
    w = np.float32(box[2] - box[0])
    center = np.array([box[0] + w / 2, box[1] + h / 2])
    center = np.float32([box[0] + w / 2, box[1] + h / 2])
    ul = transform([1, 1], center, scale, resolution).astype(int)
    br = transform([resolution, resolution], center, scale, resolution).astype(int)

    if image.ndim > 2:
        newDim = np.array([br[1] - ul[1], br[0] - ul[0], image.shape[2]], dtype=np.int32)
        newImg = np.zeros(newDim, dtype=np.uint8)
    else:
        newDim = np.array([br[1] - ul[1], br[0] - ul[0]], dtype=np.int)
        newImg = np.zeros(newDim, dtype=np.uint8)
    ht = image.shape[0]
    wd = image.shape[1]
    newX = np.array([max(1, -ul[0] + 1), min(br[0], wd) - ul[0]], dtype=np.int32)
    newY = np.array([max(1, -ul[1] + 1), min(br[1], ht) - ul[1]], dtype=np.int32)
    oldX = np.array([max(1, ul[0] + 1), min(br[0], wd)], dtype=np.int32)
    oldY = np.array([max(1, ul[1] + 1), min(br[1], ht)], dtype=np.int32)
    newImg[newY[0] - 1:newY[1], newX[0] - 1:newX[1]] = image[oldY[0] - 1:oldY[1], oldX[0] - 1:oldX[1], :]

   # newImg = cv2.resize(newImg, dsize=(int(resolution), int(resolution)), interpolation=cv2.INTER_LINEAR)
    return newImg


def crop():
    file1 = open("C:\\Users\\Dylan\\Desktop\\classify.txt", "r")  # append mode
    lines = file1.readlines()
    for index, line in enumerate(lines):
        print(line)
        name = line.split(",")[0]
        recs = str(line).split(".jpg,")[1]

        print(str(recs).replace("'", "\""))

        # x=json.JSONEncoder.__str__()
        # print(x)
        iusc = json.loads(str(recs).replace("'", "\""))
        if len(iusc) == 0:
            continue

        for i in range(len(iusc) - 1):
            box = iusc[i]['box']
            score = iusc[i]['score']
            label = iusc[i]['label']
           # image = cv2.imread("C:\\Users\\Dylan\\Pictures\\child\\" + name)
            # image =cv2.imread("D:\\classify\\"+name).astype(np.float32) / 255.0
            image=0
            image2 = crop2(image, box, 3.0, 712.0)
            name2 = name.replace(".jpg", "").replace(".JPG", "").replace(".png", "") + "-" + label + "-" + str(
                int(score * 100))
            #cv2.imwrite("C:\\Users\\Dylan\\Pictures\\genitalia\\new\\" + name2 + ".jpg", image2)




def area_of(left_top, right_bottom):
    """
    Compute the areas of rectangles given two corners.
    Args:
        left_top (N, 2): left top corner.
        right_bottom (N, 2): right bottom corner.
    Returns:
        area (N): return the area.
    """
    hw = np.clip(right_bottom - left_top, 0.0, None)
    return hw[..., 0] * hw[..., 1]


def iou_of(boxes0, boxes1, eps=1e-5):
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Args:
        boxes0 (N, 4): ground truth boxes.
        boxes1 (N or 1, 4): predicted boxes.
        eps: a small number to avoid 0 as denominator.
    Returns:
        iou (N): IoU values.
    """
    overlap_left_top = np.maximum(boxes0[..., :2], boxes1[..., :2])
    overlap_right_bottom = np.minimum(boxes0[..., 2:], boxes1[..., 2:])

    overlap_area = area_of(overlap_left_top, overlap_right_bottom)
    area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
    area1 = area_of(boxes1[..., :2], boxes1[..., 2:])
    return overlap_area / (area0 + area1 - overlap_area + eps)


def hard_nms(box_scores, iou_threshold, top_k=-1, candidate_size=200):
    """
    Perform hard non-maximum-supression to filter out boxes with iou greater
    than threshold
    Args:
        box_scores (N, 5): boxes in corner-form and probabilities.
        iou_threshold: intersection over union threshold.
        top_k: keep top_k results. If k <= 0, keep all the results.
        candidate_size: only consider the candidates with the highest scores.
    Returns:
        picked: a list of indexes of the kept boxes
    """
    scores = box_scores[:, -1]
    boxes = box_scores[:, :-1]
    picked = []
    indexes = np.argsort(scores)
    indexes = indexes[-candidate_size:]
    while len(indexes) > 0:
        current = indexes[-1]
        picked.append(current)
        if 0 < top_k == len(picked) or len(indexes) == 1:
            break
        current_box = boxes[current, :]
        indexes = indexes[:-1]
        rest_boxes = boxes[indexes, :]
        iou = iou_of(
            rest_boxes,
            np.expand_dims(current_box, axis=0),
        )
        indexes = indexes[iou <= iou_threshold]

    return box_scores[picked, :]


def annotate():
    file1 = open("C:\\Users\\Dylan\\Desktop\\classify.txt", "r")  # append mode
    lines = file1.readlines()
    for index, line in enumerate(lines):
        print(line)
        name = line.split(",")[0]
        recs = str(line).split(".jpg,")[1]

        print(str(recs).replace("'", "\""))

        # x=json.JSONEncoder.__str__()
        # print(x)
        iusc = json.loads(str(recs).replace("'", "\""))
        if len(iusc) == 0:
            continue

        for i in range(len(iusc) - 1):
            try:
                box = iusc[i]['box']
                score = iusc[i]['score']
                label = iusc[i]['label']
                if label == "FACE_F" or label == "FACE_M":
                    # image = cv2.imread( "C:\\Users\\Dylan\\anaconda3\\envs\\ImageAI\\classify\\"+name)
                    # image =cv2.imread("D:\\classify\\"+name).astype(np.float32) / 255.0
                    # image2=crop2(image,box,1.0,712.0)
                    name2 = name.replace(".jpg", "").replace(".JPG", "").replace(".png", "") + "-" + label + "-" + str(
                        int(score * 100))
                    cv2.imwrite("C:\\Users\\Dylan\\anaconda3\\envs\\ImageAI\\facemodel\\train\\images\\"+str(i)+"_"+name, image2)
                    file2 = open(
                        "C:\\Users\\Dylan\\anaconda3\\envs\\ImageAI\\facemodel\\validation\\annotations\\" + name.replace(
                            ".jpg", "").replace(".JPG", "").replace(".png", "") + ".txt", "a")  # append mode
                    if label == "FACE_F":
                        file2.write(str(0) + " 0 0 0 0\r")
                    if label == "FACE_M":
                        file2.write(str(1) + "0 0 0 0\r")
                    file2.close()
            except:
                print("EXCEPT")


def resize():
    file1 = open("C:\\Users\\Dylan\\Pictures\\Genitalia", "r")  # append mode
    lines = file1.readlines()
    for index, line in enumerate(lines):
        print(line)
        name = line.split(",")[0]
        recs = str(line).split(".jpg,")[1]

        print(str(recs).replace("'", "\""))

        # x=json.JSONEncoder.__str__()
        # print(x)
        iusc = json.loads(str(recs).replace("'", "\""))
        if len(iusc) == 0:
            continue

        for i in range(len(iusc) - 1):
            box = iusc[i]['box']
            score = iusc[i]['score']
            label = iusc[i]['label']
            #image = cv2.imread("C:\\Users\\Dylan\\Pictures\\child\\" + name)
            # image =cv2.imread("D:\\classify\\"+name).astype(np.float32) / 255.0
            #image2 = crop2(image, box, 3.0, 712.0)
            name2 = name.replace(".jpg", "").replace(".JPG", "").replace(".png", "") + "-" + label + "-" + str(
                int(score * 100))
            #cv2.imwrite("C:\\Users\\Dylan\\Pictures\\genitalia\\new\\" + name2 + ".jpg", image2)
            entries = os.listdir('C:\\Users\\Dylan\\anaconda3\\envs\\ImageAI\\facemodel\\validation\\annotations')


# /

import numpy as np
from PIL import Image
def postprocess(scores):
    '''
    This function takes the scores generated by the network and returns the class IDs in decreasing
    order of probability.
    '''
    #prob = softmax(scores)
    prob = np.squeeze(scores)
    classes = np.argsort(prob)[::-1]
    return classes


emotion_classifier = ort.InferenceSession("C:\\Users\\Dylan\\anaconda3\\envs\\deepface\\emotion_ferplus\\model\\emotion-ferplus-8.onnx")
emotin_header = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt']

def preprocess(imag):
    input_shape = (1, 1, 64, 64)
    img = Image.open(imag)
    img = img.resize((64, 64), Image.ANTIALIAS)
    img_data = np.array(img)
    img_data = np.resize(img_data, input_shape)
    input_name = emotion_classifier.get_inputs()[0].name
    ages = emotion_classifier.run(None, {input_name: img_data})
    return ages

if __name__ == '__main__':
    from deepface import DeepFace

    """
   for entry in entries:
   
      file1 = open("C:\\Users\\Dylan\\anaconda3\\envs\\ImageAI\\facemodel\\validation\\annotations\\"+str(entry),"r")#append mode
      lines = file1.readlines()
      file1.close()
      for index, line in enumerate(lines):
          file2 = open("C:\\Users\\Dylan\\anaconda3\\envs\\ImageAI\\facemodel\\validation\\annotations\\"+str(entry),"w")#append mode
          file2.write(str(line)+" 0"+" 0"+" 0"+" 0"+"\r")
          file2.close()
   annotate()
   """
detector = NudeDetector()  # detector = NudeDetector('base') for the "base" version of detector.
entries = os.listdir('D:\\New folder\\')
for entry in entries:
    for ent in os.listdir('D:\\New folder\\' + str(entry)):
        img_path = 'D:\\New folder\\' + str(entry) + "\\" + str(ent)
        x = detector.detect(img_path)
        s = str(ent) + "," + str(x)
        print(s)


        color = (255, 128, 0)
        orig_image = cv2.imread(img_path)
        boxes, labels, probs = ss.faceDetector(orig_image)
        processed_boxes = []
        try:
         for i in range(boxes.shape[0]):
          box = scale(boxes[i, :])
          box2 = boxes[i].astype(int).tolist()
          cropped = cropImage(orig_image, box)
          cv2.imwrite("C:\\Users\\Dylan\\Pictures\\aaPreviews\\faces\\" + str(ent).replace(".jpg", "").replace(".JPG", "").replace(".png", "") + "-"+str(i) + ".jpg", cropped)
          gender = ss.genderClassifier(cropped)
          age = ss.ageClassifier(cropped)
          gender2 = ss2.genderClassifier(cropped)
          age2 = ss2.ageClassifier(cropped)
         #x=emotion_ferplus.src.ferplus.FERPlusReader.create("C:\\Users\\Dylan\\Pictures\\aaPreviews\\faces\\","",str(ent).replace(".jpg", "").replace(".JPG", "").replace(".png", "") + "-"+str(i) + ".jpg")
         #print(emotion_classifier(cropped).)
          emotions=preprocess("C:\\Users\\Dylan\\Pictures\\aaPreviews\\faces\\" + str(ent).replace(".jpg", "").replace(".JPG", "").replace(".png", "") + "-"+str(i) + ".jpg")
          print(str(emotions))


         #print(postprocess(scores))
          processed_boxes.append({"box": [int(c) for c in box2], "score": float(probs[i]),"gender1": str(gender),"age1":str(age),"gender2": str(gender2),"age2":str(age2)})
         s = str(s) + "," + str(processed_boxes)
        except:
          print("except")
        try:
          obj = DeepFace.analyze2(img_path)
          s = str(s) + "," + str(obj)
        except:
         print("except")
        file1 = open("C:\\Users\\Dylan\\Desktop\\aio.txt", "a")  # append mode
        file1.write(str(s) + "\n")
        file1.close()
        #color = (255, 128, 0)
       # try:
       # orig_image = cv2.imread(img_path)
        """
        boxes, labels, probs = faceDetector(img_path)
        for i in range(boxes.shape[0]):
             box = scale(boxes[i, :])
             cropped = cropImage(orig_image, box)
             gender = genderClassifier(cropped)
             age = ageClassifier(cropped)
             print(f'Box {i} --> {gender}, {age}')
      #  except:
            # print("fail")
            """


# for i in range(len(x)):

# Detect single image
