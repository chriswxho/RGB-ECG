import numpy as np
from PIL import Image
import glob
import os
import cv2
from collections import defaultdict
import random
np.random.seed(2017)
prototxt_path='deploy.prototxt'#os.path.abspath('deploy.prototxt')
caffemodel_path='res10_300x300_ssd_iter_140000.caffemodel'#os.path.abspath('res10_300x300_ssd_iter_140000.caffemodel')

face_detector=cv2.dnn.readNetFromCaffe(prototxt_path,caffemodel_path)

def getvideoname(name):
    return '_'.join(name.split('\\')[-1].split('_')[:3])

def getframe(name):
    return name.split('\\')[-1].split('_')[-1].split('.')[0]

def getlabel(name):
    if len(name.split('\\')[-1].split('_'))>=5:
        #print(name.split('/')[-1].split('_')[3:-1])
        try:
            return [float(x) for x in name.split('\\')[-1].split('_')[3:-1]]
        except:
            return [0,0,0]
    else:
        return [0,0,0]

def pad(img, h, w):
    # https://ai-pool.com/d/padding-images-with-numpy
    top_pad = np.floor((h - img.shape[0]) / 2).astype(np.uint16)
    bottom_pad = np.ceil((h - img.shape[0]) / 2).astype(np.uint16)
    right_pad = np.ceil((w - img.shape[1]) / 2).astype(np.uint16)
    left_pad = np.floor((w - img.shape[1]) / 2).astype(np.uint16)
    return np.copy(np.pad(img, ((top_pad, bottom_pad), (left_pad, right_pad), (0, 0)), mode='constant', constant_values=0))

def create_crop_images(dataset_path):
    frames=glob.glob('{0}\\*\\*frames\\*.jpg'.format(dataset_path))
    for frame in frames:
        #folder_name = frame.split('/')[-3]
        dest=os.path.dirname(frame)+'_face'
        #print(dest)
        #dest = os.path.join(folder_name, 'face_frames')
        
        frame_dest = os.path.join(dest, os.path.basename(frame))
        #print(frame_dest)
        if not os.path.exists(dest):
            os.mkdir(dest)
        cv2.imwrite(frame_dest, crop_face(cv2.imread(frame)))

def crop_face(image):
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_detector.setInput(blob)
    detections = face_detector.forward()
    if detections==[]:
        return
    confidences=[detections[0,0,i,2] for i in range(detections.shape[2])]
    maxconfidence=confidences.index(max(confidences))
    bestboundingbox=detections[0,0,maxconfidence,3:7]*np.array([w,h,w,h])
    (startX,startY,endX,endY)=bestboundingbox.astype('int')
    crop_img=image[startY:endY,startX:endX]
    return resize(crop_img)

def resize(crop_img):
    max_width, max_height = 540,540
    return pad(crop_img, max_height, max_width)

def get_max_dimension():
    # returns 540
    dataset_path = r'C:\Users\Chris\Documents\projects\cs172b\aicure-dataset'
    frames=glob.glob('{0}\\**\\*frames\\*.jpg'.format(dataset_path))
    max_dimension = 0
    for frame in frames:
        crop_img = crop_face(cv2.imread(frame))
        crop_height, crop_width, _ = crop_img.shape
        if crop_height > max_dimension:
            max_dimension = crop_height
        if crop_width > max_dimension:
            max_dimension = crop_width
    return max_dimension

dataset_path = r'C:\Users\Chris\Documents\projects\cs172b\aicure-dataset'
create_crop_images(dataset_path)






