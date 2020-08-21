import skvideo.io
import glob
import os
import numpy as np
from moviepy.editor import ImageClip, concatenate_videoclips, VideoFileClip
import pandas as pd
import cv2
import dlib
from imutils import face_utils

PREDICTOR_PATH = r"C:\Users\Chris\Documents\projects\cs172b\project\172Bproj\shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(PREDICTOR_PATH)
cascade_path = 'haarcascade_frontalface_default.xml'
cascade = cv2.CascadeClassifier(cascade_path)


def convert_data(data_path):
    videos = glob.glob(data_path)
    size = 224
    for v in videos:
        '''
        if 'small' not in v:
            resize_path=resize_video(v)
            print (v)
            #continue
        v=resize_path
        file_path = '.'.join(v.split('.')[:-2])

        videodata = skvideo.io.vread(v)
        cam = cv2.VideoCapture(v)
        fps = cam.get(cv2.CAP_PROP_FPS)
        framelength=videodata.shape[0]
        try:
            df=pd.read_csv(file_path+'.csv', delimiter=',')
        except:
            continue
        labels=[]
        acctime=float(df.iloc[0]['Time'])

        for frame in range(framelength):
            acctime+=1/(fps*60)
            tempdf=df.iloc[(df['Time']-acctime).abs().argsort()[:1]]
            if abs(tempdf.iloc[0]['Time']-acctime)<1/(fps*60):
                labels.append([tempdf.iloc[0]['ECG']])
            else:
                print ("got here", frame, v, tempdf.iloc[0]['Time'], acctime)
                break
        labels=np.array(labels)
        #np.save(file_path+'_label.npy',labels)
        height = videodata.shape[2]
        reshaped = videodata[:,:, (height-size)//2:(height+size)//2, :]
        '''
        file_path = '.'.join(v.split('.')[:-1])
        print(file_path + "_video.npy")
        try:
            reshaped=np.load(file_path + "_video.npy")
        except:
            print('failure on '+v)
            continue
        cropped=np.zeros(reshaped.shape)

        for idx,frames in enumerate(reshaped):
            #cv2.imwrite('test.jpg',frames)
            gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(gray, 1.3, 5)
            if len(faces)!=0:
                (x, y, w, h)=faces[0]
                rect = dlib.rectangle(x, y, x + w, y + h)
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)
                up=shape[1][1]
                left=shape[2][0]
                right=shape[16][0]
                down=shape[3][1]
            cropped[idx,up:down,left:right,:]+=frames[up:down,left:right]
        np.save(file_path+"_cheek.npy",cropped.astype(np.uint8))
            #cv2.imwrite('image.png',frames[up:down,left:right])
         
        #print (v, reshaped.shape, labels.shape)
        #np.save(file_path + "_video.npy", reshaped)

def resize_video(file_path):
    clip = VideoFileClip(file_path)
    clip_resized = clip.resize(height=224) 
    write_file_path = file_path+"_small.mp4"
    #clip_resized.write_videofile(write_file_path)
    return write_file_path

def write_to_video(arr):
    # use for testing and making sure it outputs correctly
    images=[ImageClip(x).set_duration(0.05) for x in arr]
    concat_clip = concatenate_videoclips(images, method="compose")
    concat_clip.write_videofile('test.mp4', fps=15)

data_path = r'C:\Users\Chris\Documents\projects\cs172b\aicure-dataset\*\*.mov'
# data_path='../trainingdata/*/*.mov'
convert_data(data_path)
