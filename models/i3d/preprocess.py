import skvideo
import glob
import os
import numpy as np
from moviepy.editor import ImageClip, concatenate_videoclips, VideoFileClip
import pandas as pd
import cv2

skvideo.setFFmpegPath(r'C:\Users\Chris\Documents\projects\cs172b\ffmpeg-4.2.2-win64-shared\bin')

import skvideo.io

def convert_data(data_path):
    videos = glob.glob(data_path)
    size = 224

    for v in videos:
        # if '13_4m_r' not in v:
        #     continue
        # print (v)
        file_path = '.'.join(v.split('.')[:-2])

        videodata = skvideo.io.vread(v)
        cam = cv2.VideoCapture(v)
        fps = cam.get(cv2.CAP_PROP_FPS)
        framelength=videodata.shape[0]
        try:
            df=pd.read_csv(file_path+'.csv', delimiter=',')
        except:
            print('Failed to find csv for',os.path.basename(v))
        threshold = df.max()['ECG']*0.6

        labels=[]
        acctime=df.iloc[0]['Time']
        peaks = []
        last_index = 0

        for x in range(framelength):
            tempdf=df.iloc[(df['Time']-acctime).abs().argsort()[:1]]
            index = tempdf.index.values[0]
            ecg_val = tempdf['ECG'].values[0]

            ecg_values = [df.loc[x]['ECG'] for x in range(last_index, index+1)]
            max_ecg = max(ecg_values)
            if max_ecg > threshold:
                peaks.append([1])
                labels.append([max_ecg])
            else:
                peaks.append([0])
                labels.append([ecg_val])
            acctime+=1/(fps*60)
            last_index = index

        labels=np.array(labels)
        peaks=np.array(peaks)
        np.save(file_path+'_label.npy',labels)
        np.save(file_path + '_peak.npy',peaks)

        height = videodata.shape[2]
        reshaped = videodata[:,:, (height-size)//2:(height+size)//2, :]
        print (v, reshaped.shape, labels.shape)
        np.save(file_path + "_video.npy",reshaped)

def resize_video(file_path):
    clip = VideoFileClip(file_path)
    clip_resized = clip.resize(height=224) 
    write_file_path = file_path+"_small.mp4"
    clip_resized.write_videofile(write_file_path)
    return write_file_path

def write_to_video(arr):
    # use for testing and making sure it outputs correctly
    images=[ImageClip(x).set_duration(0.05) for x in arr]
    concat_clip = concatenate_videoclips(images, method="compose")
    concat_clip.write_videofile('test.mp4', fps=15)

#data_path = '/users/kevin/downloads/aicure-dataset/*/*_small.mp4'
data_path = r'C:\Users\Chris\Documents\projects\cs172b\aicure-dataset\*\*_small.mp4'
convert_data(data_path)
