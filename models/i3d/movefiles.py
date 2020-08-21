import glob
import shutil
import os

new_path = '/users/kevin/downloads/aicure-dataset/'
data_path = '/users/kevin/downloads/aicure2/*/*.npy'

videos = glob.glob(data_path)
for video in videos:
    if 'label' in video or 'cheek' in video:
        end = '/'.join(video.split('/')[-2:])
        folder = video.split('/')[-2]
        file_name = video.split('/')[-1]
        if not os.path.exists(new_path+ '/' + folder):
            os.makedirs(new_path+'/'+folder)
        shutil.move(video, new_path + end)
