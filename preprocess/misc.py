
import os
import glob
import shutil

dataset_path = r'C:\Users\Chris\Documents\projects\cs172b\aicure-dataset'

def get_dataset_length(path):
    frames=glob.glob('{0}\\*\\*frames\\*.*.jpg'.format(path))
    print(len(frames))

def clean_files(path):
    folders=glob.glob('{}\\*\\face_frames'.format(path))+glob.glob('{}\\*\\face_frames_face'.format(path))
    for f in folders:
        print('Deleting {}'.format(f))
        shutil.rmtree(f)


if __name__ == "__main__":
    get_dataset_length(dataset_path)
    