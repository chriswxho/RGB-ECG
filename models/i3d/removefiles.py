import glob
import os

def reset(data_path):
    numpys = glob.glob(data_path)
    for x in numpys:
        os.remove(x)

data_path = '/users/kevin/downloads/aicure-dataset/*/*.npy'
reset(data_path)
