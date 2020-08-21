'''
Loads pretrained model of I3d Inception architecture for the paper: 'https://arxiv.org/abs/1705.07750'
Evaluates a RGB and Flow sample similar to the paper's github repo: 'https://github.com/deepmind/kinetics-i3d'
Unfreezes all layers and re-trains the entire network.
'''

import numpy as np
import pandas as pd
import glob
import random
from tqdm import tqdm

import tensorflow as tf
from collections import defaultdict
from keras.models import Model, load_model, model_from_json
import keras.optimizers


NUM_FRAMES = 30
FRAME_HEIGHT = 224
FRAME_WIDTH = 224
NUM_RGB_CHANNELS = 3
BATCH_SIZE = 2

STEPS = 174381 / NUM_FRAMES / BATCH_SIZE

gpus = tf.config.list_physical_devices('GPU')
print("Num GPUs Available: ", len(gpus))
tf.config.experimental.set_memory_growth(gpus[0], True)

def getvideoname(name):
    return '_'.join(name.split('\\')[-1].split('.')[0].split('_')[:-1])

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

def getfoldername(path):
    return path.split('\\')[-2]

def getprepath(path):
    return '\\'.join(path.split('\\')[:-2])

def load_data(data_path):
    # this creates a dictionary of key to video, another key to labels
    # data dict is x by 224 by 224 by 3
    # labels dict is x by 2
    np_files = glob.glob(data_path)
    data_list = []
    labels_list = []
    og_path = getprepath(data_path)
    num_files = len(np_files)
    num_frames_total = 0

    for i in tqdm(range(num_files),ncols=80): #num_files
        f = np_files[i]
        if 'video' in f or 'peak' in f:
            continue
        folder = getfoldername(f)
        key = getvideoname(f)

        full_path = '\\'.join([og_path,folder,key])
        video = np.load(full_path + '_cheek.npy')
        labels = np.load(full_path + '_label.npy')

        #num_samples = video.shape[0]//12*12
        num_samples = int(video.shape[0]//(NUM_FRAMES/2)*(NUM_FRAMES/2))
        videos = video[:num_samples]
        labels = labels[:num_samples]
        data_list.extend(videos)
        labels_list.extend(labels)
        num_frames_total += num_samples
    
    #assert(num_frames_total // 12 == num_frames_total/12)
    indexes=[x for x in range(int(num_frames_total//(NUM_FRAMES/2)))]
    random.shuffle(indexes)

    high = max(labels_list)
    low = min(labels_list)
    for i in range(len(labels_list)):
        labels_list[i] = (labels_list[i] - low) / (high - low)

    return indexes,data_list,labels_list


height,width=224,224
def data_generator(X,Y,indexes,batch_size=BATCH_SIZE):
    # this creates a dictionary of key to 6 frames, another same key to 1 label

    while True:
        idxs=np.random.permutation(len(indexes))
        p,q = [],[]
        for index in idxs:
            x = np.array(X[int(NUM_FRAMES/2)*index: int(NUM_FRAMES/2)*index+NUM_FRAMES])
            y = np.array(Y[int(NUM_FRAMES/2)*index: int(NUM_FRAMES/2)*index+NUM_FRAMES])

            p.append(x)
            q.append(y.T)
            if len(p)==batch_size:
                # q = np.expan_dims()
                yield np.array(p),np.array(q)
                p,q=[],[]
        if p:
            yield np.array(p),np.array(q)
            p,q=[],[]

def write_out(history, outpath):
    hist_df=pd.DataFrame(history.history)
    with open(outpath,mode='w') as f:
        hist_df.to_json(f)

def save_model(model, model_name):
    model_json=model.to_json()
    with open("{0}.json".format(model_name),'w') as json_file:
        json_file.write(model_json)

    model.save_weights('{0}.h5'.format(model_name))

def load_model(nn_path,name):
            
    nn_path += '\\{}\\i3d'.format(name)+'{}'

    json_file = open(nn_path.format('.json'),'r')
    data = json_file.read()
    json_file.close()

    model = model_from_json(data,custom_objects = {'tf':tf})
    model.load_weights(nn_path.format('.h5'))
    return model

def train():
    # load the kinetics classes
    # datapath='/users/kevin/downloads/aicure-dataset/*/*.npy'
    datapath = r'C:\Users\Chris\Documents\projects\cs172b\aicure-dataset\*\*.npy'
    indexes,data,labels=load_data(datapath)

    nnpath = r'C:\Users\Chris\Documents\projects\cs172b\project\172Bproj\results'

    model = load_model(nnpath,'i3d_cheek_normalized')

    for layer in model.layers[:-125]:
        layer.trainable=True

    optimizer = keras.optimizers.Adam(lr=1e-7)
    model.compile(optimizer=optimizer,loss='mse', metrics=['mae'])

    model.summary()

    train_indexes=indexes[:int(0.7*len(indexes))]
    validation_indexes=indexes[int(0.7*len(indexes)):]

    earlystop = tf.keras.callbacks.EarlyStopping( monitor = 'val_mae', patience=3 )

    callbacks = [ earlystop ]

    history=model.fit_generator(data_generator(data,labels,train_indexes, BATCH_SIZE),
                                steps_per_epoch=int(STEPS*.7),
                                epochs=10,
                                validation_data=data_generator(data,labels,validation_indexes),
                                validation_steps=int(STEPS*.3),
                                callbacks=callbacks)
    write_out(history, 'hist.csv')
    save_model(model, 'i3d')

if __name__ == '__main__':
    train()
