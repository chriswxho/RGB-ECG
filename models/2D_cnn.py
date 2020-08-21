'''
Simple 2D CNN
'''

import numpy as np
import pandas as pd
import glob
import random
from tqdm import tqdm

import tensorflow as tf
from collections import defaultdict
from keras.applications.mobilenet import MobileNet
from keras.layers import Input, Dense, Dropout, Lambda
from keras.models import Model
import keras.optimizers

gpus = tf.config.list_physical_devices('GPU')
print("Num GPUs Available: ", len(gpus))
tf.config.experimental.set_memory_growth(gpus[0], True)

NUM_FRAMES = 1
FRAME_HEIGHT = 224
FRAME_WIDTH = 224
BATCH_SIZE = 16

STEPS = (121113 - 2160) / NUM_FRAMES / BATCH_SIZE

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
        if not 'cheek' in f:
            continue
        if any(badfile in f for badfile in ['03_2m_r2','12_4m_p','13_4m_p','13_4m_r','14_4m_p']):
            continue
        folder = getfoldername(f)
        key = getvideoname(f)

        full_path = '\\'.join([og_path,folder,key])
        video = np.load(full_path + '_cheek.npy')
        labels = np.load(full_path + '_peak.npy')

        num_samples = int(video.shape[0])
        videos = video[:num_samples]
        labels = labels[:num_samples]
        data_list.extend(videos)
        labels_list.extend(labels)
        num_frames_total += num_samples
    
    #assert(num_frames_total // 12 == num_frames_total/12)
    print(num_frames_total)
    indexes=[x for x in range(num_frames_total//NUM_FRAMES)]
    random.shuffle(indexes)
    return indexes,data_list,labels_list

def write_out(history, outpath):
    hist_df=pd.DataFrame(history.history)
    with open(outpath,mode='w') as f:
        hist_df.to_json(f)

def save_model(model, model_name):
    model_json=model.to_json()
    with open("{0}.json".format(model_name),'w') as json_file:
        json_file.write(model_json)

    model.save_weights('{0}.h5'.format(model_name))

def data_generator(X,Y,indexes,batch_size=BATCH_SIZE):
    # this creates a dictionary of 1 key to 1 label

    while True:
        idxs=np.random.permutation(len(indexes))
        p,q = [],[]
        for index in idxs:
            x = np.array(X[index])
            y = np.array(Y[index])

            x = np.squeeze(x)
            y = np.squeeze(y)
            
            p.append(x)
            q.append(y.T)
            if len(p)==batch_size:
                yield np.array(p),np.array(q)
                p,q=[],[]
        if p:
            yield np.array(p),np.array(q)
            p,q=[],[]


def train():
    datapath = r'C:\Users\Chris\Documents\projects\cs172b\aicure-dataset\*\*.npy'
    indexes,data,labels=load_data(datapath)

    # build the model
    input_image=Input(shape=(FRAME_HEIGHT,FRAME_WIDTH,3))
    base_model=MobileNet(input_tensor=input_image,include_top=False,pooling='avg')
    output=Dropout(0.2)(base_model.output)
    predict=Dense(1,kernel_initializer='normal')(output)

    model=Model(inputs=input_image, outputs=predict)
    optimizer = keras.optimizers.Adam(lr=1e-4)
    model.compile(optimizer=optimizer,loss='mse', metrics=['mae'])

    #model.summary()

    train_indexes=indexes[:int(0.7*len(indexes))]
    validation_indexes=indexes[int(0.7*len(indexes)):]

    reducelr = tf.keras.callbacks.ReduceLROnPlateau( monitor = 'mae', patience=2, factor=0.2, min_lr=1e-8 )
    earlystop = tf.keras.callbacks.EarlyStopping( monitor = 'val_mae', patience=5 )

    callbacks = [ reducelr, earlystop ]

    history=model.fit_generator(data_generator(data,labels,train_indexes),
                                steps_per_epoch=int(STEPS*.7),
                                epochs = 50,
                                validation_data = data_generator(data,labels,validation_indexes),
                                validation_steps=int(STEPS*.3),
                                callbacks = callbacks)

    write_out(history, 'hist.csv')
    save_model(model, 'MobileNetV2')

if __name__ == '__main__':
    train()