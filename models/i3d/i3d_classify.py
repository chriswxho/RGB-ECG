'''
Loads pretrained model of I3d Inception architecture for the paper: 'https://arxiv.org/abs/1705.07750'
Evaluates a RGB and Flow sample similar to the paper's github repo: 'https://github.com/deepmind/kinetics-i3d'
Peak classification model
'''

import numpy as np
import pandas as pd
import glob
import random
from tqdm import tqdm

import tensorflow as tf
from i3d_inception import Inception_Inflated3d
from collections import defaultdict
from keras.layers import Input, Dense, Dropout, Reshape, AveragePooling1D, ThresholdedReLU
from keras.models import Model
from keras import initializers
import keras.optimizers


NUM_FRAMES = 12
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
        if not 'cheek' in f:
            continue
        if any(badfile in f for badfile in ['03_2m_r2','12_4m_p','13_4m_p','13_4m_r','14_4m_p']):
            continue
        folder = getfoldername(f)
        key = getvideoname(f)

        full_path = '\\'.join([og_path,folder,key])
        video = np.load(full_path + '_cheek.npy')
        labels = np.load(full_path + '_peak.npy') # keep as peak

        num_samples = int(video.shape[0]//(NUM_FRAMES/2)*(NUM_FRAMES/2))
        videos = video[:num_samples]
        labels = labels[:num_samples]
        data_list.extend(videos)
        labels_list.extend(labels)
        num_frames_total += num_samples
    
    indexes=[x for x in range(int(num_frames_total//(NUM_FRAMES/2)))]
    random.shuffle(indexes)
    print(len(data_list))
    return indexes,data_list,labels_list

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

def scheduler(epoch, lr):
    if epoch > 5:
        return lr * tf.math.exp(-0.1)
    return lr

def train():
    # load the kinetics classes
    # datapath='/users/kevin/downloads/aicure-dataset/*/*.npy'
    datapath = r'C:\Users\Chris\Documents\projects\cs172b\aicure-dataset\*\*.npy'
    indexes,data,labels=load_data(datapath)

    base_model = Inception_Inflated3d(
        weights='rgb_imagenet_and_kinetics',
        include_top=False,
        input_shape=(NUM_FRAMES, FRAME_HEIGHT,FRAME_WIDTH,NUM_RGB_CHANNELS)
    )

    output=Dropout(0.5)(base_model.output)
    predict = Reshape((-1,1024))(output)
    #predict = AveragePooling1D(pool_size=3)(predict)
    predict = Dense(NUM_FRAMES,kernel_initializer='normal',activation='sigmoid')(predict)
    predict = ThresholdedReLU(theta=0.8, trainable=False)(predict)
    model = Model(inputs=base_model.input, outputs=predict)

    # freeze the first 100 layers
    for layer in model.layers[:100]:
        layer.trainable = False

    # randomize the weights for the remaining trainable layers
    # for layer in model.layers[150:195]: # change to 150:195 later
    #     layer.kernel_initializer = 'glorot_uniform'

    optimizer = keras.optimizers.Adam(lr=1e-4)
    model.compile(optimizer=optimizer,loss='mae', metrics=['accuracy'])

    #model.summary()

    train_indexes=indexes[:int(0.7*len(indexes))]
    validation_indexes=indexes[int(0.7*len(indexes)):]

    reducelr = tf.keras.callbacks.ReduceLROnPlateau( monitor = 'accuracy', patience=2, factor=0.2, min_lr=1e-8 )
    earlystop = tf.keras.callbacks.EarlyStopping( monitor = 'val_accuracy', patience=5 )

    callbacks = [ reducelr, earlystop ]

    history=model.fit_generator(data_generator(data,labels,train_indexes, BATCH_SIZE),
                                steps_per_epoch=int(STEPS*.7),
                                epochs=50,
                                validation_data=data_generator(data,labels,validation_indexes),
                                validation_steps=int(STEPS*.3),
                                callbacks=callbacks)
    write_out(history, 'hist.csv')
    save_model(model, 'i3d')

if __name__ == '__main__':
    train()
