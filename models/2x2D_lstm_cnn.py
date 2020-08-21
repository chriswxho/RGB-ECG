'''
Siamese 2D CNN with LSTM
'''

import numpy as np
import pandas as pd
import glob
import random
from tqdm import tqdm

import tensorflow as tf
from collections import defaultdict
from keras.applications.mobilenet import MobileNet
from keras.layers import Input, Dense, Dropout, Lambda, concatenate, LSTM
from keras.models import Model
import keras.optimizers

gpus = tf.config.list_physical_devices('GPU')
print("Num GPUs Available: ", len(gpus))
tf.config.experimental.set_memory_growth(gpus[0], True)

NUM_FRAMES = 12
FRAME_HEIGHT = 224
FRAME_WIDTH = 224
BATCH_SIZE = 4

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

        num_samples = int(video.shape[0]//(NUM_FRAMES/2)*(NUM_FRAMES/2))
        videos = video[:num_samples]
        labels = labels[:num_samples]
        data_list.extend(videos)
        labels_list.extend(labels)
        num_frames_total += num_samples
    
    #assert(num_frames_total // 12 == num_frames_total/12)
    indexes=[x for x in range(int(num_frames_total//(NUM_FRAMES/2)))]
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
        p0,p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,q = [],[],[],[],[],[],[],[],[],[],[],[],[]
        for index in idxs:
            index = NUM_FRAMES//2 * index
            x0, y0 = np.squeeze(np.array(X[index])), np.squeeze(np.array(Y[index]))
            x1, y1 = np.squeeze(np.array(X[index+1])), np.squeeze(np.array(Y[index+1]))
            x2, y2 = np.squeeze(np.array(X[index+2])), np.squeeze(np.array(Y[index+2]))
            x3, y3 = np.squeeze(np.array(X[index+3])), np.squeeze(np.array(Y[index+3]))
            x4, y4 = np.squeeze(np.array(X[index+4])), np.squeeze(np.array(Y[index+4]))
            x5, y5 = np.squeeze(np.array(X[index+5])), np.squeeze(np.array(Y[index+5]))
            x6, y6 = np.squeeze(np.array(X[index+6])), np.squeeze(np.array(Y[index+6]))
            x7, y7 = np.squeeze(np.array(X[index+7])), np.squeeze(np.array(Y[index+7]))
            x8, y8 = np.squeeze(np.array(X[index+8])), np.squeeze(np.array(Y[index+8]))
            x9, y9 = np.squeeze(np.array(X[index+9])), np.squeeze(np.array(Y[index+9]))
            x10, y10= np.squeeze(np.array(X[index+10])), np.squeeze(np.array(Y[index+10]))
            x11, y11 = np.squeeze(np.array(X[index+11])), np.squeeze(np.array(Y[index+11]))
            
            
            p0.append(x0)
            p1.append(x1)
            p2.append(x2)
            p3.append(x3)
            p4.append(x4)
            p5.append(x5)
            p6.append(x6)
            p7.append(x7)
            p8.append(x8)
            p9.append(x9)
            p10.append(x10)
            p11.append(x11)
            q.append([y0,y1,y2,y3,y4,y5,y6,y7,y8,y9,y10,y11])
            
            if len(q)==batch_size:
                yield { 'input0': np.array(p0), 
                        'input1': np.array(p1),
                        'input2': np.array(p2),
                        'input3': np.array(p3),
                        'input4': np.array(p4),
                        'input5': np.array(p5), 
                        'input6': np.array(p6),
                        'input7': np.array(p7),
                        'input8': np.array(p8),
                        'input9': np.array(p9),
                        'input10': np.array(p10), 
                        'input11': np.array(p11),
                      }, np.array(q)
                p0,p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,q = [],[],[],[],[],[],[],[],[],[],[],[],[]
        if p0:
            yield { 'input0': np.array(p0), 
                    'input1': np.array(p1),
                    'input2': np.array(p2),
                    'input3': np.array(p3),
                    'input4': np.array(p4),
                    'input5': np.array(p5), 
                    'input6': np.array(p6),
                    'input7': np.array(p7),
                    'input8': np.array(p8),
                    'input9': np.array(p9),
                    'input10': np.array(p10), 
                    'input11': np.array(p11),
                    }, np.array(q)
            p0,p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,q = [],[],[],[],[],[],[],[],[],[],[],[],[]

def custom_stack_layer(tensor):
    return tf.keras.backend.stack(
        tensor, axis=1
    )

def train():
    datapath = r'C:\Users\Chris\Documents\projects\cs172b\aicure-dataset\*\*.npy'
    indexes,data,labels=load_data(datapath)

    # build the model
    input_image0=Input(shape=(FRAME_HEIGHT,FRAME_WIDTH,3), name='input0')
    input_image1=Input(shape=(FRAME_HEIGHT,FRAME_WIDTH,3), name='input1')
    input_image2=Input(shape=(FRAME_HEIGHT,FRAME_WIDTH,3), name='input2')
    input_image3=Input(shape=(FRAME_HEIGHT,FRAME_WIDTH,3), name='input3')
    input_image4=Input(shape=(FRAME_HEIGHT,FRAME_WIDTH,3), name='input4')
    input_image5=Input(shape=(FRAME_HEIGHT,FRAME_WIDTH,3), name='input5')
    input_image6=Input(shape=(FRAME_HEIGHT,FRAME_WIDTH,3), name='input6')
    input_image7=Input(shape=(FRAME_HEIGHT,FRAME_WIDTH,3), name='input7')
    input_image8=Input(shape=(FRAME_HEIGHT,FRAME_WIDTH,3), name='input8')
    input_image9=Input(shape=(FRAME_HEIGHT,FRAME_WIDTH,3), name='input9')
    input_image10=Input(shape=(FRAME_HEIGHT,FRAME_WIDTH,3), name='input10')
    input_image11=Input(shape=(FRAME_HEIGHT,FRAME_WIDTH,3), name='input11')
    mobilenet = MobileNet(include_top=False,pooling='avg')
    x0 = mobilenet(input_image0)
    x1 = mobilenet(input_image1)
    x2 = mobilenet(input_image2)
    x3 = mobilenet(input_image3)
    x4 = mobilenet(input_image4)
    x5 = mobilenet(input_image5)
    x6 = mobilenet(input_image6)
    x7 = mobilenet(input_image7)
    x8 = mobilenet(input_image8)
    x9 = mobilenet(input_image9)
    x10 = mobilenet(input_image10)
    x11 = mobilenet(input_image11)
    x0 = Dropout(0.2)(x0)
    x1 = Dropout(0.2)(x1)
    x2 = Dropout(0.2)(x2)
    x3 = Dropout(0.2)(x3)
    x4 = Dropout(0.2)(x4)
    x5 = Dropout(0.2)(x5)
    x6 = Dropout(0.2)(x6)
    x7 = Dropout(0.2)(x7)
    x8 = Dropout(0.2)(x8)
    x9 = Dropout(0.2)(x9)
    x10 = Dropout(0.2)(x10)
    x11 = Dropout(0.2)(x11)

    # x = concatenate([x0,x1,x2,x3,x4])

    x = Lambda(custom_stack_layer, name="lambda_layer")([x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11])
    lstm = LSTM(units=256, return_sequences=False, dropout=0.2)(x)
    out = Dense(NUM_FRAMES, activation='sigmoid', trainable=False, name='out')(lstm)

    input_images = [input_image0,input_image1,input_image2,input_image3,input_image4,
                   input_image5,input_image6,input_image7,input_image8,input_image9,
                   input_image10,input_image11]
    model = Model(inputs=input_images, outputs=out)
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
