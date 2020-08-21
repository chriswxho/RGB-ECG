import numpy as np
from tqdm import tqdm
from PIL import Image
import tensorflow as tf
import glob
import os
import cv2
from collections import defaultdict
import random
from keras.applications.mobilenet import MobileNet
from keras.layers import Input, Dense, Dropout, Lambda, Activation, concatenate, LSTM
from keras.models import Model
from keras import backend as K
#K.tensorflow_backend._get_available_gpus()
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
#np.random.seed(2017)
#tf.set_random_seed(2017)

height,width=(224,224)

def getvideoname(name):
    return '_'.join(name.split('\\')[-1].split('_')[:3])

def getframe(name):
    return name.split('\\')[-1].split('_')[-1].split('.')[0]

def getlabel(name):
    if len(name.split('\\')[-1].split('_'))>=5:
        #print(name.split('\\')[-1].split('_')[3:-1])
        try:
            return [float(x) for x in name.split('\\')[-1].split('_')[3:-1]]
        except:
            return [0,0,0]
    else:
        return [0,0,0]


def load_data():
    frames=glob.glob('{0}\\*\\*frames\\*.*.jpg'.format(r'C:\Users\Chris\Documents\projects\cs172b\aicure-dataset'))
    indexes=[x for x in range(len(frames))]
    datadict=defaultdict(list)
    labelsdict={}
    for frame in frames:
        datadict[getvideoname(frame)].append(frame)
    for key,data in datadict.items():
        #print(key)
        datadict[key] = sorted(data, key=lambda x:int(getframe(x)))
        labelsdict[key]=[getlabel(x) for x in data]
    #random.shuffle(indexes)
    return indexes,datadict,labelsdict

def find_index(index,data,labels):
    for key in data.keys():
        if index>=len(data[key]) - 2:
            index-=len(data[key])
            continue
        return data[key][index],labels[key][index]


def get_img(index, indexes, X, Y):
    x,y=find_index(index,X,Y)
    if Y==[0,0,0]:
        x,y=find_index(random.randint(len(indexes),X,Y))
    x=Image.open(x)
    x.convert('RGB')
    x=np.array(x)
    #x.resize((224,224,3))
    return x,y

def data_generator(X,Y,indexes,batch_size=4):
    while True:
        idxs=np.random.permutation(len(indexes))
        p0,p1,p2, q = [],[],[],[]
        for i in range(len(X) - 1):
            x_2, y_2 = get_img(idxs[i]+2, indexes, X, Y)
            # while y_1==[0,0,0]:
            #     tempindex=random.randint(len(indexes)-1)
            #     x_1, y_1 = get_img(temp+1, indexes, X, Y)
            # x,y = get_img(temp, indexes, X, Y)
            x_1, y_1 = get_img(idxs[i]+1, indexes, X, Y)
            x,y = get_img(idxs[i], indexes, X, Y)
            while y_2==[0,0,0]:
                tempindex=random.randint(0,len(indexes)-2)
                x_2, y_2 = get_img(idxs[tempindex]+2, indexes, X, Y)
                x_1, y_1 = get_img(idxs[tempindex]+1, indexes, X, Y)
                x,y = get_img(idxs[tempindex]+1, indexes, X, Y)
            

            p0.append(x)
            p1.append(x_1)
            p2.append(x_2)
            q.append(y_2[0])
            if len(q)==batch_size:
                yield {'input1': np.array(p0), 'input2': np.array(p1),'input3':np.array(p2)}, np.array(q)
                p0, p1,p2, q=[],[], [],[]
        if p0:
            yield {'input1': np.array(p0), 'input2': np.array(p1),'input3':np.array(p2)}, np.array(q)
            p0, p1,p2, q=[],[],[], []

def custom_stack_layer(tensor):
    return tf.keras.backend.stack(
        tensor, axis=1
    )

indexes,data,labels=load_data()
randomprints=open('random.txt','w')
#randomprints.write(len(indexes))

input_image=Input(shape=(height,width,3), name='input1')
input_image2=Input(shape=(height,width,3), name='input2')
input_image3=Input(shape=(height,width,3), name='input3')
randomprints.write(str(input_image.shape))
randomprints.flush()
mobilenet = MobileNet(include_top=False,pooling='avg')
x1 = mobilenet(input_image)
x2 = mobilenet(input_image2)
x3= mobilenet(input_image3)
x = Lambda(custom_stack_layer, name="lambda_layer")([x1,x2,x3])

lstm = LSTM(
    units=256, return_sequences=False, dropout=0.2
)(x)

out = Dense(1, activation='relu', trainable=False, name='out')(lstm)

model=Model(inputs=[input_image, input_image2,input_image3], outputs=out)
model.compile(optimizer='adam',loss='mean_squared_error', metrics=['mae'])

train_indexes=indexes[:int(0.7*len(indexes))]
validation_indexes=indexes[int(0.7*len(indexes)):]

earlystop = tf.keras.callbacks.EarlyStopping( monitor = 'val_mae', patience=5 )
reducelr = tf.keras.callbacks.ReduceLROnPlateau( monitor = 'val_mae', patience=2, factor=0.2, min_lr=0.0001 )

history=model.fit_generator(data_generator(data,labels,train_indexes),
                            steps_per_epoch=900*8,
                            epochs=50,
                            validation_data=data_generator(data,labels,validation_indexes),
                            validation_steps=385*8,
                            callbacks=[earlystop,reducelr])




import pandas as pd
hist_df=pd.DataFrame(history.history)
with open("hist.csv",mode='w') as f:
    hist_df.to_json(f)


model_json=model.to_json()
with open("MobileNetTransfer.json",'w') as json_file:
    json_file.write(model_json)

model.save_weights('MobileNetTransfer.h5')
