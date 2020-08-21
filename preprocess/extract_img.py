import cv2
import glob
import os
import pandas as pd
import shutil


logfile = open('failures.txt','w')
frames=glob.glob('{0}/*/*frames'.format(dataset_path))
for folder in frames:
    shutil.rmtree(folder)
movs=glob.glob('{0}/**/*.mov'.format(dataset_path))

for mov in movs:
    print (mov)
    vidcap= cv2.VideoCapture(mov)
    try:
        df=pd.read_csv(mov.split('.')[0]+'.csv')
    except:
        logfile.write('Failure on %s'%mov)
        logfile.flush()
    success,image=vidcap.read()
    count=0
    #print(mov)
    dest,tail=os.path.split(mov)
    #print(dest)
    #print(tail)
    dest=os.path.join(dest,tail.split('.')[0]+'_frames')
    #print(dest)
    if not os.path.exists(dest):
        os.mkdir(dest)
    if not success:
        print('Failure on', mov)
        continue
    while success:
        acctime=count/12
        try:
            tempdf=df[df['Time'].between(acctime/60-1/2880,acctime/60+1/2880)]
        except:
            tempdf=pd.DataFrame()
        if tempdf.empty:
            #print(os.path.join(dest,tail.split('.')[0]+'_%s.jpg'%count))
            cv2.imwrite(os.path.join(dest,tail.split('.')[0]+'_%s.jpg'%count),image)
        else:
            (ecg,skin_conductance,respiration)=0,0,0
            for index,row in tempdf.iterrows():
                ecg+=row['ECG']
                skin_conductance+=row['Skin_Conductance']
                respiration+=row['Respiration']
            ecg/=len(tempdf.index)
            skin_conductance/=len(tempdf.index)
            respiration/=len(tempdf.index)
            cv2.imwrite(os.path.join(dest,tail.split('.')[0]+'_%s_%s_%s_%s.jpg'%(ecg,skin_conductance,respiration,count)),image)
            del tempdf
        #if count==0:
        #print(os.path.join(dest,tail.split('.')[0]+'_%s.jpg'%count))
        success,image=vidcap.read()
        count+=1
