import glob
import numpy as np
from sklearn.metrics import roc_auc_score as roc_auc_score
def folderfinder(filename):
	return filename.split('\\')[-2]

def converttobinary(arr):
	return np.array([1 if x>0.7 else 0 for x in arr])

resultspath=r'results\final_classify_results\**\*.npy'
datapath=r'C:\Users\Chris\Documents\projects\cs172b\aicure-dataset'


resultspath=glob.glob(resultspath)
#datapath=glob.glob(datapath)



for files in resultspath:
	folder=folderfinder(files)
	origfile=glob.glob(datapath+'/'+folder+'/*peak.npy')[0]
	y=converttobinary(np.load(files))
	x=np.load(origfile)
	rocval=roc_auc_score(x,y)
	print(files)
	print(rocval)
