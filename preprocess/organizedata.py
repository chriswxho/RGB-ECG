import glob
import os
from shutil import copyfile
folders=glob.glob('aicure*')
try:
    os.mkdir('trainingdata')
except:
    pass
for f in folders:
    for root,dirs,files in os.walk(f):
        for fil in files:
            if fil.endswith('.mov') or fil.endswith('.csv'):
                newdir='trainingdata/%s'%root.split('/')[-1]
                print(newdir)
                try:
                    os.mkdir(newdir)
                except:
                    pass
                copyfile(os.path.join(root,fil),os.path.join(newdir,fil))


