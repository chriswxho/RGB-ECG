# changes all the fucking \t in 13_4m_r or something 
# whatever the fuck im tilted into the comma separated
import pandas as pd
import glob

def convert_csvs(data_path):
    csvs = glob.glob(data_path)
    for c in csvs:
        print (c)
        tabs = False
        content = ''
        with open(c, 'r') as f:
            content = f.read()
            by_line = content.split('\n')
            second_line = by_line[1]
            if '\t' in second_line:
                content = content.replace('\t', ',')
                tabs = True
        if tabs:
            with open(c, 'w') as f:
                f.write(content)

data_path = r'C:\Users\Chris\Documents\projects\cs172b\aicure-dataset\*\*.csv'
convert_csvs(data_path)
