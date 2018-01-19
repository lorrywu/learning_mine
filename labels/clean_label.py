import csv

import csv
import numpy as np
from os import walk



def clean_csv(f):
    clean_dict = {}
    output = open('cleaned_label2.csv', 'w')

    for line in open(f, 'r').readlines():
        print line
        if 'user_id' not in line:
            csv_line = line.split(';')
            clean_dict[csv_line[0]] = line


    for k, v in clean_dict.items():
        output.write(v)
    output.close()

clean_csv('traindata_corrected.csv')
