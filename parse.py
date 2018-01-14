import numpy as np
import pandas as pd


with open('hidden1_to_hidden2.txt') as fin, open('newfile2.txt', 'w') as fout:
    for line in fin:
        fout.write(line.replace('\t', ','))

