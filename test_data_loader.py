from data_loader import *
from time import time
import multiprocessing as mp

for num_workers in range(2, mp.cpu_count(), 2):  
    # how to set up data paths to test data loader
    # train_loader = DataLoaderClassification(hr_path=, orig_path=, ttt_path=, threshold=,shuffle=True,num_workers=num_workers,batch_size=64,pin_memory=True)
    start = time()
    for epoch in range(1, 3):
        for i, data in enumerate(train_loader, 0):
            pass
    end = time()
    print("Time elapsed:{} seconds, num_workers={}".format(end - start, num_workers))