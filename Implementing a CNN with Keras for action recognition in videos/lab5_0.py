# -*- coding: utf-8 -*-
from __future__ import print_function
import time, random, datetime, gc
from src.functions import *
from src.model import *
from src.data import *


if __name__ == "__main__":
    make_path('/espace/DLCV2')
    extract_RGB(data_path='/net/ens/DeepLearning/lab5/Data_TP/Videos', output_path='/espace/DLCV2/Data')
    stat_dataset(path='/espace/DLCV2/Data')
    compute_flow(data_path='/espace/DLCV2/Data', flow_calculation=True)


    
