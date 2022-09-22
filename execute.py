import time
import sys
import numpy as np
import os
import PIL
from PIL import Image
import time
import torch
import postprocess
from postprocess import *
import classification
import detection
import cv2
import random

def executeModel(modelName, batch, dataChoice, nDLA, nClusters, appChoice, modelChoice):
    """
    Execute the loaded model
    """
    numSamples = 10

    raw_latency = []                                            # Creating objects
    dla_latency = []
    thrpt_raw = []
    thrpt_dla = []
    cwd_path = (os.path.abspath(os.getcwd()))
    postprocess = PostProcess()

    if dataChoice == 'ImageNet':
        categories = cwd_path + '/Datasets/imagenet1k-test/imagenet1k_labels.txt'
        imgpath = cwd_path + '/Datasets/imagenet1k-test/dataset'
    elif dataChoice == 'COCO':
        categories = cwd_path + '/Datasets/coco/coco_labels.txt'
        imgpath = cwd_path + '/Datasets/coco/dataset'

    f = open(categories, "r")
    categories = f.read().splitlines()

    # Load the dataset
    for idx, img_name in random.sample(list(enumerate(os.listdir(imgpath))), numSamples):
        start_time = time.time()
        img = os.path.join(imgpath, img_name)
        im = Image.open(img)                                   #Load image into a numpy array

        im = modelName.preprocess(im)                         # Input preprocessing required by ResNet
        dla_start = time.time()                                 # Note DLA latency
        result = modelName(im)                                 # Model forward pass
        dla_lat = round((time.time() - dla_start)*1000, 2)
        #dla_latency.append(dla_lat)                             # Measure DLA latency
        dla_thrpt = round((batch*1000/dla_lat), 1)
        #thrpt_dla.append(dla_thrpt)                             # Measure DLA throughput
        
        if appChoice == 'Classification':
            acc, acc5, pred_label1, pred_label5 = classification.classify(result, idx, img_name, categories)

            raw_lat = round((time.time() - start_time)*1000, 2)
            #raw_latency.append(raw_lat)                             # Save raw latency value
            raw_thrpt = round((batch*1000/raw_lat), 1)
            #thrpt_raw.append(raw_thrpt)
            
        #s    import pdb; pdb.set_trace()
            postprocess.postProcess(modelName, img_name, nDLA, nClusters, appChoice, modelChoice, dataChoice, 
                                raw_lat, dla_lat, batch, acc, acc5, pred_label1, pred_label5, 
                                raw_thrpt, dla_thrpt)

            # st.write('Run completed! Now calculating performance metrics...')

            # postprocess.postProcessSum() #modelName, nDLA, nClusters, appChoice, modelChoice, dataChoice, 
            #                         #raw_latency, dla_latency, batch, acc, thrpt_raw, thrpt_dla)

        if appChoice == 'Detection': 
            im = cv2.imread(img)
            label_list = detection.detect(result, im, idx, img_name, categories)
            # acc, acc5, pred_label1, pred_label5 = detection.detect(
            # result, im, idx, img_name, categories)

          #  print(pred_label1,"####",pred_label5)

            raw_lat = round((time.time() - start_time)*1000, 2)
            #raw_latency.append(raw_lat)                             # Save raw latency value
            raw_thrpt = round((batch*1000/raw_lat), 1)
            #thrpt_raw.append(raw_thrpt)
        
    #s    import pdb; pdb.set_trace()
            postprocess.postProcess(modelName, img_name, nDLA, nClusters, appChoice, modelChoice, dataChoice, 
                                    raw_lat, dla_lat, batch, 0, 0, label_list, 'N/A', 
                                    raw_thrpt, dla_thrpt)
    st.write('Run completed! Now calculating performance metrics...')

    if appChoice == 'Detection':
        postprocess.postProcessDetSum() #modelName, nDLA, nClusters, appChoice, modelChoice, dataChoice, 
#                         #raw_latency, dla_latency, batch, acc, thrpt_raw, thrpt_dla)
    else:
        postprocess.postProcessSum()
    

    postprocess.save()
    del modelName                                               # Free


