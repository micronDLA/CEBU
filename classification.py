import numpy as np
import torch
import streamlit as st
import cv2

def classify(result, idx, img_name, categories):
    acc = []
    acc5 = []
    pred_label1 = []
    pred_label5 = []

    #Top-k results
    k = 5
    
    result = np.squeeze(result, axis=0)                     # Remove 1-D

    res_tens = torch.from_numpy(result)                     # Convert to tensor to perform softmax

    prob = torch.nn.functional.softmax(res_tens,0)

    top5, ix = torch.topk(prob, k)                         # Accuracy
    top5 = (top5).tolist()
    acc = round(top5[0]*100, 2)
    acc5 = [(round(b, 4)) for b in top5]
    
    
    for i in range(k):
        pred_label5.append(categories[ix[i]])
        if i == 0:
            pred_label1.append(categories[ix[i]])

    return [acc, acc5, pred_label1, pred_label5] 
