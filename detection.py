import numpy as np
import torch
import streamlit as st
import cv2
import os
from pretrained_models.Ssd import *

path = './testOuts/'

if not os.path.exists(path):
    os.makedirs(path)
else:
    os.system('rm -r ./testOuts/*')

def detect(ort_outs, img, idx, img_name, categories):
    coord_y=10
    detected = 0

    lab_lst = []
    
    for coord,cat,score in zip(ort_outs[0].squeeze(),ort_outs[1].squeeze(),ort_outs[2].squeeze()):
        if score>0.6:
            detected = 1
            cv2.rectangle(img, (int(coord[0]*img.shape[1]),int(coord[1]*img.shape[0])), (int(coord[2]*img.shape[1]),int(coord[3]*img.shape[0])) , (250,0,0),3)
            cv2.putText(img, categories[int(cat)], (int(coord[0]*img.shape[1]),int(coord[1]*img.shape[0])+coord_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            coord_y+=25
            lab_lst.append(categories[int(cat)])
    
 #   if detected: cat_dict
 #       cv2.imwrite("./testOuts/"+imgName.replace(".JPEG","_classi_ann.JPEG"), img)
   # for coord,cat in zip(ort_outs[0].squeeze()[:predictions],ort_outs[1].squeeze()[:predictions]):
    # for coord,cat in zip(ort_outs[0].squeeze(),ort_outs[1].squeeze()):
    #     cv2.rectangle(img, (int(coord[0]*img.shape[1]),int(coord[1]*img.shape[0])), 
    #                         (int(coord[2]*img.shape[1]),int(coord[3]*img.shape[0])) , (250,0,0),3)
    # #    print(img_name)
    # #    print(categories[int(cat)])
    #     cv2.putText(img, categories[int(cat)], (int(coord[0]*img.shape[1]),
    #                 int(coord[1]*img.shape[0]+ycoord)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, 
    #                 cv2.LINE_AA)
    #     ycoord += 10
        cv2.imwrite("./testOuts/"+img_name, img)#.replace(".jpg","_det_ann.jpg"), img)
        # cv2.imshow('image', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    return lab_lst

    """ 
    acc = []
    acc5 = []
    pred_label1 = []
    pred_label5 = []

    #Top-k results
    k = 5
    
    result = np.squeeze(result, axis=0)                     # Remove 1-D

    res_tens = torch.from_numpy(result)                     # Convert to tensor to perform softmax
    prob = torch.nn.functional.softmax(res_tens)            # Softmax probabilities
    top5, ix = torch.topk(prob, k)                         # Accuracy
    top5 = (top5*100).tolist()
    top1 = top5[0]
    acc.append(top1)
    top5 = [(str(round(b, 3))+' %') for b in top5]
    acc5.append(top5)
    
    st.write('')
    st.write('Top-1 Accuracy: ' f"{top1 :.4f} %")
    st.write('Top-5 Accuracy: ', top5)
    st.write('Top category and prediction: ', )
    for i in range(k):
        pred_label5.append(categories[ix[i]])
        if i == 0:
            st.write('Image %d: ' %idx + img_name + ': ' + categories[ix[i]], result[ix[i]])
            pred_label1.append(categories[ix[i]])

    return [acc, acc5, pred_label1, pred_label5] 



img = cv2.imread("/home/demo-1/Downloads/CocoRand.jpg")

#img = cv2.resize(img, (300,300), interpolation = cv2.INTER_AREA)

categories = '/home/demo-1/Downloads/coco/coco_labels.txt'

f = open(categories, "r")

categories = f.read().splitlines()



for cat in ort_outs[1].squeeze():

print(categories[int(cat)])



for coord in ort_outs[0].squeeze():

print(type(coord[0]))

cv2.rectangle(img, (int(coord[0]*300),int(coord[1]*300)), (int(coord[2]*300),int(coord[3]*300)) , (250,0,0),3)



cv2.imshow('image', img)

cv2.waitKey(0)

cv2.destroyAllWindows()

for cat in ort_outs[3].squeeze():
    print(categories[int(cat)])

print(img.shape[0])

for coord in ort_outs[1].squeeze():
    cv2.rectangle(img, (int(coord[0]*img.shape[1]),int(coord[1]*img.shape[0])), (int(coord[2]*img.shape[1]),int(coord[3]*img.shape[0])) , (250,0,0),3)

cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows() """

# predictions = int(ort_outs[0].squeeze())

#print(predictions)



# for cat in ort_outs[3].squeeze()[:predictions]:

# catname = categories[str(int(cat))]

#print(catname)



# img = cv2.imread(dir+imgName)



#for coord,cat in zip(ort_outs[1].squeeze()[:predictions],ort_outs[3].squeeze()[:predictions]):

#print(cat)

#cv2.rectangle(img, (int(coord[0]*img.shape[1]),int(coord[1]*img.shape[0])), (int(coord[2]*img.shape[1]),int(coord[3]*img.shape[0])) , (250,0,0),3)

# cv2.putText(img, categories[str(int(cat))], (int(coord[0]*img.shape[1]),int(coord[1]*img.shape[0]+10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

# cv2.imwrite("./testOuts/"+imgName.replace(".jpg","_det_ann.jpg"), img)
