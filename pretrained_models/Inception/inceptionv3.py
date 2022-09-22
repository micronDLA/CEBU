'''
Script to run classification model on MDLA
Models: Inceptionv3
'''
import sys
import os
sys.path.append("../..")
import microndla
import PIL
from PIL import Image
import numpy as np
import onnxruntime
import torch


# Color Palette
CP_R = '\033[31m'
CP_G = '\033[32m'
CP_Y = '\033[33m'
CP_C = '\033[36m'
CP_0 = '\033[0m'


class InceptionV3:
    """
    Load MDLA and run classification model on it
    """
    def __init__(self, numfpga, numclus):
        """
        Compile Inceptionv3 model, preprocess the data and run on MDLA
        """

        print('{}{}{}...'.format(CP_Y, 'Initializing MDLA', CP_0))
        ################################################################################
        # Initialize Micron DLA
        self.ie = microndla.MDLA()

        # Run the network in batch mode 
        self.ie.SetFlag('clustersbatchmode', '1')
        self.ie.SetFlag('nclusters', '4')

        cwd_path = (os.path.abspath(os.getcwd()))

        self.ort_session = onnxruntime.InferenceSession(cwd_path + '/onnx_files/inception_v3_sim-cut0.onnx')

        self.modelpath = cwd_path + '/onnx_files/inception_v3_sim-cut1.onnx'
        
        self.ie.Compile(self.modelpath)                                 # Compile the NN and generate instructions <save.bin> for MDLA

        self.device = torch.device("cpu")
        
        print('{}{}{}'.format(CP_G, 'MDLA initialization complete', CP_0))
        print('{:-<80}'.format(''))

    def __call__(self, img):
        return self.forward(img)

    def __del__(self):
        self.ie.Free()

    def normalize(self, img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        for i in range(3):
            img[i] = (img[i] - mean[i]) / std[i]
        return img

    def forward(self, img):
        ort_inputs = {}

        ort_inputs[self.ort_session.get_inputs()[0].name] = img
        ort_outs = self.ort_session.run(None, ort_inputs)
        ort_outs = np.asarray(ort_outs).squeeze(0)

        dla_output = self.ie.Run(ort_outs)
     #   dla_output = self.ie.Run(img)
        return dla_output   

    def preprocess(self, img):
        # Preprocessing of input image required by ResNet model  
        img = img.convert('RGB').resize((224, 224), resample=PIL.Image.BILINEAR)       # #Change image mode to RGB and resize it to the size expected by the network
        img = np.array(img).astype(np.float32) / 255                    #Convert to numpy float
        img = np.ascontiguousarray(img.transpose(2,0,1))                #Transpose to plane-major, as required by our API (HWC -> CHW)
        img = self.normalize(img)                                       #Normalize the image
        img = np.expand_dims(img, axis=0)

        print('{}{}{}'.format(CP_G, 'Preprocessing of input image complete', CP_0))

        return img
