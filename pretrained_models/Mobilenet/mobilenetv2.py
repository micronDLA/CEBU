'''
Script to run classification model on MDLA
Models: MobileNetv2
'''
import sys
sys.path.append("../..")
import microndla
import PIL
from PIL import Image
import numpy as np
import os


# Color Palette
CP_R = '\033[31m'
CP_G = '\033[32m'
CP_Y = '\033[33m'
CP_C = '\033[36m'
CP_0 = '\033[0m'


class MobileNetV2:
    """
    Load MDLA and run classification model on it
    """
    def __init__(self, numfpga, numclus):
        """
        Compile MobileNetv2 model, preprocess the data and run on MDLA
        """

        print('{}{}{}...'.format(CP_Y, 'Initializing MDLA', CP_0))
        ################################################################################
        # Initialize Micron DLA
        self.ie = microndla.MDLA()

        # Run the network in batch mode 
        self.ie.SetFlag('clustersbatchmode', '0')

        cwd_path = (os.path.abspath(os.getcwd()))
        print(cwd_path)

        self.modelpath = cwd_path + '/onnx_files/mobilenet_v2.onnx'
        
        self.ie.Compile(self.modelpath)                                 # Compile the NN and generate instructions <save.bin> for MDLA

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
        dla_output = self.ie.Run(img)
        return dla_output   

    def preprocess(self, img):
        # Preprocessing of input image required by ResNet model  
        img = img.convert('RGB').resize((224, 224), resample=PIL.Image.BILINEAR)       # #Change image mode to RGB and resize it to the size expected by the network
        img = np.array(img).astype(np.float32) / 255                    #Convert to numpy float
        img = np.ascontiguousarray(img.transpose(2,0,1))                #Transpose to plane-major, as required by our API (HWC -> CHW)
        img = self.normalize(img)                                       #Normalize the image

        print('{}{}{}'.format(CP_G, 'Preprocessing of input image complete', CP_0))

        return img
