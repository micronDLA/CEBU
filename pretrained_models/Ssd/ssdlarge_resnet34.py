'''
Script to run detection task on MDLA
Models: ssdlarge_resnet34
'''
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import torch.onnx
from PIL import Image
import onnxruntime
import microndla
import os
from utils import *
#from microndla import MDLA

# Color Palette
CP_R = '\033[31m'
CP_G = '\033[32m'
CP_Y = '\033[33m'
CP_C = '\033[36m'
CP_0 = '\033[0m'

class ssdLargeResnet34:
    """
    Load MDLA and run classification model on it
    """
    def __init__(self, numfpga, numclus):
        """
        Compile ssdlarge model with resnet34 as the backbone, preprocess the data and run on MDLA
        """

        print('{}{}{}...'.format(CP_Y, 'Compiling MDLA', CP_0))
        ################################################################################
        # Compile on Micron DLA
        self.ie = microndla.MDLA()

        # Run the network in batch mode 
        self.ie.SetFlag('clustersbatchmode', '1')
        self.ie.SetFlag('nclusters', '4')

        self.cwd_path = (os.path.abspath(os.getcwd()))

        self.modelpath = self.cwd_path + '/onnx_files/ssdLarge-cut0.onnx'
        
        self.ie.Compile(self.modelpath)                                 # Compile the NN and generate instructions <save.bin> for MDLA

        self.device = torch.device("cpu")

        print('{}{}{}'.format(CP_G, 'MDLA compilation complete', CP_0))
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
        
        # Run the second cut half on CPU
        ort_session = onnxruntime.InferenceSession(self.cwd_path + '/onnx_files/ssdLarge-cut1.onnx')
        ort_inputs = {}

        for outObj, ortInps in zip(ort_session.get_inputs(), dla_output):
        #    npArr=npArr.reshape(np.array(inpObj.shape))
        #    tensorInp = torch.from_numpy(npArr).to(self.device).cpu().numpy()
            ort_inputs[outObj.name] = ortInps

        ort_outs = ort_session.run(None, ort_inputs)
        return ort_outs   

    def preprocess(self, img):
        # Preprocessing of input image required by ssd-small model
          
        img = img.convert('RGB').resize((1200, 1200))           #, Image.LANCZOS        # Resize it to the size expected by the network
        content_transform = transforms.Compose([transforms.ToTensor(), transforms.CenterCrop((1200, 1200))])
                                               # transforms.Lambda(lambda x: x.mul(255))])
        content_image = content_transform(img)
        content_image = self.normalize(content_image)

        content_image = content_image.unsqueeze(0).to(self.device)
        content_image = to_numpy(content_image) #(torch.transpose(content_image,1,3))    # Transpose to plane-major, as required by our API (HWC -> CHW)
    #    img = cv2.imread(content_image)
    #   content_image = content_image.cpu().numpy()                     # Convert to numpy 

        #print('{}{}{}'.format(CP_G, 'Preprocessing of input image complete', CP_0))

        return content_image #img

