import time
import sys
import streamlit as st
import execute


def modelManager(verbose):
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    nDLA = col1.radio("Number of MDLA:",['1','2'])

    clusterOptions = ['1','2','4','8']
    nClusters = col2.radio("Number of clusters (per MDLA):",clusterOptions)

    appChoice = col3.radio("AI application:",
    ['Classification', 'Detection', 'Segmentation', 'Pose Estimation'])

    if appChoice == "Classification":
        modelOptions = ['ResNet50','MobileNetV2','InceptionV3']
    elif appChoice == "Detection":
        modelOptions = ['SSD-Large (ResNet34)','SSD-Small (MobileNetV2)','YoloV5']
    elif appChoice == "Segmentation":
        modelOptions = ['Unet-3D']
    elif appChoice == "Pose Estimation":
        modelOptions = ['PoseNet']
    else:
        modelOptions = ['']

    modelChoice = col4.radio("AI model:",modelOptions)

    if appChoice == "Classification":
        dataChoice = col5.radio("Dataset:",['ImageNet'])
    elif appChoice == "Segmentation":
        dataChoice = col5.radio("Dataset:",['MedicalDataset'])
    elif appChoice == "Pose Estimation":
        dataChoice = col5.radio("Dataset:",['HumanActivityDataset'])
    else:
        dataChoice = col5.radio("Dataset:",['COCO'])

    batchSize = col6.radio('Batch size:',['1', '8'])
    batchSize = int(batchSize.split()[0])

    st.markdown("""---""")
 
    if verbose:
        st.write("Evaluation setup: **"+appChoice+"** application using **"+modelChoice+"** model")

    if st.button('Run inference'):       
        runModel(modelChoice,nDLA,nClusters,dataChoice,batchSize,verbose,appChoice)
    else:
        pass

    """latest_iteration = st.empty()
    bar = st.progress(0)

    for i in range(100):
        # Update the progress bar with each iteration.
        latest_iteration.text(f'Iteration {i+1}')
        bar.progress(i + 1)
        time.sleep(0.05)

    '...and now we\'re done!'"""

    return 0



def runModel(modelChoice, nDLA, nClusters, dataChoice, batchSize, verbose, appChoice):
    #Add function calls to respective models here
    if verbose:
        st.write("Running "+ modelChoice + "-" + nDLA + "-" + nClusters + "-" + dataChoice + "-" + batchSize)

    if modelChoice == "ResNet50":
        #st.success("Inference in progress >> ResNet50")
        from pretrained_models.Resnet.resnet50 import ResNet50
        resnet50 = ResNet50(nDLA, nClusters)
        execute.executeModel(resnet50, batchSize, dataChoice, nDLA, nClusters, appChoice, modelChoice)        
    elif modelChoice == "MobileNetV2":
        #st.success("Inference in progress >> MobileNetV2")
        from pretrained_models.Mobilenet.mobilenetv2 import MobileNetV2
        mobilenetv2 = MobileNetV2(nDLA, nClusters)
        execute.executeModel(mobilenetv2, batchSize, dataChoice, nDLA, nClusters, appChoice, modelChoice)        
    elif modelChoice == "InceptionV3":
        #st.success("Inference in progress >> InceptionV3")
        from pretrained_models.Inception.inceptionv3 import InceptionV3
        inceptionv3 = InceptionV3(nDLA, nClusters)
        execute.executeModel(inceptionv3, batchSize, dataChoice, nDLA, nClusters, appChoice, modelChoice)
    elif modelChoice == "SSD-Large (ResNet34)":
        #st.success("Inference in progress >> SSD-Large")
        from pretrained_models.Ssd.ssdlarge_resnet34 import ssdLargeResnet34
        ssdlarge_resnet34 = ssdLargeResnet34(nDLA, nClusters)
        execute.executeModel(ssdlarge_resnet34, batchSize, dataChoice, nDLA, nClusters, appChoice, modelChoice) 
    elif modelChoice == "SSD-Small (MobileNetV2)":
        #st.success("Inference in progress >> SSD-Small")
        from pretrained_models.Ssd.ssdsmall_mobilenetv2 import ssdSmallMobileNetV2
        ssdsmall_mobilenetv2 = ssdSmallMobileNetV2(nDLA, nClusters)
        execute.executeModel(ssdsmall_mobilenetv2, batchSize, dataChoice, nDLA, nClusters, appChoice, modelChoice)
    elif modelChoice == "YoloV5":
        #st.write("Inference in progress...")
        yoloV5(nDLA, nClusters, dataChoice, batchSize, modelPath)
        time.sleep(2)
    elif modelChoice == "Unet":
        #st.write("Inference in progress...")
        uNet(nDLA, nClusters, dataChoice, batchSize, modelPath)
        time.sleep(2)
    else:
        #st.write("Inference in progress...")
        time.sleep(2)
        
    st.success("Inference task executed!")
    #st.balloons()

