Micron Deep Learning Accelerator - Customer Evaluation Utility

Version 1.0
---------------------------------
Classification: ResNet50, MobileNetV2 and InceptionV3
Detection: SSD-Small and SSD-Large
Dependencies: Streamlit, OpenCV, Numpy, Pandas, PIL
Release tested with Micron DLA SDK 2022.1 Release
Note: Datasets are not provided. They need to be added according to below file paths:
    if dataChoice == 'ImageNet':
        categories = cwd_path + '/Datasets/imagenet1k-test/imagenet1k_labels.txt'
        imgpath = cwd_path + '/Datasets/imagenet1k-test/dataset'
    elif dataChoice == 'COCO':
        categories = cwd_path + '/Datasets/coco/coco_labels.txt'
        imgpath = cwd_path + '/Datasets/coco/dataset'

Note: Separate OneDrive links exist for the required Datasets.