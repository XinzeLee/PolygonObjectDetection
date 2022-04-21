# Polygon-Yolov5
This repository is based on Ultralytics/yolov5, with adjustments to enable polygon prediction boxes.

## Section I. Description
The codes are based on Ultralytics/yolov5, and several functions are added and modified to enable polygon prediction boxes.

The modifications compared with Ultralytics/yolov5 and their brief descriptions are summarized below:

  1. data/polygon_ucas.yaml : Exemplar UCAS-AOD dataset to test the effects of polygon boxes
  2. data/images/UCAS-AOD : For the inference of polygon-yolov5s-ucas.pt

  3. models/common.py :
    <br/> 3.1. class Polygon_NMS : Non-Maximum Suppression (NMS) module for Polygon Boxes
    <br/> 3.2. class Polygon_AutoShape : Polygon Version of Original AutoShape, input-robust polygon model wrapper for passing cv2/np/PIL/torch inputs. Includes preprocessing, inference and Polygon_NMS
    <br/> 3.3. class Polygon_Detections : Polygon detections class for Polygon-YOLOv5 inference results
  4. models/polygon_yolov5s_ucas.yaml : Configuration file of polygon yolov5s for exemplar UCAS-AOD dataset
  5. **models/yolo.py :**
    <br/> 5.1. class Polygon_Detect : Detect head for polygon yolov5 models with polygon box prediction
    <br/> 5.2. class Polygon_Model : Polygon yolov5 models with polygon box prediction
    
  6. **utils/iou_cuda : CUDA extension for iou computation of polygon boxes**
    <br/> 6.1. extensions.cpp : CUDA extension file
    <br/> 6.2. inter_union_cuda.cu : CUDA code for computing iou of polygon boxes
    <br/> 6.3. setup.py : for building CUDA extensions module polygon_inter_union_cuda, with two functions polygon_inter_union_cuda and polygon_b_inter_union_cuda
  7. **utils/autoanchor.py :**
    <br/> 7.1. def polygon_check_anchors : Polygon version of original check_anchors
    <br/> 7.2. def polygon_kmean_anchors : Create kmeans-evolved anchors from polygon-enabled training dataset, use minimum outter bounding box as approximations
  8. **utils/datasets.py :**
    <br/> 8.1. def polygon_random_perspective : Data augmentation for datasets with polygon boxes (augmentation effects: HSV-Hue, HSV-Saturation, HSV-Value, rotation, translation, scale, shear, perspective, flip up-down, flip left-right, mosaic, mixup)
    <br/> 8.2. def polygon_box_candidates : Polygon version of original box_candidates
    <br/> 8.3. class Polygon_LoadImagesAndLabels : Polygon version of original LoadImagesAndLabels
    <br/> 8.4. def polygon_load_mosaic : Loads images in a 4-mosaic, with polygon boxes
    <br/> 8.5. def polygon_load_mosaic9 : Loads images in a 9-mosaic, with polygon boxes
    <br/> 8.6. def polygon_verify_image_label : Verify one image-label pair for polygon datasets
    <br/> 8.7. def create_dataloader : Has been modified to include polygon datasets
  9. **utils/general.py :**
    <br/> 9.1. def xyxyxyxyn2xyxyxyxy : Convert normalized xyxyxyxy or segments into pixel xyxyxyxy or segments
    <br/> 9.2. def polygon_segment2box : Convert 1 segment label to 1 polygon box label
    <br/> 9.3. def polygon_segments2boxes : Convert segment labels to polygon box labels
    <br/> 9.4. def polygon_scale_coords : Rescale polygon coords (xyxyxyxy) from img1_shape to img0_shape
    <br/> 9.5. def polygon_clip_coords : Clip bounding polygon xyxyxyxy bounding boxes to image shape (height, width)
    <br/> 9.6. def polygon_inter_union_cpu : iou computation (polygon) with cpu
    <br/> 9.7. def polygon_box_iou : Compute iou of polygon boxes via cpu or cuda
    <br/> 9.8. def polygon_b_inter_union_cpu : iou computation (polygon) with cpu for class Polygon_ComputeLoss in loss.py
    <br/> 9.9. def polygon_bbox_iou : Compute iou of polygon boxes for class Polygon_ComputeLoss in loss.py via cpu or cuda
    <br/> 9.10. def polygon_non_max_suppression : Runs Non-Maximum Suppression (NMS) on inference results for polygon boxes
    <br/> 9.11. def polygon_nms_kernel : Non maximum suppression kernel for polygon-enabled boxes
    <br/> 9.12. def order_corners : Return sorted corners for loss.py::class Polygon_ComputeLoss::build_targets
  10. **utils/loss.py :**
    <br/> 10.1. class Polygon_ComputeLoss : Compute loss for polygon boxes
  11. utils/metrics.py :
    <br/> 11.1. class Polygon_ConfusionMatrix : Polygon version of original ConfusionMatrix
  12. utils/plots.py :
    <br/> 12.1. def polygon_plot_one_box : Plot one polygon box on image
    <br/> 12.2. def polygon_plot_one_box_PIL : Plot one polygon box on image via PIL
    <br/> 12.3. def polygon_output_to_target : Convert model output to target format (batch_id, class_id, x1, y1, x2, y2, x3, y3, x4, y4, conf)
    <br/> 12.4. def polygon_plot_images : Polygon version of original plot_images
    <br/> 12.5. def polygon_plot_test_txt : Polygon version of original plot_test_txt
    <br/> 12.6. def polygon_plot_targets_txt : Polygon version of original plot_targets_txt
    <br/> 12.7. def polygon_plot_labels : Polygon version of original plot_labels
  
  13. **polygon_train.py : For training polygon-yolov5 models**
  14. **polygon_test.py : For testing polygon-yolov5 models**
  15. **polygon_detect.py : For detecting polygon-yolov5 models**
  16. requirements.py : Added python model shapely
  
## Section II. How Does Polygon Boxes Work? How Does Polygon Boxes Different from Axis-Aligned Boxes?
  1. build_targets in class Polygon_ComputeLoss & forward in class Polygon_Detect
<br/>
<p align="center">
<img src="https://user-images.githubusercontent.com/87064748/129332763-93d821f4-19f2-4f82-a8f5-67cd40de1449.jpg" width="800">
</p>
  2. order_corners in general.py
<br/>
<p align="center">
<img src="https://user-images.githubusercontent.com/87064748/129332816-ca445e21-090b-47ae-b950-9329fe0c35eb.jpg" width="800">
</p>
  3. Illustrations of box loss of polygon boxes
<br/>
<p align="center">
<img src="https://user-images.githubusercontent.com/87064748/124885366-c5a03000-e005-11eb-974e-14d61956955f.jpg" width="516">
</p>

## Section III. Installation
***For the CUDA extension to be successfully built without error, please use CUDA version >= 11.2. The codes have been verified in Ubuntu 16.04 with Tesla K80 GPU.***
<div class="highlight highlight-source-shell position-relative">
<pre>
# The following codes install CUDA 11.2 from scratch on Ubuntu 16.04, if you have installed it, please ignore
# If you are using other versions of systems, please check https://tutorialforlinux.com/2019/12/01/how-to-add-cuda-repository-for-ubuntu-based-oses-2/
# Install Ubuntu kernel head
sudo apt install linux-headers-$(uname -r)
<br/># Pinning CUDA repo
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-ubuntu1604.pin
sudo mv cuda-ubuntu1604.pin /etc/apt/preferences.d/cuda-repository-pin-600
<br/># Add CUDA GPG key
sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
<br/># Setting up CUDA repo
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/ /"
<br/># Refresh apt repositories
sudo apt update
<br/># Installing CUDA 11.2
sudo apt install cuda-11-2 -y
sudo apt install cuda-toolkit-11-2 -y
<br/># Setting up path
echo 'export PATH=/usr/local/cuda-11.2/bin${PATH:+:${PATH}}' >> $HOME/.bashrc
# You are done installing CUDA 11.2
<br/># Check NVIDIA
nvidia-smi
# Update all apts
sudo apt-get update
sudo apt-get -y upgrade
<br/># Begin installing python 3.7
curl -o ~/miniconda.sh -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x ~/miniconda.sh
./miniconda.sh -b
echo "PATH=~/miniconda3/bin:$PATH" >> ~/.bashrc 
source ~/.bashrc
conda install -y python=3.7
# You are done installing python</pre>
</div>

***The following codes set you up with the Polygon Yolov5.***
<div class="highlight highlight-source-shell position-relative">
<pre>
# clone git repo
git clone https://github.com/XinzeLee/PolygonObjectDetection
cd PolygonObjectDetection/polygon-yolov5
# install python package requirements
pip install -r requirements.txt
# install CUDA extensions
cd utils/iou_cuda
python setup.py install
# cd back to polygon-yolov5 folder
cd .. && cd ..</pre>
</div>

## Section IV. Polygon-Tutorial 1: Deploy the Polygon Yolov5s
**Try Polygon Yolov5s Model by Following** [Polygon-Tutorial 1](https://github.com/XinzeLee/PolygonObjectDetection/blob/main/polygon-yolov5/Polygon-Tutorial1.ipynb)
  1. **Inference**
    <div class="highlight highlight-source-shell position-relative">
      <pre>
      $ python polygon_detect.py --weights polygon-yolov5s-ucas.pt --img 1024 --conf 0.75 \
          --source data/images/UCAS-AOD --iou-thres 0.4 --hide-labels</pre>
      <p align="center">
      <img src="https://user-images.githubusercontent.com/87064748/125021658-ad83eb80-e0ad-11eb-9a61-7824cc09b4ba.png" width="500">
      </p></div>
  2. **Test**
    <div class="highlight highlight-source-shell position-relative">
      <pre>
      $ python polygon_test.py --weights polygon-yolov5s-ucas.pt --data polygon_ucas.yaml \
          --img 1024 --iou 0.65 --task val</pre>
      <p align="center">
      <img src="https://user-images.githubusercontent.com/87064748/125021771-ddcb8a00-e0ad-11eb-8a4e-bef79280c258.png" width="500">
      </p></div>
  3. **Train**
    <div class="highlight highlight-source-shell position-relative">
      <pre>
      $ python polygon_train.py --weights polygon-yolov5s-ucas.pt --cfg polygon_yolov5s_ucas.yaml \
          --data polygon_ucas.yaml --hyp hyp.ucas.yaml --img-size 1024 \
          --epochs 3 --batch-size 12 --noautoanchor --polygon --cache</pre></div>
  4. **Performance**
    <div class="highlight highlight-source-shell position-relative">
      4.1. Confusion Matrix
        <br/>
        <p align="center">
        <img src="https://user-images.githubusercontent.com/87064748/125020614-c25f7f80-e0ab-11eb-9d11-eb2f918b8dee.png" width="500">
        </p>
      4.2. Precision Curve
        <br/>
        <p align="center">
        <img src="https://user-images.githubusercontent.com/87064748/125020716-f044c400-e0ab-11eb-8031-93a2cfede2cc.png" width="500">
        </p>
      4.3. Recall Curve
        <br/>
        <p align="center">
        <img src="https://user-images.githubusercontent.com/87064748/125020755-0488c100-e0ac-11eb-898e-d9399778f4bc.png" width="500">
        </p>
      4.4. Precision-Recall Curve
        <br/>
        <p align="center">
        <img src="https://user-images.githubusercontent.com/87064748/125020796-15393700-e0ac-11eb-8eb9-ade967396fc9.png" width="500">
        </p>
      4.5. F1 Curve
        <br/>
        <p align="center">
        <img src="https://user-images.githubusercontent.com/87064748/125020836-23875300-e0ac-11eb-9b6f-4971c78663a5.png" width="500">
        </p></div>

## Section V. Polygon-Tutorial 2: Transform COCO Dataset to Polygon Labels Using Segmentation
**Transform COCO Dataset to Polygon Labels by Following** [Polygon-Tutorial 2](https://github.com/XinzeLee/PolygonObjectDetection/blob/main/polygon-yolov5/Polygon-Tutorial2.ipynb)
<div class="highlight highlight-source-shell position-relative">
  Transformed Exemplar Figure
  <br/>
  <p align="center">
  <img src="https://user-images.githubusercontent.com/87064748/129332893-19d8d396-0817-450f-bcb5-c1b451b0871a.png" width="500">
  </p>
</div>

## Section VI. Expansion to More Than Four Corners
<div class="highlight highlight-source-shell position-relative">
  <br/>
  <p align="center">
  <img src="https://user-images.githubusercontent.com/87064748/125053398-a2de4c00-e0d7-11eb-8dd2-fbe803a7b428.jpg" width="600">
  </p>
</div>

## Section VII. References
* [ultralytics/yolov5](https://github.com/ultralytics/yolov5)
* [NVIDIA/retinanet-examples](https://github.com/NVIDIA/retinanet-examples)
* [ming71/yolov3-polygon](https://github.com/ming71/yolov3-polygon)

## Section VIII. Contributions
* [tak-s](https://github.com/tak-s) : problems of [disjointed segmentations](https://github.com/XinzeLee/PolygonObjectDetection/issues/2) for some connected objects are solved.
* [moshe](https://github.com/18112330636) : problems of [for loops in def order_corners & ambiguity when diagonal of quadrilateral is parallel to x-axis](https://github.com/XinzeLee/PolygonObjectDetection/issues/20) are solved.
