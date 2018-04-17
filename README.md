# 2D-Face-Alignment
The implementation of "Face Alignment at 3000 FPS via Regressing Local Binary Features" on CVPR 2014.
***

### Requirements
* Visual Studio 2015+ (Below without Tests)
* OpenCV 3.3.0+ (Below without Tests)

### Result

![Result](/gallary/gallary_1.png)

Blue rectangle is the bounding box of face, red points are 68 landmarks

### Trainset

The trainset Helen can be downloaded from ![Here](https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/)

### Training

##### Code Settings

In main.cpp, comment/uncomment the main function as following:

![Train](/gallary/main_training.png)

##### Config Settings

Set the configs files as following:

* config/helen.cfg

'''
ImgFolderPath = "E:\helen\trainset\"
ImgNameFile = "config\helen_filenames.cfg"
'''

ImgFolderPath: Folder which store the trainset images

ImgNameFile: File which store the filenames of the trainset images

* config/train.cfg

'''
feature_num = 500
landmark_num = 68
stage_num = 6
tree_depth = 5
tree_num_per_forest = 12
forest_overlap = 0.3
local_region_size = { 0.29, 0.21, 0.16, 0.12, 0.08, 0.04 }
special_point_id = { 28, 8, 36, 45 }
'''

feature_num: The randomly picked point pairs generated around the landmark

landmark_num: The number of landmark in the dataset

stage_num: The stage number in cascade training

tree_depth: The depth of a single tree in the random forest when generating linear binary feature

tree_num_per_forest: The number of trees in the random forest

forest_overlap: The overlap argument when generating data for random forest

local_region_size: The region size to get features in each training stage

special_point_id: The landmark id of up, down, left, right, used for coarse alignment

##### Start Training

Just run and wait (about 1h on Helen dataset and Intel core i7 4710hq)

### Running

##### Code Settings

In main.cpp, comment/uncomment the main function as following:

![Train](/gallary/main_running.png)

The predictImage function's first argument is the image path

The predictImage function can also have a second argument which has a type cv::Rect and referred to the bounding box of face

If without this second argument, the program will use OpenCV's default bounding box extracting function

##### Models

This program will use models in the following list:

* model/random_forest/*

* model/regressor/*

* S_0.mdl

* haarcascade_frontalface_alt2.xml

The first three models will be generated in the training stage, if you want to use my model, just unzip the regressor.zip in model/

### Reference

* Face Alignment at 3000 FPS via Regressing Local Binary Features. CVPR 2014. Ren et al.
