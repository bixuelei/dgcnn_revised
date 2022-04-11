# Motor_segmentation_net
a project of deep learning architecture to forecast the categories of each point of a clamping and motor scene.

# Environments Requirement
CUDA = 10.2

Python = 3.7.0

PyTorch = 1.6

The mentioned API are the basic API. In the training  process,if there is warning that some modul is missing. you could direct use pip install to install specific modul.
For example, if there is no installation of open3d in the process of running of the script, you could direct use pip install open3d to install conrresponding toolkits

# How to run

## Training the pretraining model

You can use below command line to run the pretraining script and gain the pretraining model:
```
CUDA_VISIBLE_DEVICES=0 python train_semseg.py --batch_size 16 --test_batch_size 16 --npoints 2048 --epoch 50 --model My_Net5 --change without_rotation_16_2048_50  --exp training_125 --bolt_weight 1  --root /home/ies/bi/data/training_125
```
Explanation of every important parameter
* train_semseg.py: choose of which script will be run
* CUDA_VISIBLE_DEVICES: set the visible gpu 
* model: choose of which model will be used to train a pretraining model
* change: give the information of a specific experiment(here without rotation means that i dont use the STN Net, 16 represent bacht_size, 2048 represents points of every training unit, 50 represents the epoch
* exp: the paremeter means that i training the net with dataset(i generate) with size of 125 motors scene
* root: the root of training dataset

## Train the finetune model
You can use below command line to run the finetune script and gain the training result:
```
CUDA_VISIBLE_DEVICES=0 python train_semseg.py --batch_size 16 --test_batch_size 16 --npoints 2048 --epoch 300 --model My_Net3 --change without_rotation_16_2048_50  --exp training_125 --bolt_weight 1 --finetune True  --root /home/ies/bi/data/previous_finetune
```
Here we have another parameter,finetune,which is set by default as False, if we set it as True, we will train the finetune model.

## Test the finetune model
You can use below command line to run the test finetune script and gain the test result:
```
CUDA_VISIBLE_DEVICES=0 python train_semseg_rotation.py --batch_size 16 --test_batch_size 16 --npoints 2048 --model dgcnn_rotate_conv --change allaround_STN_conv   --exp training_125 --bolt_weight 1 --root /home/ies/bi/data/previous_test --finetune True --eval True
```
Explanation of every important parameter
* train_semseg_rotation.py: here we choose train_semseg_rotation.py, this means we add STN Net to whole architecture

# Inportant Info
If you want to debug the script, you should change the parameter manually. For PCT_Nico, it needs huge storage of gpu. It is advible to set batch_size as 2 and to set n_points as 1024, if you want to get to know how this net works.




