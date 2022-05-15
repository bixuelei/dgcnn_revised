# Motor_segmentation_net
a project of deep learning architecture to forecast the categories of each point of a clamping and motor scene.

# Environments Requirement
CUDA = 10.2

Python = 3.7.0

PyTorch = 1.6

The mentioned API are the basic API. In the training  process,if there is warning that some modul is missing. you could direct use pip install to install specific modul.
For example, if there is no installation of open3d in the process of running of the script, you could direct use pip install open3d to install conrresponding toolkits

# The Models We have

## Mdoels without rotation as augmentation

* dgcnn: the original dgcnn model
* dgcnn_conv: dgcnn with con1d as additional aggregation
* dgcnn_self: dgcnn with one layer of self-attention
* dgcnn_3_layers_self: dgcnn with three layer of self-attention 
* dgcnn_self_conv: dgcnn with one layer of self-attention and con1d as additional aggregation
* PCT: PCT net

## Mdoels with rotation as augmentation

* dgcnn_rotate: the original dgcnn model integrated with T-Net
* dgcnn_conv_rotate: dgcnn with con1d as additional aggregation integrated with T-Net
* dgcnn_self_rotate: dgcnn with one layer of self-attention integrated with T-Net
* dgcnn_3_layers_self_rotate: dgcnn with three layer of self-attention integrated with T-Net
* dgcnn_self_conv_rotate: dgcnn with one layer of self-attention and con1d as additional aggregation integrated with T-Net
* PCT_rotate: PCT net integrated with T-Net

# How to run

## Training the pretraining model without rotation

You can use below command line to run the pretraining script and gain the pretraining model:
```
CUDA_VISIBLE_DEVICES=0 python train_semseg.py --batch_size 16 --test_batch_size 16 --npoints 2048 --epoch 50 --model dgcnn --change without_rotation_16_2048_50  --exp training_125 --bolt_weight 1  --root /home/ies/bi/data/training_125
```
Explanation of every important parameter
* train_semseg.py: choose of which script will be run
* CUDA_VISIBLE_DEVICES: set the visible gpu 
* model: choose of which model will be used to train a pretraining model
* change: give the information of a specific experiment(here without rotation means that i dont use the  T-Net, 16 represent bacht_size, 2048 represents points of every training unit, 50 represents the epoch
* exp: the paremeter(training_125) means that i training the net with dataset that includes  125 motors scenes
* root: the root of training dataset

## Training the pretraining model with rotation

You can use below command line to run the pretraining script and gain the pretraining model:
```
CUDA_VISIBLE_DEVICES=0 python train_semseg_rotation.py --batch_size 16 --test_batch_size 16 --npoints 2048 --epoch 50 --model dgcnn --change rotation_16_2048_50  --exp training_125 --bolt_weight 1  --root /home/ies/bi/data/training_125
```

## Train the finetune model without rotation
You can use below command line to run the finetune script and gain the training result:
```
CUDA_VISIBLE_DEVICES=0 python train_semseg.py --batch_size 16 --test_batch_size 16 --npoints 2048 --epoch 300 --model dgcnn --change without_rotation_16_2048_50  --exp training_125 --bolt_weight 1 --finetune True  --root /home/ies/bi/data/previous_finetune
```
Here we have another parameter,finetune,which is set by default as False, if we set it as True, we will train the finetune model.

## Train the finetune model with rotation
You can use below command line to run the finetune script and gain the training result:
```
CUDA_VISIBLE_DEVICES=0 python train_semseg_rotation.py --batch_size 16 --test_batch_size 16 --npoints 2048 --epoch 300 --model dgcnn --change rotation_16_2048_50  --exp training_125 --bolt_weight 1 --finetune True  --root /home/ies/bi/data/previous_finetune
```
<<<<<<< HEAD

## Test the finetune model without rotation
You can use below command line to run the test finetune script and gain the test result:
```
CUDA_VISIBLE_DEVICES=0 python train_semseg.py --batch_size 16 --test_batch_size 16 --npoints 2048 --model dgcnn --change without_rotation_16_2048_50   --exp training_125 --bolt_weight 1 --root /home/ies/bi/data/previous_test --finetune True --eval True
```
The change and model parameter are exactly same with the paramater you set in the finetune process.
=======

## Test the finetune model without rotation
You can use below command line to run the test finetune script and gain the test result:
```
CUDA_VISIBLE_DEVICES=0 python train_semseg.py --batch_size 16 --test_batch_size 16 --npoints 2048 --model dgcnn --change without_rotation_16_2048_50   --exp training_125 --bolt_weight 1 --root /home/ies/bi/data/previous_test --finetune True --eval True
```
The change and model parameter are exactly same with the paramater you set in the finetune process.

## Test the finetune model with rotation
You can use below command line to run the test finetune script and gain the test result:
```
CUDA_VISIBLE_DEVICES=0 python train_semseg_rotation.py --batch_size 16 --test_batch_size 16 --npoints 2048 --model dgcnn --change rotation_16_2048_50   --exp training_125 --bolt_weight 1 --root /home/ies/bi/data/previous_test --finetune True --eval True
```
The change and model parameter are exactly same with the paramater you set in the finetune proces
>>>>>>> 86a21d4d8e2e1e4d418504bab0991f743bc6246e

## Test the finetune model with rotation
You can use below command line to run the test finetune script and gain the test result:
```
CUDA_VISIBLE_DEVICES=0 python train_semseg_rotation.py --batch_size 16 --test_batch_size 16 --npoints 2048 --model dgcnn --change rotation_16_2048_50   --exp training_125 --bolt_weight 1 --root /home/ies/bi/data/previous_test --finetune True --eval True
```
The change and model parameter are exactly same with the paramater you set in the finetune proces
