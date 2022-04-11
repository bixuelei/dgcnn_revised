"""
@Author: bixuelei
@Contact: bxueleibi@gmial.com
@File: dataloader.py
@Time: 2022/1/16 3:49 PM
"""

import os
import numpy as np
import random
from numpy.random import choice
from tqdm import tqdm           #used to display the circulation position, to see where the code is running at
from torch.utils.data import Dataset
from sklearn.neighbors import NearestNeighbors


####### normalize the point cloud############
def pc_normalize(pc):
    centroid=np.mean(pc,axis=0)
    pc=pc-centroid
    max_distance=np.sqrt(np.max(np.sum(pc**2,axis=1)))
    pc=pc/max_distance
    return pc


def Get_ObjectID(x) :    #get all kinds of ObjectID from numpy file

    dic = []
    for i in range(x.shape[0]):
        if x[i][6] not in dic:
            dic.append(x[i][6])

    return dic


def densify_blots(patch_motor):
    add=[]
    for i in range(len(patch_motor)):
        if (patch_motor[i][6]==5):
            add.append(patch_motor[i])
    add=np.array(add)
    twonn=NearestNeighbors(n_neighbors=2,algorithm='ball_tree').fit(add[:,0:3])
    _,indices=twonn.kneighbors(add[:,0:3])
    inter=[]
    for i in range(indices.shape[0]):
        interpolation=np.zeros(7)
        interpolation[3:7]=add[0][3:7]
        #if the bolt points are closest to eachonter
        if(indices[indices[i][1]][1]==i):
            interpolation[0:3]=add[i][0:3]+(add[indices[i][1]][0:3]-add[i][0:3])/3
            inter.append(interpolation)
        else:
            interpolation[0:3]=add[i][0:3]+(add[indices[i][1]][0:3]-add[i][0:3])/2
            inter.append(interpolation)
    patch_motor=np.concatenate((patch_motor,inter),axis=0)
    return patch_motor


class MotorDataset(Dataset):
    def __init__(self,split='train',data_root='directory to training data',num_points=4096,bolt_weight=1,test_area='Validation',block_size=1.0,sample_rate=1.0,transform=None):
        super().__init__()
        self.num_points=num_points
        self.block_size=block_size
        self.transform=transform      
        motor_list=sorted(os.listdir(data_root))        #list all subdirectory    
        #motor_filter=[motor for motor in motor_list if 'Type' in motor]     #filter all the file, whose name has no Type      
        if split == 'train':        #load training files or validation files
            motor_positions=[motor for motor in motor_list if '{}'.format(test_area) not in motor]
        else:
            motor_positions=[motor for motor in motor_list if '{}'.format(test_area) in motor]

        
        ######################load the np file###################################################    
        
        self.motors_points=[]       #initial object_motor_points and object_motor_lables
        self.motors_labels=[]      
        num_points_eachmotor=[]     #initial a list to count the num of points for each motor   
        label_num_eachtype=np.zeros(6)      #initial a array to count how much points is there for each type
        for motor_position in tqdm(motor_positions,total=len(motor_positions)):
            motor_directory=os.path.join(data_root,motor_position)
            motor_data=np.load(motor_directory)
            motor_data=densify_blots(motor_data)
            motor_points=motor_data[:,0:6]
            motor_labels=motor_data[:,6]            #result is a np array           
            num_eachtype_in_one_motor,_=np.histogram(motor_labels,bins=6,range=(0,6))       #count how much points is there for each type(usage of np.histotram)
            label_num_eachtype+=num_eachtype_in_one_motor
            self.motors_points.append(motor_points)
            self.motors_labels.append(motor_labels)
            num_points_eachmotor.append(motor_labels.size)
        #id=Get_ObjectID(motor_data)
        #print(id)
        ############################################################################################


        ###########according to lable_num_eachmotor and bolt_weight, caculate the labelweights######
        label_num_eachtype[-1]/=bolt_weight
        labelweights=label_num_eachtype/np.sum(label_num_eachtype)
        labelweights=np.power(np.max(labelweights)/labelweights,1/3)
        self.labelweight=labelweights/np.sum(labelweights)
        ############################################################################################


        #############caculate the index for choose of points from the motor according to the number of points of a specific motor#########      
        sample_prob_eachmotor=num_points_eachmotor/np.sum(num_points_eachmotor)     #probability for choosing from a specific motor      
        num_interation=sample_rate*np.sum(num_points_eachmotor)/self.num_points     #num_of_all to choose npoints cloud       
        self.motors_indes=[]        #initial motors_indes list    
        for index in range(len(num_points_eachmotor)):      #allocate the index according to probability
            sample_times_to_onemotor=int(round(sample_prob_eachmotor[index]*num_interation))
            motor_indes_onemotor=[index]*sample_times_to_onemotor
            self.motors_indes.extend(motor_indes_onemotor)
        ####################################################################################################################################
         
                
    def __getitem__(self,index):
        
        points=self.motors_points[self.motors_indes[index]][:,0:3]      #initialize the parameter
        labels=self.motors_labels[self.motors_indes[index]]
        n_points=points.shape[0]   
        ########################have a randow choose of points from points cloud#######################
        choice=np.random.choice(n_points,self.num_points,replace=True)
        chosed_points=points[choice,:]
        chosed_labels=labels[choice]
        ###############################################################################################

        return chosed_points,chosed_labels

    def __len__(self):                                                                            
        return len(self.motors_indes) 


class MotorDataset_validation(Dataset):
    def __init__(self,split='train',data_root='directory to training data',num_points=4096,bolt_weight=1,test_area='Validation',block_size=1.0,sample_rate=1.0,transform=None):
        super().__init__()
        self.num_points=num_points
        self.block_size=block_size
        self.transform=transform      
        motor_list=sorted(os.listdir(data_root))        #list all subdirectory    
        #motor_filter=[motor for motor in motor_list if 'Type' in motor]     #filter all the file, whose name has no Type      
        if split == 'train':        #load training files or validation files
            motor_positions=[motor for motor in motor_list if '{}'.format(test_area) not in motor]
        else:
            motor_positions=[motor for motor in motor_list if '{}'.format(test_area) in motor]

        
        ######################load the np file###################################################    
        
        self.motors_points=[]       #initial object_motor_points and object_motor_lables
        self.motors_labels=[]      
        num_points_eachmotor=[]     #initial a list to count the num of points for each motor   
        label_num_eachtype=np.zeros(6)      #initial a array to count how much points is there for each type
        for motor_position in tqdm(motor_positions,total=len(motor_positions)):
            motor_directory=os.path.join(data_root,motor_position)
            motor_data=np.load(motor_directory)
            motor_points=motor_data[:,0:6]
            motor_labels=motor_data[:,6]            #result is a np array           
            num_eachtype_in_one_motor,_=np.histogram(motor_labels,bins=6,range=(0,6))       #count how much points is there for each type(usage of np.histotram)
            label_num_eachtype+=num_eachtype_in_one_motor
            self.motors_points.append(motor_points)
            self.motors_labels.append(motor_labels)
            num_points_eachmotor.append(motor_labels.size)
        ############################################################################################


        ###########according to lable_num_eachmotor and bolt_weight, caculate the labelweights######
        label_num_eachtype[-1]/=bolt_weight
        labelweights=label_num_eachtype/np.sum(label_num_eachtype)
        labelweights=np.power(np.max(labelweights)/labelweights,1/3)
        self.labelweight=labelweights/np.sum(labelweights)
        ############################################################################################


        #############caculate the index for choose of points from the motor according to the number of points of a specific motor#########      
        sample_prob_eachmotor=num_points_eachmotor/np.sum(num_points_eachmotor)     #probability for choosing from a specific motor      
        num_interation=sample_rate*np.sum(num_points_eachmotor)/self.num_points     #num_of_all to choose npoints cloud       
        self.motors_indes=[]        #initial motors_indes list    
        for index in range(len(num_points_eachmotor)):      #allocate the index according to probability
            sample_times_to_onemotor=int(round(sample_prob_eachmotor[index]*num_interation))
            motor_indes_onemotor=[index]*sample_times_to_onemotor
            self.motors_indes.extend(motor_indes_onemotor)
        ####################################################################################################################################
         
                
    def __getitem__(self,index):
        
        points=self.motors_points[self.motors_indes[index]][:,0:3]      #initialize the parameter
        labels=self.motors_labels[self.motors_indes[index]]
        n_points=points.shape[0]   
        #points=rotate(points)   
        #normalized_points=pc_normalize(points)      #normalize the points


        ########################have a randow choose of points from points cloud#######################
        choice=np.random.choice(n_points,self.num_points,replace=True)
        chosed_points=points[choice,:]
        chosed_labels=labels[choice]
        ###############################################################################################

        return chosed_points,chosed_labels

    def __len__(self):                                                                            
        return len(self.motors_indes) 


class MotorDataset_patch(Dataset):
    def __init__(self,split='train',data_root='directory to training data',num_points=4096,bolt_weight=1,test_area='Validation',block_size=1.0,sample_rate=1.0,transform=None):
        super().__init__()
        self.num_points=num_points
        self.block_size=block_size
        self.transform=transform      
        motor_list=sorted(os.listdir(data_root))        #list all subdirectory    
        motor_filter=[motor for motor in motor_list if 'Type' in motor]     #filter all the file, whose name has no Type      
        if split == 'train':        #load training files or validation files
            motor_positions=[motor for motor in motor_filter if '{}'.format(test_area) not in motor]
        else:
            motor_positions=[motor for motor in motor_filter if '{}'.format(test_area) in motor]

        
        ######################load the np file###################################################    
        
        self.motors_points=[]       #initial object_motor_points and object_motor_lables
        self.motors_labels=[]      
        self.interation_times_eachmotor = []
        label_num_eachtype=np.zeros(6)      #initial a array to count how much points is there for each type
        for motor_position in tqdm(motor_positions,total=len(motor_positions)):
            motor_directory=os.path.join(data_root,motor_position)
            motor_data=np.load(motor_directory)
            motor_points=motor_data[:,0:3]
            motor_points=pc_normalize(motor_points)      #normalize the points
            motor_labels=motor_data[:,6]            
            motor_points_labels=[] 

            ################       judge if current motor points cloud can just include integel             ########
            ###############     times of sub points_clouds of num_point,if not, patch it to integel one     ########
            current_motor_size=motor_points.shape[0]
            if current_motor_size % self.num_points !=0:
                num_add_points=self.num_points-(current_motor_size % self.num_points)
                choice=np.random.choice(current_motor_size,num_add_points,replace=True)     #pick out some points from current cloud to patch up the current cloud
                add_points=motor_points[choice,:]
                motor_points=np.vstack((motor_points,add_points))
                add_labels=motor_labels[choice]
                motor_labels=np.hstack((motor_labels,add_labels))
            #########################################################################################################
            #########################################################################################################

            motor_points_labels=np.hstack((motor_points,motor_labels.reshape((motor_labels.size,1))))       #merge the labels and points in order to schuffle it
            np.random.shuffle(motor_points_labels)
            motor_points=motor_points_labels[:,0:3]     #get the schuffled points and lables
            motor_labels=motor_points_labels[:,3]
            self.interation_times_eachmotor.append(motor_labels.size/self.num_points)       #record how money 4096 points could be taken out for one motor points cloud after patch
            num_eachtype_in_one_motor,_=np.histogram(motor_labels,bins=6,range=(0,6))       #count how much points is there for each type(usage of np.histotram)
            label_num_eachtype+=num_eachtype_in_one_motor
            self.motors_points.append(motor_points)
            self.motors_labels.append(motor_labels)
        ############################################################################################


        ###########according to lable_num_eachmotor and bolt_weight, caculate the labelweights######
        label_num_eachtype[-1]/=bolt_weight
        labelweights=label_num_eachtype/np.sum(label_num_eachtype)
        labelweights=np.power(np.max(labelweights)/labelweights,1/3)
        self.labelweight=labelweights/np.sum(labelweights)
        ############################################################################################


        #############caculate the index for choose of points from the motor according to the number of points of a specific motor#########      
        self.motors_indes=[]        #initial motors_indes list    
        for index in range(len(self.interation_times_eachmotor)):      #allocate the index according to probability
            motor_indes_onemotor=[index]*int(self.interation_times_eachmotor[index])
            self.motors_indes.extend(motor_indes_onemotor)
        ####################################################################################################################################


        #####################################   set the dictionary for dataloader index according to motors_points structure        ########
        self.dic_block_accumulated_per_motors={}
        key=0
        for index in range(len(self.interation_times_eachmotor)):
            if index!=0:
                key=key+self.interation_times_eachmotor[index-1]
            for num_clouds_per_motor in range(int(self.interation_times_eachmotor[index])):
                self.dic_block_accumulated_per_motors[int(key+num_clouds_per_motor)]=num_clouds_per_motor
        ####################################################################################################################################


    def __getitem__(self,index):   
        points=self.motors_points[self.motors_indes[index]]      #initialize the points cloud for each motor
        labels=self.motors_labels[self.motors_indes[index]]
        

        sequence=np.arange(self.num_points)
        chosed_points=points[self.num_points*self.dic_block_accumulated_per_motors[index]+sequence,:]       #ensure all the points could be picked out by the ways of patch
        chosed_labels=labels[self.num_points*self.dic_block_accumulated_per_motors[index]+sequence]    
        return chosed_points,chosed_labels

    def __len__(self):                                                                            
        return len(self.motors_indes) 