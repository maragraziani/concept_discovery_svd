import torchvision
import ctypes
import torchvision
import pandas
import os
import matplotlib.pyplot as plt
import PIL.Image as Image
import numpy as np
import torchvision
import torch
import json
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

transform = torchvision.transforms.Compose([
 torchvision.transforms.Resize(299),
 torchvision.transforms.CenterCrop(299),
 torchvision.transforms.ToTensor()
])

transform_normalize = torchvision.transforms.Normalize(
     mean=[0.485, 0.456, 0.406],
     std=[0.229, 0.224, 0.225]
 )

def get_label(idx_to_labels,id_):
    for k in list(idx_to_labels.keys()):
        if id_ in idx_to_labels[k]:
            return k

def get_ilsvrc_dir(path):
    return path.split('/')[-2]
    
def get_dataset(dname, source=None, train=True):
    dataset=dname
    count=0
    if dname=='ilsvrc12':
        if source==None:
            source = f'/mnt/nas4/datasets/ToReadme/{dataset}/'
        if train:
            train_files_path = (source+'/train')
        labels_path = os.getenv("HOME") + '/.torch/models/imagenet_class_index.json'
        lim=10

    elif dname=='imagenette':
        if not source:
            source = f'./{dataset}2-320/'
            train_files_path=(source+'train')
        labels_path = os.getenv("HOME") + '/.torch/models/imagenet_class_index.json'
        lim=500
            
    elif dname=='imagewoof':
        if not source:
            source = f'./{dataset}2-320/'
            train_files_path=(source+'train')
        labels_path = os.getenv("HOME") + '/.torch/models/imagenet_class_index.json'
        lim=500
    
    with open(labels_path) as json_data:
        idx_to_labels = json.load(json_data)
            
    paths=[]
    labels = {}
    k=0
    image_counter=0
    for dir_ in os.listdir(source+'/train'):
        count+= len(os.listdir(source+'/train'+'/'+dir_))
        labels[dir_]=get_label(idx_to_labels, dir_)
        k+=1
        image_counter=0
        for p in os.listdir(source+'/train'+'/'+dir_):
            if image_counter < lim:
                paths.append(source+'/train'+'/'+dir_+'/'+p)
                image_counter+=1

    count = int(len(paths))
    
    y=np.zeros(count)

    for i in range(count):   
            dir_ = get_ilsvrc_dir(paths[i])
            y[i] = labels[dir_]

#     x=np.zeros((count,299,299,3))
#     y=np.zeros(count)

#     for i in range(0,count,1):
#         img = Image.open(paths[i])
#         x[i] = np.asarray(img.resize((299,299),Image.ANTIALIAS).convert("RGB"), dtype=np.float32)

#     for i in range(count):   
#         dir_ = get_ilsvrc_dir(paths[i])
#         y[i] = labels[dir_]
    return paths, count, y, idx_to_labels
