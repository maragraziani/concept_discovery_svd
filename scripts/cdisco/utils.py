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

