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




count=0
paths=[]
labels = {}
k=0
image_counter=0

dataset='imagewoof'
source = f'./{dataset}2-320/'
train_files_path = (source+'/train')
labels_path = os.getenv("HOME") + '/.torch/models/imagenet_class_index.json'
output_log = open('outputs/log.txt', 'w')
print(f"source folder: {source},\n train data folder: {train_files_path}, \n labels data path: {labels_path}")

SAVEFOLD=f'outputs/{dataset}'
if not os.path.exists(SAVEFOLD):
    os.mkdir(SAVEFOLD)


def relu_mul(x):
    out = x * (x > 0)
    return out
def log(text, file=output_log):
    output_log.write(text+'\n')
    return
def get_label(idx_to_labels,id_):
    for k in list(idx_to_labels.keys()):
        if id_ in idx_to_labels[k]:
            return k


with open(labels_path) as json_data:
    idx_to_labels = json.load(json_data)


for dir_ in os.listdir(source+'/train'):
    #paths.append=os.listdir(source/'train'/dir)[1]
    count+= len(os.listdir(source+'/train'+'/'+dir_))
    labels[dir_]=get_label(idx_to_labels, dir_)
    k+=1
    image_counter=0
    for p in os.listdir(source+'/train'+'/'+dir_):
        if image_counter < 500:
            paths.append(source+'/train'+'/'+dir_+'/'+p)
            image_counter+=1

count = len(paths)
x=np.zeros((count,299,299,3))
y=np.zeros(count)


for i in range(count):
    img = Image.open(paths[i])
    x[i] = np.asarray(img.resize((299,299),Image.ANTIALIAS).convert("RGB"), dtype=np.float32)
    
for i in range(count):   
    dir_ = paths[i].split('/')[4]
    y[i] = labels[dir_]
    

transform = torchvision.transforms.Compose([
 torchvision.transforms.Resize(299),
 torchvision.transforms.CenterCrop(299),
 torchvision.transforms.ToTensor()
])

transform_normalize = torchvision.transforms.Normalize(
     mean=[0.485, 0.456, 0.406],
     std=[0.229, 0.224, 0.225]
 )
classes = np.unique(y)


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"
try:
    del model
except:
    pass
model = torch.hub.load('pytorch/vision:v0.9.0', 'inception_v3', pretrained=True)
model.eval()
train_nodes, eval_nodes = torchvision.models.feature_extraction.get_graph_node_names(torchvision.models.inception_v3())
#print(train_nodes)

SAVEFOLD=f'outputs/{dataset}'
#layer='cat_15'
#layer='cat_9'
#layer='cat_8'
layer='Conv2d_2b_3x3.relu'
return_nodes={f'{layer}': 'conv', 'avgpool':'avgpool', 'fc':'fc'
             }
model = torchvision.models.feature_extraction.create_feature_extractor(model, return_nodes=return_nodes)

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
inputs=torch.zeros((8,3,299,299), dtype=torch.float).to(device)
model.to(device)
outs=model(inputs)
dim_c = outs['conv'].shape[1]
dim_w = outs['conv'].shape[2]
dim_h = outs['conv'].shape[3]


batch_size=32
tot_acc = 0
i=0
batch_start=0


#gradients_wrt_conv_layer = np.zeros((len(y), dim_c,dim_w,dim_h), dtype=np.float32)
embeddings=np.zeros((len(y),2048))
gradients = np.zeros((len(y), 2048))
predictions = np.zeros((len(y), 1000))
conv_embeddings=np.zeros((len(y),dim_c))
gradients_wrt_conv_layer = np.zeros((len(y), dim_c,dim_w,dim_h), dtype=np.float32)
conv_maps = np.zeros((len(y),dim_c,dim_w,dim_h))


print(f"embeddings shape: {embeddings.shape}")
print(f"gradients shape: {gradients.shape}")
print(f"predictions shape: {predictions.shape}")

while batch_start+batch_size < len(y)+batch_size: 
    # preprocessing the inputs 
    inputs = torch.stack([transform_normalize(transform(Image.open(paths[i]).convert("RGB"))) for i in range(batch_start, min(batch_start+batch_size, len(y)))])
    inputs = inputs.clone().detach().requires_grad_(True)
    batch_y=y[batch_start:min(batch_start+batch_size, len(y))]
    
    # transfering to GPU
    inputs=inputs.to(device)
    model=model.to(device)
    
    # inference pass
    outs = model(inputs)
    
    # extracting embeddings
    # note: convolutional outputs should be avg pooled for this to actually make sense
    pooled_embeddings=torch.nn.functional.adaptive_avg_pool2d(outs['conv'], (1, 1))
    conv_embeddings[batch_start:min(batch_start+batch_size, len(y)),:]=pooled_embeddings[:,:,0,0].cpu().detach().numpy()
    embeddings[batch_start:min(batch_start+batch_size, len(y)),:]=outs['avgpool'][:,:,0,0].cpu().detach().numpy()
    
    # computing prediction loss
    loss = torch.nn.CrossEntropyLoss()
    pred = outs['fc']
    len_=pred.shape[0]
    target=np.zeros((len_, 1000))
    for i in range(len(pred)):
        target[i,int(batch_y[i])]=1.
    target=torch.tensor(target, requires_grad=True).to(device)
    outloss = loss(pred, target)
    
    # Storing predictions
    softmaxf = torch.nn.Softmax(dim=1)
    predictions[batch_start:min(batch_start+batch_size, len(y)),:]=softmaxf(pred).detach().cpu()
     
    
    # Computing the gradients and storing them 
    grads_wrt_conv = torch.autograd.grad(outloss, outs['conv'], retain_graph=True)[0]
    gradients_wrt_conv_layer[batch_start:min(batch_start+batch_size, len(y)),:,:,:] = grads_wrt_conv[:,:,:,:].cpu()
    conv_maps[batch_start:min(batch_start+batch_size, len(y)),:,:,:] = outs['conv'].cpu().detach()
    
    grads = torch.autograd.grad(outloss, outs['avgpool'], retain_graph=True)[0]
    gradients[batch_start:min(batch_start+batch_size, len(y)),:] = grads[:,:,0,0].cpu()
    
    batch_start += batch_size


#u, s, vh = np.linalg.svd(embeddings)
#print(f"SVD decomposition done. U: {u.shape}, S: {s.shape}, VT: {vh.shape}")
#log(f"SVD decomposition done. U: {u.shape}, S: {s.shape}, VT: {vh.shape}")
print(f"gradients shape {gradients.shape}, conv_embs shape {conv_embeddings.shape}, conv_maps.shape {conv_maps.shape}")


try:
    os.mkdir(f"{SAVEFOLD}/{layer}/")
except:
    print("Maybe the directory already exists? ")
SAVEFOLD=f"{SAVEFOLD}/{layer}/"

import sklearn.metrics
acc = sklearn.metrics.accuracy_score(y[:],np.argmax(predictions,axis=1)[:])#, labels=np.arange(1000))
print(f"Test accuracy: {acc}")
classes_by_names=[]
for c in classes:
    classes_by_names.append(idx_to_labels[str(int(c))])
cf_matrix = sklearn.metrics.confusion_matrix(y[:],np.argmax(predictions,axis=1)[:], labels=np.arange(1000))
plt.rcParams['figure.figsize']=(10,10)

small_cf_matrix=np.zeros((10,10))

i=0
for c in classes:
    small_cf_matrix[i,:]=np.asarray([cf_matrix[int(c),int(j)] for j in classes])
    i+=1
fig, ax = plt.subplots()
ax.matshow(small_cf_matrix)
for (i, j), z in np.ndenumerate(small_cf_matrix):
    ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))

plt.xticks(np.arange(10),classes_by_names, rotation=90)
plt.yticks(np.arange(10),classes_by_names)
plt.title(f"Acc: {acc}")
plt.savefig(f"outputs/imagewoof/cmatrix.png")











