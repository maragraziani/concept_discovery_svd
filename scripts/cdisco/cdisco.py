import scipy.ndimage
import numpy as np
import sys
from datasets import transform
import torch
import torchvision 
import ctypes
import pandas
import os
import matplotlib.pyplot as plt
import PIL.Image as Image
import numpy as np
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

# ImageNet train accuracy 0.92
def cdisco(conv_maps, gradients_wrt_conv_layer, predictions, classes):
    def relu(data):
        return data * (data>0)
    conv_embeddings=np.mean(conv_maps, axis=(2,3))
    pu, ps, pvh = np.linalg.svd(conv_embeddings)
    
    rg=relu(gradients_wrt_conv_layer)
    
    g_k = np.multiply(rg, conv_maps)
    pooled_g = np.mean(g_k, axis=(2,3))
    projected_g = np.dot(pooled_g, pvh)

    z = {}
    mean_gk={}
    mean_gnotk={}
    std_gnotk={}
    
    for k in classes:
        idxs_of_class_k = np.asarray(np.argwhere(np.argmax(predictions, axis=1)==k)).ravel() # predicitons = model predictions
        all_the_rest = np.asarray(np.argwhere(y!=k)).ravel()

        sel_info=np.asarray([projected_g[n] for n in idxs_of_class_k])
        mean_gk[int(k)] = np.mean(sel_info, axis=0) #std

        rest_info = np.asarray([projected_g[n] for n in all_the_rest])
        mean_gnotk[int(k)] = np.mean(rest_info, axis=0)
        std_gnotk[int(k)] = np.std(rest_info, axis=0)
        z[int(k)] = mean_gk[int(k)] - mean_gnotk[int(k)]
        z[int(k)] = z[int(k)] / std_gnotk[int(k)]


    class_reordered_eigenvalues={}
    class_concept_candidates = {}

    for k in classes:
        class_reordered_eigenvalues[int(k)] = z[int(k)]
        class_concept_candidates[int(k)] = np.asarray(class_reordered_eigenvalues[int(k)]).argsort()[::-1]
    return class_concept_candidates, pvh

def cdisco_concepts_list(class_concept_candidates,classes,limit=1):
    if limit==1:
        first_candidates_only = {}
        concepts=set()
        for k in classes:
            concepts.add(class_concept_candidates[k][0])
        for c in concepts:
            first_candidates_only[c]=[]
        for k in classes:
            first_candidates_only[class_concept_candidates[k][0]].append(k)
        return concepts, first_candidates_only
    

#def cdisco(model, layer, dataset):
    
def get_model_state(model, paths, y, dim_c, dim_w, dim_h, SAVEFOLD=''):
    batch_size=32
    tot_acc = 0
    i=0
    batch_start=0
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    embeddings=np.zeros((len(y),2048))
    gradients = np.zeros((len(y), 2048))
    predictions = np.zeros((len(y), 1000))
    conv_embeddings=np.zeros((len(y)
                              ,dim_c))
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


    print(f"gradients shape {gradients.shape}, conv_embs shape {conv_embeddings.shape}, conv_maps.shape {conv_maps.shape}")
    """
    SAVE INTERMEDIATE RESULTS
    """
    np.save(f"{SAVEFOLD}/predictions.npy", predictions)
    np.save(f"{SAVEFOLD}/gradients_wrt_conv_layer.npy", gradients_wrt_conv_layer)
    np.save(f"{SAVEFOLD}/conv_maps.npy", conv_maps)

def evaluate(predictions, y, dataset=''):
    import sklearn.metrics
    acc = sklearn.metrics.accuracy_score(y[:],np.argmax(predictions,axis=1)[:])#, labels=np.arange(1000))
    print(f"Train accuracy: {acc}")
    classes_by_names=[]
    for c in classes:
        classes_by_names.append(idx_to_labels[str(int(c))])
    cf_matrix = sklearn.metrics.confusion_matrix(y[:],np.argmax(predictions,axis=1)[:], labels=np.arange(1000))
    plt.rcParams['figure.figsize']=(10,10)

    if dataset=='imagenette':
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
        plt.savefig(f"outputs/{dataset}/cmatrix.png")

    else:
        plt.imshow(cf_matrix)
    return

def cdisco(conv_maps, gradients_wrt_conv_layer, predictions, classes):
    def relu(data):
        return data * (data>0)
    conv_embeddings=np.mean(conv_maps, axis=(2,3))
    pu, ps, pvh = np.linalg.svd(conv_embeddings)
    
    rg=relu(gradients_wrt_conv_layer)
    
    g_k = np.multiply(rg, conv_maps)
    pooled_g = np.mean(g_k, axis=(2,3))
    projected_g = np.dot(pooled_g, pvh)

    z = {}
    mean_gk={}
    mean_gnotk={}
    std_gnotk={}
    
    for k in classes:
        idxs_of_class_k = np.asarray(np.argwhere(np.argmax(predictions, axis=1)==k)).ravel() # predicitons = model predictions
        all_the_rest = np.asarray(np.argwhere(np.argmax(predictions, axis=1)!=k)).ravel()

        sel_info=np.asarray([projected_g[n] for n in idxs_of_class_k])
        mean_gk[int(k)] = np.mean(sel_info, axis=0) #std

        rest_info = np.asarray([projected_g[n] for n in all_the_rest])
        mean_gnotk[int(k)] = np.mean(rest_info, axis=0)
        std_gnotk[int(k)] = np.std(rest_info, axis=0)
        z[int(k)] = mean_gk[int(k)] - mean_gnotk[int(k)]
        z[int(k)] = z[int(k)] / std_gnotk[int(k)]


    class_reordered_eigenvalues={}
    class_concept_candidates = {}

    for k in classes:
        class_reordered_eigenvalues[int(k)] = z[int(k)]
        class_concept_candidates[int(k)] = np.asarray(class_reordered_eigenvalues[int(k)]).argsort()[::-1]
    return class_concept_candidates, pvh