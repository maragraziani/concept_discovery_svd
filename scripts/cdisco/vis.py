### cdisco_vis
import numpy as np
import PIL.Image as Image
import scipy
import matplotlib.pyplot as plt
import sys
sys.path.append('.')
import torchvision
import os 

transform = torchvision.transforms.Compose([
 torchvision.transforms.Resize(299),
 torchvision.transforms.CenterCrop(299),
 torchvision.transforms.ToTensor()
])

transform_normalize = torchvision.transforms.Normalize(
     mean=[0.485, 0.456, 0.406],
     std=[0.229, 0.224, 0.225]
 )

def get_base_vector(idx,dim):
    base_vector = np.zeros(dim)
    base_vector[idx]=1
    return base_vector

def cdisco_concept_vis(img_id, concept_v, dim_c, dim_w, dim_h, conv_maps):
    concept_v = np.reshape(concept_v, (1,dim_c))
    fm= np.reshape(conv_maps[img_id], (dim_c, -1))
    concept_map = np.dot(concept_v, fm).reshape((dim_h, dim_w))
    return concept_map

def cdisco_vis_extremes(concepts, candidates, pvh, conv_embeddings, conv_maps, paths, predictions, idx_to_labels, savefold=''):
    plt.rcParams['figure.figsize']=(20,10)
    mostleast={}
    SAVEFOLD=savefold
    for c in concepts:

        p_data= np.dot(conv_embeddings, pvh[c,:].T)
        plt.figure()
        plt.subplot(1,4,1)
        plt.imshow(Image.open(paths[np.argmin(p_data)]))
        plt.subplot(1,4,2)
        plt.imshow(Image.open(paths[np.argmax(p_data)]))

        plt.subplot(1,4,3)
        #p_data= np.dot(conv_embeddings, pvh[34,:].T)
        plt.imshow(Image.open(paths[np.argmax(p_data)]))
        concept_vector=pvh[c,:]
        img_id=np.argmax(p_data)
        idx_pred=np.argmax(predictions[img_id])
        fmap=cdisco_concept_vis(img_id, concept_vector, dim_c, dim_w, dim_h, conv_maps)

        plt.imshow((transform(Image.open(paths[img_id]))).swapaxes(0,1).swapaxes(1,-1))#,alpha=0.5)
        hmap=scipy.ndimage.zoom(fmap, 299/fmap.shape[0],order=1)
        plt.imshow(hmap, cmap='jet', alpha=0.8, vmin=-np.abs(hmap).max(), vmax=np.abs(hmap).max())

        plt.subplot(1,4,4)

        upsampled_fmap=scipy.ndimage.zoom(fmap, 299/fmap.shape[0],order=1)
        image=transform(Image.open(paths[img_id])).swapaxes(0,1).swapaxes(1,-1)

        th = np.percentile(upsampled_fmap,80) 
        mask=np.zeros((299,299,3))
        bmask=(upsampled_fmap>th)*1
        mask[:,:,0]=mask[:,:,1]=mask[:,:,2]=bmask
        image=transform(Image.open(paths[img_id])).swapaxes(0,1).swapaxes(1,-1)
        th_image=mask*image.cpu().numpy()
        #plt.imshow(th_image)
        #patch=image[max(0,xm-20):min(299,xm+20),max(ym-20,0):min(ym+20,299)]
        plt.imshow(th_image)
        mostleast[c]=(np.argmin(p_data), np.argmax(p_data))
        plt.title(f"{c}, {idx_to_labels[str(idx_pred)]}")
        plt.axis("off")
        plt.savefig(f"{SAVEFOLD}/ml_{c}")

def cdisco_vis_extremes_extensive(concept, candidates, pvh, conv_embeddings, conv_maps, paths, predictions, idx_to_labels, savefold=''):
    dim_c, dim_w, dim_h = conv_maps[0,:,:,:].shape
    plt.rcParams['figure.figsize']=(20,10)
    mostleast={}
    SAVEFOLD=savefold
    c=concept

    p_data= np.dot(conv_embeddings, pvh[c,:].T)
    plt.figure()
    plt.subplot(1,4,1)
    concept_vector=pvh[c,:]
    img_id=np.argsort(p_data)[::-1][0]
    idx_pred=np.argmax(predictions[img_id])
    fmap=cdisco_concept_vis(img_id, concept_vector, dim_c, dim_w, dim_h, conv_maps)
    upsampled_fmap=scipy.ndimage.zoom(fmap, 299/fmap.shape[0],order=1)
    image=transform(Image.open(paths[img_id])).swapaxes(0,1).swapaxes(1,-1)

    th = np.percentile(upsampled_fmap,80) 
    mask=np.zeros((299,299,3))
    bmask=(upsampled_fmap>th)*1
    mask[:,:,0]=mask[:,:,1]=mask[:,:,2]=bmask
    image=transform(Image.open(paths[img_id])).swapaxes(0,1).swapaxes(1,-1)
    th_image=mask*image.cpu().numpy()
    plt.title(f"{c}, {idx_to_labels[str(idx_pred)]}")
    plt.imshow(th_image)
    plt.axis('off')
    
    plt.subplot(1,4,2)
    #plt.imshow(Image.open(paths[np.argsort(p_data)[::-1][1]]))
    concept_vector=pvh[c,:]
    img_id=np.argsort(p_data)[::-1][1]
    idx_pred=np.argmax(predictions[img_id])
    fmap=cdisco_concept_vis(img_id, concept_vector, dim_c, dim_w, dim_h, conv_maps)
    upsampled_fmap=scipy.ndimage.zoom(fmap, 299/fmap.shape[0],order=1)
    image=transform(Image.open(paths[img_id])).swapaxes(0,1).swapaxes(1,-1)

    th = np.percentile(upsampled_fmap,80) 
    mask=np.zeros((299,299,3))
    bmask=(upsampled_fmap>th)*1
    mask[:,:,0]=mask[:,:,1]=mask[:,:,2]=bmask
    image=transform(Image.open(paths[img_id])).swapaxes(0,1).swapaxes(1,-1)
    th_image=mask*image.cpu().numpy()
    plt.title(f"{c}, {idx_to_labels[str(idx_pred)]}")
    plt.imshow(th_image)
    plt.axis('off')

    plt.subplot(1,4,3)
    concept_vector=pvh[c,:]
    img_id=np.argsort(p_data)[::-1][2]
    idx_pred=np.argmax(predictions[img_id])
    fmap=cdisco_concept_vis(img_id, concept_vector, dim_c, dim_w, dim_h, conv_maps)
    upsampled_fmap=scipy.ndimage.zoom(fmap, 299/fmap.shape[0],order=1)
    image=transform(Image.open(paths[img_id])).swapaxes(0,1).swapaxes(1,-1)

    th = np.percentile(upsampled_fmap,80) 
    mask=np.zeros((299,299,3))
    bmask=(upsampled_fmap>th)*1
    mask[:,:,0]=mask[:,:,1]=mask[:,:,2]=bmask
    image=transform(Image.open(paths[img_id])).swapaxes(0,1).swapaxes(1,-1)
    th_image=mask*image.cpu().numpy()
    plt.title(f"{c}, {idx_to_labels[str(idx_pred)]}")
    plt.imshow(th_image)
    plt.axis('off')
    
    plt.subplot(1,4,4)

    concept_vector=pvh[c,:]
    img_id=np.argsort(p_data)[::-1][3]
    idx_pred=np.argmax(predictions[img_id])
    fmap=cdisco_concept_vis(img_id, concept_vector, dim_c, dim_w, dim_h, conv_maps)
    upsampled_fmap=scipy.ndimage.zoom(fmap, 299/fmap.shape[0],order=1)
    image=transform(Image.open(paths[img_id])).swapaxes(0,1).swapaxes(1,-1)

    th = np.percentile(upsampled_fmap,80) 
    mask=np.zeros((299,299,3))
    bmask=(upsampled_fmap>th)*1
    mask[:,:,0]=mask[:,:,1]=mask[:,:,2]=bmask
    image=transform(Image.open(paths[img_id])).swapaxes(0,1).swapaxes(1,-1)
    th_image=mask*image.cpu().numpy()
    plt.title(f"{c}, {idx_to_labels[str(idx_pred)]}")
    plt.imshow(th_image)
    plt.axis('off')
    
    mostleast[c]=(np.argmin(p_data), np.argmax(p_data))
    
    plt.axis("off")
    plt.savefig(f"{SAVEFOLD}/ml_extensive_{c}")

def img_3c_hist(img):
    image=img
    colors = ("red", "green", "blue")
    # create the histogram plot, with three lines, one for
    # each color
    #plt.figure()
    plt.xlim([0, 254])
    max_=0
    for channel_id, color in enumerate(colors):
        histogram, bin_edges = np.histogram(
            image[:, :, channel_id], bins=256, range=(0, 256)
        )
        plt.plot(bin_edges[0:-1], histogram, color=color)
        max_y=np.max(bin_edges[:254])
        if max_y>max_:
            max_=max_y
    plt.ylim([0, 600])
    plt.title("Color Histogram")
    plt.xlabel("Color value")
    plt.ylabel("Pixel count")
    
def img_to_colorness(img):
    cms=get_all_color_measures(img)
    i=0
    colormapping={'red':'red', 'orange':'orange', 'yellow':'yellow', 'green':'green', 'cyano':'lightblue', 
              'blue':'blue', 'purple':'purple', 'magenta':'fuchsia', 'black':'black', 'white':'grey'}
    for k in cms.keys():
        #print(k)
        plt.bar(i, cms[k], color=colormapping[k])
        i+=1
    plt.xticks(np.arange(i), cms.keys())
    return cms
def cdisco_conceptboard(concept, savefold=''):
    
    savefold+=f'concept_segmentations_{concept}/'
    try:
        files=os.listdir(savefold)
    except:
        print("Run cdisco_concept_segmentation() first.")
    count=len(files)
    rows=int(count/10)
    plt.rcParams['figure.figsize']=(20,10)#10+2,rows)
    plt.figure()
    idx=1
    for i in range(count):
        if '.png' in files[i]:
            try:
                plt.subplot(rows, 10, idx)
                plt.imshow(Image.open(savefold+files[i]))
                plt.axis("off")
                idx+=1
            except:
                break
    plt.savefig(f"{savefold}_cboard.svg")
    plt.savefig(f"{savefold}_cboard.png")
    plt.show()
    
    return

def cdisco_histoboard(concept, savefold=''):
    
    savefold+=f'concept_segmentations_{concept}/'
    try:
        files=os.listdir(savefold)
    except:
        print("Run cdisco_concept_segmentation() first.")
    count=len(files)
    rows=int(count/10)
    plt.rcParams['figure.figsize']=(20,10)#10+2,rows)
    plt.figure()
    idx=1
    for i in range(count):
        if '.png' in files[i]:
            try:
                plt.subplot(rows, 10, idx)
                img= np.asarray(Image.open(savefold+files[i]))
                plt.hist(img)
                plt.axis("off")
                idx+=1
            except:
                break
    plt.savefig(f"{savefold}_chistboard.svg")
    plt.savefig(f"{savefold}_chistboard.png")
    plt.show()
    
    return
def vis_fmaps(channel, conv_maps, savefold=''):
    plt.rcParams['figure.figsize']=(20,10)
    n,c,h,w=conv_maps.shape
    i=1
    for img in range(0,n,1000):
        plt.subplot(1,10,i)
        plt.imshow(conv_maps[i,channel,:,:])
        i+=1
    return

def cdisco_vis_overlay_fmaps(channel, conv_maps, paths, span=500, savefold='', concept_vector=None, start_from=0):
    plt.rcParams['figure.figsize']=(30,10)
    N, dim_c, dim_w, dim_h = conv_maps.shape
    
    if concept_vector is not None:
        channel_vector=concept_vector
        channel='manual_cvector'
    else:
        channel_vector=get_base_vector(channel, dim_c)
    cols=int(N/span)
    i=1
    for img_id in range(start_from,N,span): 
        plt.subplot(1,cols,i)
        fmap=cdisco_concept_vis(img_id, channel_vector, dim_c, dim_w, dim_h, conv_maps)
        plt.imshow((transform(Image.open(paths[img_id]))).swapaxes(0,1).swapaxes(1,-1))#,alpha=0.5)
        hmap=scipy.ndimage.zoom(fmap, 299/fmap.shape[0],order=1)
        plt.imshow(hmap, cmap='jet', alpha=0.4, vmin=-np.abs(hmap).max(), vmax=np.abs(hmap).max())
        plt.axis('off')
        i+=1
    plt.savefig(f'{savefold}/cdisco_fmaps_{channel}.png')
    return
        
        