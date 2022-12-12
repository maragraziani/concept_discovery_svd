import numpy as np
import os
import matplotlib.pyplot as plt

# Cdisco concept analysis
def cosine(a,b):
    return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))

def get_base_vector(idx, dim):
    base_vector = np.zeros(dim)
    base_vector[idx]=1
    return base_vector

def cdisco_concepts_vs_axes(concepts, candidates, pvh, conv_maps):
    print(len(concepts))
    print(conv_maps.size())
    return

def cdisco_alignment(concept, class_concept_candidates, idx_to_labels):
    aligned=[]
    not_aligned=[]
    aligned_ks=[]
    for k in class_concept_candidates.keys():
        if class_concept_candidates[k][0]==concept:
            aligned.append(idx_to_labels[str(k)])
            aligned_ks.append(k)
        else:
            not_aligned.append(idx_to_labels[str(k)])

    print("aligns: ", aligned)

def cdisco_direction_alignment(pvh, conc):
    angles=[]
    for i in range(len(pvh[conc,:])):
        angle= cosine(pvh[conc,:], get_base_vector(i, len(pvh[conc,:])))
        angles.append(angle)
    return angles, np.argmax(np.abs(angles))
        
def cdisco_pop_concepts(class_concept_candidates, classes, pvh, savefold='',top=3):
    k=classes[0]
    
    pop_first=np.zeros(len(class_concept_candidates[int(k)]))#.keys()))
    pop_sec=np.zeros(len(class_concept_candidates[int(k)]))#.keys()))
    pop_third=np.zeros(len(class_concept_candidates[int(k)]))#.keys()))
    
    for k in classes:
        pop_first[class_concept_candidates[k][0]]+=1
        pop_sec[class_concept_candidates[k][1]]+=1
        pop_third[class_concept_candidates[k][2]]+=1
    first=np.argmax(pop_first)
    sec=np.argmax(pop_sec)
    third=np.argmax(pop_third)
    
    print("First candidate: ", first, np.max(pop_first))
    print("Second:",sec, np.max(pop_sec))
    print("Third:",third, np.max(pop_third))
    
    plt.rcParams['figure.figsize']=(20,10)
    plt.figure()
    j=1
    for conc in [first, sec, third]:
        plt.subplot(3,1,j)
        angles, channel = cdisco_direction_alignment(pvh, conc)
        for i in range(len(pvh[conc,:])):
            plt.bar(i, angles[i])
        plt.title(f'Concept: {conc}, Channel: {channel}, Angle: {angles[channel]}')
        plt.ylim(-1,1)
        #plt.xtick(channel, channel)
        j+=1
    try:
        plt.savefig(f'{savefold}/cdisco_analyze/pop_concepts.png')
    except:
        os.mkdir(f'{savefold}/cdisco_analyze/')
        plt.savefig(f'{savefold}/cdisco_analyze/pop_concepts.png')
    return

def cdisco_angle_dissection(pvh, candidates, savefold=''):
    max_angles=[]
    channels=[]
    plt.rcParams['figure.figsize']=(30,10)
    plt.figure()

    for conc in candidates.keys():
        angles, channel=cdisco_direction_alignment(pvh, conc)
        #if angles[channel]>0.:
        channels.append(f'{conc}.{channel}')
        max_angles.append(angles[channel])
    sorted_idxs = np.argsort(max_angles)
    for i in range(len(channels)):
        plt.bar(i, max_angles[sorted_idxs[i]])
    plt.axhline(y = 0.6, color = 'r', linestyle = '-')
    plt.axhline(y = -0.6, color = 'r', linestyle = '-')
    plt.ylim(-1,1)
    plt.xticks(np.arange(len(channels)),[channels[sidx] for sidx in sorted_idxs], rotation=90)
    plt.title(f'Concept alignment to canonical basis ([concept].[channel]). N candidates: {len(candidates)} of {len(pvh[0,:])} dimensions.')
    plt.savefig(f'{savefold}/cdisco_analyze/angle_dissection.png')
    return