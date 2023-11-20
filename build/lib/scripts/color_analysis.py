import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_color_measure(image, mask=None, mtype=None, verbose=False):
    if mask is not None:
        print("A mask was specified.")
        print("This feature has not been implemented yet.")
        return None
    if mtype is None:
        if verbose:
            print("No type was given")
        return None
    if mtype=='colorfulness':
        return colorfulness(image)
    else:
        return colorness(image, mtype, threshold=0, verbose=verbose)

def get_all_color_measures(image, mask=None, verbose=False):
    all_types = ['colorfulness',
                 'red',
                 'orange',
                 'yellow',
                 'green',
                 'cyano',
                 'blue',
                 'purple',
                 'magenta',
                 'black',
                 'white'
                ]
    cms={}
    for mtype in all_types:
        if verbose:  print(mtype)
        cms[mtype]=get_color_measure(image,mask=mask,mtype=mtype, verbose=verbose)
    return cms

def colorfulness(img):
    """Colorfulness metric by .. & ..
    """
    # split the image into its respective RGB components
    (B, G, R) = cv2.split(img.astype("float"))
    # compute rg = R - G
    rg = np.absolute(R - G)
    # compute yb = 0.5 * (R + G) - B
    yb = np.absolute(0.5 * (R + G) - B)
    # compute the mean and standard deviation of both `rg` and `yb`
    (rbMean, rbStd) = (np.mean(rg), np.std(rg))
    (ybMean, ybStd) = (np.mean(yb), np.std(yb))
    # combine the mean and standard deviations
    stdRoot = np.sqrt((rbStd ** 2) + (ybStd ** 2))
    meanRoot = np.sqrt((rbMean ** 2) + (ybMean ** 2))
    # derive the "colorfulness" metric and return it
    return stdRoot + (0.3 * meanRoot)

def colorness(image, color_name, threshold=0, verbose=False):
    """ Colorness as defined in submission to ICCV
        blue-ness = #blue pixels / # pixels
        Use threshold = 0 for quantization of hue ranges
    """
    image = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    # show color histograms for validation
    if verbose:
        h,s,v = hsv_histograms(image)
        plt.figure()
        plt.plot(h)
        plt.figure()
        plt.plot(s)
        plt.figure()
        plt.plot(v)
    # quantization of hue ranges
    # if threshold not 0, color name is changed into hue window
    if threshold == 0:
        hue_min, hue_max = quantize_hue_ranges(image, color_name)
        if verbose:
            print('hue min, hue max: ', hue_min, hue_max)
    else:
        h_point = color_picker(color_name)
        hue_min = round_hue(h_point[0][0][0]-threshold)
        hue_max = round_hue(h_point[0][0][0]+threshold)
        if verbose:
            print('hue min, hue max: ', hue_min, hue_max)
    if (hue_min == hue_max == 0) or (hue_min == 0 and hue_max == 255):
        #it is either black or white
        if color_name=='black':
            low_c = np.array([0,
                              0,
                              0])
            upp_c = np.array([hue_max,
                              100,
                              100])
        if color_name=='white':
            low_c = np.array([0,
                              0,
                              190])
            upp_c = np.array([hue_max,
                              50,
                              255])
        if verbose:
            print('low_c', low_c, 'upp_c', upp_c)
        mask = cv2.inRange(image, low_c, upp_c)
    elif hue_min>hue_max:
        low_c = np.array([0,
                      50,
                      77])
        upp_c = np.array([hue_max,
                      255,
                      255])
        mask1 = cv2.inRange(image, low_c, upp_c)

        low_c = np.array([hue_min,
                      50,
                      77])
        upp_c = np.array([180,
                      255,
                      255])
        mask2 = cv2.inRange(image, low_c, upp_c)
        mask = cv2.bitwise_or(mask1, mask1, mask2)
    else:
        low_c = np.array([hue_min,
                          50,
                          77])
        upp_c = np.array([hue_max,
                          255,
                          255])
        if verbose:
            print('low_c', low_c, 'upp_c', upp_c)
        mask = cv2.inRange(image, low_c, upp_c)
    if verbose:
        print(mask)
    res = cv2.bitwise_and(image, image, mask = mask)
    if verbose:
        plt.figure()
        plt.imshow(mask, cmap='Greys')
        plt.colorbar()
        plt.figure()
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_HSV2RGB))
        plt.figure()
        plt.imshow(cv2.cvtColor(res, cv2.COLOR_HSV2RGB))
    x,y,z = image.shape
    if verbose:
        print(np.sum(mask==255)/(float(x)*float(y)))
    return float(np.sum(mask==255))/(float(x)*float(y))
"""
Functions called by colorness module
"""
def hsv_histograms(image):
    hist_hue = cv2.calcHist([image], [0], None, [180], [0, 180])
    hist_sat = cv2.calcHist([image], [1], None, [256], [0, 256])
    hist_val = cv2.calcHist([image], [2], None, [256], [0, 256])
    #print np.mean(image[:,:,0])
    #print np.min(image[:,:,0])
    #print np.max(image[:,:,0])
    return hist_hue, hist_sat, hist_val

def color_picker(color_name):
    """
    Converts a color name into HSV values
    """
    brg_colors={}
    brg_colors['red']= np.uint8([[[0,0,255 ]]])
    brg_colors['orange'] = np.uint8([[[0,128,255 ]]])
    brg_colors['yellow'] = np.uint8([[[0,255,255 ]]])
    brg_colors['green'] = np.uint8([[[0,255,0 ]]])
    brg_colors['cyano'] = np.uint8([[[255,255,0 ]]])
    brg_colors['blue'] = np.uint8([[[255,0,0]]])
    brg_colors['purple'] = np.uint8([[[255,0,128]]])
    brg_colors['magenta'] = np.uint8([[[255,0,255 ]]])
    brg_colors['white'] = np.uint8([[[255,255,255 ]]])
    brg_colors['black'] = np.uint8([[[0,0,0 ]]])
    rgb_color_code = brg_colors[color_name]
    return cv2.cvtColor(rgb_color_code,cv2.COLOR_BGR2HSV)

def round_hue(hue_val):
    hues = np.arange(0,180)
    if hue_val<180:
        hue_def = hues[hue_val]
    else:
        hue_def = hues[(hue_val)%179]
    return hue_def

def quantize_hue_ranges(image, color_name):
    """
    Quantization of HSV space as in ICCV submission
    """
    if color_name == 'red':
        hue_min = 165
        hue_max = 10
    elif color_name == 'orange':
        hue_min = 10
        hue_max = 25
    elif color_name == 'yellow':
        hue_min = 25
        hue_max = 40
    elif color_name == 'green':
        hue_min = 40
        hue_max = 75
    elif color_name == 'cyano':
        hue_min = 75
        hue_max = 100
    elif color_name == 'blue':
        hue_min = 100
        hue_max = 125
    elif color_name == 'purple':
        hue_min = 125
        hue_max = 145
    elif color_name == 'magenta':
        hue_min = 145
        hue_max = 165
    elif (color_name == 'white' or color_name == 'black'):
        hue_min = 0
        hue_max = 255
    return hue_min, hue_max

def get_color_measure(image, mask=None, mtype=None, verbose=False):
    if mask is not None:
        print("A mask was specified.")
        print("This feature has not been implemented yet.")
        return None
    if mtype is None:
        if verbose:
            print("No type was given")
        return None
    if mtype=='colorfulness':
        return colorfulness(image)
    else:
        return colorness(image, mtype, threshold=0, verbose=verbose)

def get_all_color_measures(image, mask=None, verbose=False):
    all_types = [#'colorfulness',
                 'red',
                 'orange',
                 'yellow',
                 'green',
                 'cyano',
                 'blue',
                 'purple',
                 'magenta',
                 'black',
                 #'white'
                ]
    cms={}
    for mtype in all_types:
        if verbose:  print(mtype)
        cms[mtype]=get_color_measure(image,mask=mask,mtype=mtype, verbose=verbose)
    return cms