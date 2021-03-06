import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pickle 
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import scipy.stats as stats
from scipy.signal import savgol_filter
from astropy.visualization import make_lupton_rgb

big = 32
med = 24
smol = 18

plt.rc('font', size=smol)
plt.rc('figure', titlesize=big)
plt.rc('legend', fontsize=smol)
plt.rc('axes', titlesize=med)
plt.rc('axes', labelsize=smol)
plt.rc('xtick', labelsize=smol)
plt.rc('ytick', labelsize=smol)

def cutimg(x, size=32):
    xdim = np.shape(x)[1]
    start=xdim//2-(size//2)
    return x[start:start+size, start:start+size]

def load_folder(path):
    filelist = os.listdir(path) 
    filelist = [f for f in filelist if '.pickle' in f]
    histories = []
    labels = []
    
    for thisfile in filelist:
        f = open(path + '/' + thisfile, 'rb')
        hist = pickle.load(f)
        histories.append(hist)
        labels.append(thisfile[:-7])
        f.close()

    return histories, labels


def _cols_to_channels(cols):
    n = len(cols)
    channels = []
    for i in range(n):
        c = cols[i]
        if c == 'u':
            channels.append(0)
        if c == 'g':
            channels.append(1)
        if c == 'r':
            channels.append(2)
        if c == 'i':
            channels.append(3)
        if c == 'z':
            channels.append(4)
    return channels

def plot_colour(img, r=1,g=2,b=3, normed=True):
    channel1 = img[:,:,r:r+1]
    channel2 = img[:,:,g:g+1]
    channel3 = img[:,:,b:b+1]

    if normed:
        mn = np.min([channel1, channel2, channel3])
        mx = np.max([channel1, channel2, channel3])

        channel1 = (channel1 - mn) / (mx - mn)
        channel2 = (channel2 - mn) / (mx - mn)
        channel3 = (channel3 - mn) / (mx - mn)

    image = np.concatenate([channel1, channel2, channel3], axis=2)

    plt.axis('off')
    plt.imshow(image)

def plot_lupton(img):
    g = img[:,:,1:2]
    r = img[:,:,2:3]
    i = img[:,:,3:4]

    rgb = make_lupton_rgb(i,r,g, Q=10, stretch=0.5)

    plt.axis('off')
    plt.imshow(rgb)

def plot_collection(imgs):
    n = len(imgs)
    rows = int(np.ceil(n/3))
    cols = 3

    fig = plt.figure(figsize=(cols*2, rows*2))
    grid = gs.GridSpec(rows,cols, wspace=0.05,hspace=0.05)

    for i in range(rows):
        for j in range(cols):
            ax1 = fig.add_subplot(grid[i,j])
            plot_colour(imgs[i*cols+j])

 
def compare_images(imgs1, imgs2, normed=True):
    n = len(imgs1)
    rows = int(np.ceil(n/3))
    cols = 3

    fig = plt.figure(figsize=(cols*4, rows*2))
    outer = gs.GridSpec(1,2, wspace=0.1)
    grid1 = gs.GridSpecFromSubplotSpec(rows, cols, subplot_spec=outer[0], wspace=0.05, hspace=0.05)
    grid2 = gs.GridSpecFromSubplotSpec(rows, cols, subplot_spec=outer[1], wspace=0.05, hspace=0.05)

    for i in range(rows):
        for j in range(cols):
            ax1 = fig.add_subplot(grid1[i,j])
            plot_colour(imgs1[i*cols+j],normed=normed)

            ax2 = fig.add_subplot(grid2[i,j])
            plot_colour(imgs2[i*cols+j],normed=normed)
    

def show_aug(img, Augmenter, resize=True, size=32):

    aug = Augmenter.transform(img)

    fig = plt.figure(figsize=(6,3))
    grid = gs.GridSpec(1,2)
    
    if resize==True:
        img = cutimg(img,size)
        aug = cutimg(aug,size)

    ax1 = fig.add_subplot(grid[0])
    plot_colour(img)

    ax2 = fig.add_subplot(grid[1])
    plot_colour(aug)
           

def show_indiv_channels(img, resize=True, size=32, scale=True):

    channels = np.shape(img)[-1]

    if resize==True:
        img = cutimg(img,size)

    fig = plt.figure(figsize=(3.5*channels,3))
    grid = gs.GridSpec(1,channels)

    for i in range(channels):
        ax = fig.add_subplot(grid[i])

        min1 = np.min(img[:,:,i])
        max1 = np.max(img[:,:,i])
        
        if scale:
            ax.imshow(img[:,:,i], vmin=min1, vmax=max1)
        else:
            ax.imshow(img[:,:,i])

        ax.axis('off')
    plt.show()
 
def show_aug_indiv_channels(img, Augmenter, resize=True, size=32, scale=True):

    aug = Augmenter.transform(img)
    channels = np.shape(img)[-1]

    if resize==True:
        img = cutimg(img,size)
        aug = cutimg(aug,size)

    fig = plt.figure(figsize=(3.5*channels,6))
    grid = gs.GridSpec(2,channels)

    for i in range(channels):
        ax1 = fig.add_subplot(grid[0,i])
        ax2 = fig.add_subplot(grid[1,i])

        min1 = np.min(img[:,:,i])
        max1 = np.max(img[:,:,i])
        min2 = np.min(aug[:,:,i])
        max2 = np.min(aug[:,:,i])
        mini = min(min1, min2)
        maxi = max(max1, max2)
        if scale:
            ax1.imshow(img[:,:,i], vmin=min1, vmax=max1)
            ax2.imshow(aug[:,:,i], vmin=min1, vmax=max1)
        else:
            ax1.imshow(img[:,:,i])
            ax2.imshow(aug[:,:,i])
        ax1.axis('off')
        ax2.axis('off')
    plt.show()


def plot_resid(true, pred, y_min, y_max, xlim=[0.4,3.6], ylim=[-1,1]):

    true = true*(y_max - y_min) + y_min
    pred = pred*(y_max - y_min) + y_min

    resid = true - pred
    bias = np.median(resid)
    std = np.std(resid)

    fig = plt.figure(figsize=(8,6))
    grid = gs.GridSpec(1,2, width_ratios=[4,1])

    ax0 = fig.add_subplot(grid[0,0])
    ax0.scatter(true, resid, color='crimson', alpha=0.5)
    ax0.plot(xlim, [0,0], color='dimgrey', linestyle='--')

    ax0.set_xlim(xlim[0], xlim[1])
    ax0.set_ylim(ylim[0], ylim[1])
    ax0.set_xlabel('Test Value')
    ax0.set_ylabel('Residual')
    ax0.grid()

    n, bin_edges = np.histogram(resid, 50)
    probs = n/np.shape(resid)[0]
    bin_mid = (bin_edges[1:]+bin_edges[:-1])/2.0
    bin_wid = bin_edges[1]-bin_edges[0]
    (mu, sigma) = stats.norm.fit(resid)
    y = stats.norm.pdf(bin_mid, mu, sigma)*bin_wid

    ax1 = fig.add_subplot(grid[0,1])
    ax1.plot(y, bin_mid, color='darkred', lw=2)
    ax1.plot([0,max(y)], [0,0], color='dimgrey', linestyle='--')

    ax1.set_xlim(0,max(y))
    ax1.set_ylim(ylim[0],ylim[1])
    ax1.get_xaxis().set_visible(False)
    ax1.grid(axis='y')
    ax1.yaxis.tick_right()
    ax1.yaxis.set_label_position('right')

def plot_error(true, pred, y_min, y_max, xlim=[0.4,3.6], ylim=[-4,4]):

    true = true*(y_max - y_min) + y_min
    pred = pred*(y_max - y_min) + y_min

    resid = true - pred
    bias = np.median(resid)
    std = np.std(resid)

    fig = plt.figure(figsize=(8,6))
    grid = gs.GridSpec(1,1)

    ax0 = fig.add_subplot(grid[0,0])
    ax0.scatter(true, resid, color='crimson', alpha=0.5)
    ax0.plot(xlim, [0,0], color='dimgrey', linestyle='--')

    ax0.set_xlim(xlim[0], xlim[1])
    ax0.set_ylim(ylim[0], ylim[1])
    ax0.set_xlabel('Test Value')
    ax0.set_ylabel('Residual')
    ax0.grid()

    n, bin_edges = np.histogram(resid, 50)
    probs = n/np.shape(resid)[0]
    bin_mid = (bin_edges[1:]+bin_edges[:-1])/2.0
    bin_wid = bin_edges[1]-bin_edges[0]
    (mu, sigma) = stats.norm.fit(resid)
    y = stats.norm.pdf(bin_mid, mu, sigma)*bin_wid

def plot_diff(true, pred, y_min, y_max, xlim=[0.4,3.6]):

    true = true*(y_max - y_min) + y_min
    pred = pred*(y_max - y_min) + y_min

    fig = plt.figure(figsize=(8,8))
    grid = gs.GridSpec(1,1)

    ax0 = fig.add_subplot(grid[0,0])
    ax0.scatter(true, pred, color='crimson', alpha=0.5)
    ax0.plot(xlim, xlim, color='dimgrey', linestyle='--')

    ax0.set_xlim(xlim[0], xlim[1])
    ax0.set_ylim(xlim[0], xlim[1])
    ax0.set_xticks([1,2,3])
    ax0.set_yticks([1,2,3])
    ax0.set_xlabel('Test Redshift',fontsize=26)
    ax0.set_ylabel('Predicted Redshift',fontsize=26)
    ax0.tick_params(axis='both',labelsize=20)
    ax0.grid()


