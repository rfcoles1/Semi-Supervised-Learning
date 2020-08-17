import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import scipy.stats as stats
from scipy.signal import savgol_filter

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

def plot_loss(results, labels, lim=[100,3], filt=False, train=False):
    fig = plt.figure(figsize=(12,8))
    grid = gs.GridSpec(1,1)

    ax = fig.add_subplot(grid[0])

    for result, label in zip(results, labels):
        if train:
            mse = np.concatenate(result['batch_MSE'])
            iters = np.arange(0, len(mse),1)
        else:
            mse = np.concatenate(result['test_MSE'])
        iters = np.concatenate(result['iterations'])
        
        if filt:
            mse = savgol_filter(mse, 11, 1)

        ax.plot(iters, mse,label=label)
    
    if train:
        ax.set_ylabel('Train Loss (MSE)')
    else:
        ax.set_ylabel('Test Loss (MSE)')
    ax.set_xlabel('Iterations')
    ax.set_ylim(bottom=0, top=lim[1])
    ax.set_xlim(left=0, right=lim[0])
    ax.grid()
    ax.legend()

def plot_metrics(results, labels, lim=[100,3,1,1,2]):
    fig = plt.figure(figsize=(16,12))
    grid = gs.GridSpec(2,2)

    ax0 = fig.add_subplot(grid[0])
    ax1 = fig.add_subplot(grid[1])
    ax2 = fig.add_subplot(grid[2])
    ax3 = fig.add_subplot(grid[3])

    for result, label in zip(results, labels):
        iters = np.concatenate(result['epochs'])
        mse = np.concatenate(result['test_MSE'])
        bias = np.concatenate(result['test_abs_bias'])
        MAD = np.concatenate(result['test_MAD_loss'])
        bias_MAD = np.concatenate(result['test_bias_MAD_loss'])

        ax0.plot(iters, mse, label=label)
        ax1.plot(iters, bias, label=label)
        ax2.plot(iters, MAD, label=label)
        ax3.plot(iters, bias_MAD, label=label)

    ax0.set_xlabel('Epochs')
    ax0.set_ylabel('MSE')
    ax0.set_xlim(left=0, right=lim[0])
    ax0.set_ylim(bottom=0, top=lim[1])
    ax0.grid()

    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('abs_bias')
    ax1.set_xlim(left=0, right=lim[0])
    ax1.set_ylim(bottom=0, top=lim[2])
    ax1.grid()

    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('MAD')
    ax2.set_xlim(left=0, right=lim[0])
    ax2.set_ylim(bottom=0, top=lim[3])
    ax2.grid()

    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('bias_MAD')
    ax3.set_xlim(left=0, right=lim[0])
    ax3.set_ylim(bottom=0, top=lim[4])
    ax3.grid()

def plot_resid(true, pred, y_min, y_max):
   
   
    true = true*(y_max - y_min) + y_min
    pred = pred*(y_max - y_min) + y_min

    resid = true - pred
    bias = np.median(resid)
    std = np.std(resid)

    xlim = [0,3]
    ylim = [4,-4]
    
    fig = plt.figure(figsize=(8,6))
    grid = gs.GridSpec(1,2, width_ratios=[4,1])

    ax0 = fig.add_subplot(grid[0,0])
    ax0.scatter(true, resid, color='crimson')
    ax0.plot(xlim, [0,0], color='dimgrey', linestyle='--')

    ax0.set_xlim(xlim[0], xlim[1])
    ax0.set_ylim(ylim[0], ylim[1])
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

