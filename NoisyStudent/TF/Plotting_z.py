import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs


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

def plot_loss(results, labels, lim=[100,3]):
    fig = plt.figure(figsize=(12,8))
    grid = gs.GridSpec(1,1)

    ax = fig.add_subplot(grid[0])

    for result, label in zip(results, labels):
        iters = np.concatenate(result['iterations'])
        mse = np.concatenate(result['test_MSE'])

        ax.plot(iters, mse,label=label)

    ax.set_ylabel('Loss (MSE)')
    ax.set_xlabel('Epochs')
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
        iters = np.concatenate(result['iterations'])
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

