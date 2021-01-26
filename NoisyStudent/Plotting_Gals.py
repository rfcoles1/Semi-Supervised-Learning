import sys
sys.path.insert(1, '../Utils')
from plot_results import *


def plot_nums(results, labels, ylim=1):
    fig = plt.figure(figsize=(8,8))
    grid = gs.GridSpec(1,1)
    ax = fig.add_subplot(grid[0])
    
    supervised_mse = []
    supervised_num = []
    augstudent_mse = []
    augstudent_num = []
    basestudent_mse = []
    basestudent_num = []

    for result, label in zip(results, labels):
        mse = np.concatenate(result['test_MSE'])
        if label[:5] == 'Super':
            supervised_mse.append(mse[-1])
            #supervised_mse.append(min(mse))
            supervised_num.append(int(label[6:10]))
        elif label[-5:] == 'NoAug':
            basestudent_mse.append(mse[-1])
            #basestudent_mse.append(min(mse))
            basestudent_num.append(int(label[8:12]))
        else:
            augstudent_mse.append(mse[-1])
            #augstudent_mse.append(min(mse))
            augstudent_num.append(int(label[8:12]))

    ax.plot(supervised_num, supervised_mse, label='Supervised',\
        marker='*', markersize=14, color='crimson', lw=2.5)
    #ax.plot(augstudent_num, augstudent_mse, label='Student w/ Augs')
    ax.plot(basestudent_num, basestudent_mse, label='Student',\
        marker='*', markersize=14, color='slateblue', lw=2.5)

    ax.set_ylabel('Validation Loss (MSE)')
    ax.set_xlabel('Number of Labels')
    ax.set_ylim(bottom=0, top=ylim)
    ax.set_xlim(left=250, right=2500)
    ax.grid()
    ax.legend()


    

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
    
    ax.set_ylabel('Loss (MSE)')
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
        iters = np.concatenate(result['iterations'])
        mse = np.concatenate(result['test_MSE'])
        bias = np.concatenate(result['test_abs_bias'])
        MAD = np.concatenate(result['test_MAD_loss'])
        bias_MAD = np.concatenate(result['test_bias_MAD_loss'])

        ax0.plot(iters, mse, label=label)
        ax1.plot(iters, bias, label=label)
        ax2.plot(iters, MAD, label=label)
        ax3.plot(iters, bias_MAD, label=label)

    ax0.set_xlabel('Iterations')
    ax0.set_ylabel('MSE')
    ax0.set_xlim(left=0, right=lim[0])
    ax0.set_ylim(bottom=0, top=lim[1])
    ax0.grid()

    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('abs_bias')
    ax1.set_xlim(left=0, right=lim[0])
    ax1.set_ylim(bottom=0, top=lim[2])
    ax1.grid()

    ax2.set_xlabel('Iterations')
    ax2.set_ylabel('MAD')
    ax2.set_xlim(left=0, right=lim[0])
    ax2.set_ylim(bottom=0, top=lim[3])
    ax2.grid()

    ax3.set_xlabel('Iterations')
    ax3.set_ylabel('bias_MAD')
    ax3.set_xlim(left=0, right=lim[0])
    ax3.set_ylim(bottom=0, top=lim[4])
    ax3.grid()
    ax3.legend()
