import sys
sys.path.insert(1, '../Utils')
from plot_results import *

def plot_results(results, labels, lim=[400000,3,1]):
    fig = plt.figure(figsize=(12,8))
    grid = gs.GridSpec(1,2, wspace=0.05)
    
    ax0 = fig.add_subplot(grid[0])
    ax1 = fig.add_subplot(grid[1])
    
    for result,label in zip(results,labels):
        iters = np.concatenate(result['iterations'])
        loss = np.concatenate(result['test_loss'])
        acc = np.concatenate(result['test_acc'])
        ax0.plot(iters,loss, label=label)        
        ax1.plot(iters,acc, label=label)

    ax0.set_ylabel('Loss')
    ax0.set_xlabel('Iterations')
    ax0.set_xlim(left=0, right=lim[0])
    ax0.set_ylim(bottom=0, top=lim[1])
    ax0.grid()
    
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Iterations')
    ax1.set_xlim(left=0, right=lim[0])
    ax1.set_ylim(bottom=0, top=lim[2])
    ax1.yaxis.tick_right()
    ax1.yaxis.set_label_position('right')
    ax1.grid()
    ax1.legend()
