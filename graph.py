import numpy as np
from matplotlib import pylab as plt
from IPython.display import clear_output


def plotLineData(data, xlabel="", ylabel="", figsize=(9,6), avg=1):
    '''Plot line data.
    '''
    def running_mean(x, N=500):
        cumsum = np.cumsum(np.insert(x, 0, 0)) 
        return (cumsum[N:] - cumsum[:-N]) / float(N)
    
    plt.figure(figsize=figsize)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.plot(running_mean(data, avg))
    plt.show()
  
    
def plotScatterData(data, xlabel="", ylabel="", figsize=(9,6), avg=1):
    '''Plot scatter point data.
    '''
    plt.figure(figsize=figsize)
    plt.scatter(range(0, len(data)), data, s=90, c='b', alpha=0.1)
    #plt.plot(range(0, len(data)), data)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.show()    
