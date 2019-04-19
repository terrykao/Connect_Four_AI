import torch

def getDevice(gpu_num=0):
    '''Helper to get the GPU device if available, otherwise returns CPU device.

    RETRUNS:
        'cuda:{gpu_num}' or 'cpu'
    '''
    return torch.device("cuda:{}".format(gpu_num) if torch.cuda.is_available() else "cpu")


