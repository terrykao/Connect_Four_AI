import pickle
import time
from util import *

class Cons:
    fileName = 'fileName'
    neuralNetLayers = 'neuralNetLayers'
    batchSize = 'batchSize'
    bufferSize = 'bufferSize'
    startEpsilon = 'startEpsilon'
    minEpsilon = 'minEpsilon'
    gamma = 'gamma'
    epochs = 'epochs'
    device = 'device'
    totalWins = 'totalWins'
    totalDraws = 'totalDraws'
    epochAvgTrainTime = 'epochAvgTrainTime'
    totalTrainTime = 'totalTrainTime'
    startTime = 'startTime'
    lastUpdateTime = 'lastUpdateTime'
    losses = 'losses'
    accs = 'accs'
    epsilons = 'epsilons'
    trainTimes = 'trainTimes'
    gameResults = 'gameResults'
    totalMoves = 'totalMoves'
    models = 'models'    

    
def createTrainingHistoryFileName():
    '''Generates training history file name using a dynamic timestamp
    '''
    return 'training-history-{}.pickle'.format(str(int(time.time())))

def createTrainingHistoryData(nn_layers, epochs, 
                              batchSize, bufferSize, 
                              startEpsilon, minEpsilon, gamma):
    '''Create dictionary to store training history.
    '''
    return {
        Cons.fileName: createTrainingHistoryFileName(), # Auto generate training history file name
        Cons.neuralNetLayers: nn_layers,
        Cons.batchSize: batchSize,
        Cons.bufferSize: bufferSize,
        Cons.startEpsilon: startEpsilon,
        Cons.minEpsilon: minEpsilon,
        Cons.gamma: gamma,
        Cons.epochs: epochs,
        Cons.device: str(getDevice()),
        Cons.totalWins: 0,
        Cons.totalDraws: 0,
        Cons.epochAvgTrainTime: 0,
        Cons.totalTrainTime: 0,
        Cons.startTime: time.time(),
        Cons.lastUpdateTime: 0.,
        Cons.losses: [],
        Cons.accs: [],
        Cons.epsilons: [],
        Cons.trainTimes: [],
        Cons.totalMoves: [],
        Cons.gameResults: [],
        Cons.models: []        
    }

def updateTrainingHistoryData(trainingHistory, loss, acc, epsilon, trainTime, moves, gameResult, model,
                              epochAvgTrainTime, totalTrainTime):
    '''Updates key training history elements after each epoch.
    '''
    trainingHistory[Cons.losses].append(loss)
    trainingHistory[Cons.accs].append(acc)
    trainingHistory[Cons.epsilons].append(epsilon)
    trainingHistory[Cons.trainTimes].append(trainTime)
    trainingHistory[Cons.totalMoves].append(moves)
    trainingHistory[Cons.gameResults].append(gameResult)
    trainingHistory[Cons.models].append(model)
    trainingHistory[Cons.epochAvgTrainTime] = epochAvgTrainTime
    trainingHistory[Cons.totalTrainTime] = totalTrainTime
    trainingHistory[Cons.lastUpdateTime] = time.time()  
    
    if gameResult == 1:
        trainingHistory[Cons.totalWins] += 1
    elif gameResult == 0:
        trainingHistory[Cons.totalDraws] += 1
        

    

def saveTrainingHistory(trainingHistory):
    '''Serializes training history data to a file.
    '''
    f = open(trainingHistory[Cons.fileName], 'wb')
    pickle.dump(trainingHistory, f)
    f.close()
    
def loadTrainingHistory(filename):
    '''Loads pickle file containing previous training history data.
    
    INPUTS:
        filename - file name of pickle file to load
        
    RETURNS:
        dict
    '''
    f = open(filename, 'rb')
    trainingHist = pickle.load(f)
    f.close()
    return trainingHist


