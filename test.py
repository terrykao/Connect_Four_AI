import copy
import time
import random
import pickle
from collections import OrderedDict

import numpy as np
import torch
from torch.autograd import Variable


from minimax import *
from agents import *
from playgame import *
from train import *
from trainhistory import *
from connectfour import *


model, trainHist = trainQLearningNN(nn_layers=[(6*7), 150, 150, 150, 150, 150, 150, 7], 
                                           agent=QNeuralNetAgent(Q=None, validMovesF=validMoves, makeMoveF=makeMove),
                                           terminalTestF=terminalTest,
                                           playerWinsF=playerWins,
                                           validMovesF=validMoves,
                                           makeMoveF=makeMove,
                                           printStateF=printState,
                                           initialGameStateF=createInitialGameState,
                                           stateNdarrayConverterF=stateAsNdarray,
                                           bufferSize=2000,
                                           batchSize=250,
                                           gamma=0.9,
                                           epsilon=1.0,
                                           min_epsilon=0.,
                                           epochs=5000)


device = getDevice()
print(device)


trainHist = loadTrainingHistory('training-history-1544189239.pickle')
#print(trainHist)



letsPlayAGame(agent1=QNeuralNetAgent(Q=trainHist['models'][-1], validMovesF=validMoves, makeMoveF=makeMove),
              agent2=MinimaxAgent(aiAlgorithmF=alphaBetaSearch, actionsF=actions, resultF=result, playerWinsF=playerWins, maxDepth=2),
              terminalTestF=terminalTest,
              playerWinsF=playerWins,
              printStateF=printState,
              initialGameStateF=createInitialGameState)

contestPlay(agent1=MinimaxAgent(aiAlgorithmF=alphaBetaSearch, actionsF=actions, resultF=result, playerWinsF=playerWins, maxDepth=2), 
            agent2=RandomAgent(validMovesF=validMoves, makeMoveF=makeMove), 
            total_games=10)

contestPlay(agent1=RandomAgent(validMovesF=validMoves, makeMoveF=makeMove), 
            agent2=RandomAgent(validMovesF=validMoves, makeMoveF=makeMove), 
            total_games=10)

#contestPlay(agent1=QNeuralNetAgent(Q=trainHist['models'][4814], validMovesF=validMoves, makeMoveF=makeMove),
#            agent2=RandomAgent(validMovesF=validMoves, makeMoveF=makeMove), 
#            total_games=10000)

print(QNeuralNetAgent(Q=trainHist['models'][4814], validMovesF=validMoves, makeMoveF=makeMove))


