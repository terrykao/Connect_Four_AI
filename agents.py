import numpy as np
from train import *

class MinimaxAgent:
    '''Represents an abstract game playing agent, that can be initialized with any AI game playing algorithm
    to allow it to select the next best move.  Also requires initialization of the target game playing functions
    of actions, result, and playerWins. 
    Primarily exposes a common interface, for takeAction(state), which will select the next best move for the
    agent, and return the resulting state.
    '''
    
    def __init__(self, aiAlgorithmF, actionsF, resultF, playerWinsF, printStateF=None, maxDepth=3):
        '''Constructor to initialize this agent. Requires that we pass function references for the AI 
        algorithm to use to determine next best move, and the game playing functions for actions, result, 
        and playerWins. 

        ARGUMENTS:
            aiAlgorithmF - AI algorithm function that returns best action for agent to taken given current 
                           game state. aiAlgorithmF(state) -> action
            resultF - Function that returns new state given current state and action. 
                      resultF(state, action) -> state
                      
            playerWinsF - Function that test is player wins in given state. 
                          playerWinsF(state, player)
            printStateF - Function to pretty print game state.
            maxDepth - Maximum search depth, for AI algorithms that needs it. Defaults to 3.
        '''
        self.aiAlgorithm = aiAlgorithmF
        self.actions = actionsF
        self.result = resultF
        self.playerWins = playerWinsF
        self.printState = printStateF
        self.maxDepth = maxDepth
        
        
    def takeAction(self, state):
        '''Takes best action for agent given passed state, returning new state as a result, using the 
        AI algorithm set for this agent.
        
        ARGUMENTS:
            state - current game state
        '''
        action = self.aiAlgorithm(state, 
                                  actionsF=self.actions, 
                                  playerWinsF=self.playerWins, 
                                  printStateF=self.printState, 
                                  maxDepth=self.maxDepth)
        newState = self.result(state, action)
        return newState

    def __str__(self):
        '''String representation of agent.
        '''
        return '{{<{}> aiAlgorithm: {}}}'.format(self.__class__.__name__, self.aiAlgorithm.__name__) 

class QNeuralNetAgent:
    def __init__(self, Q, validMovesF, makeMoveF):
        self.Q = Q
        self.validMovesF = validMovesF
        self.makeMoveF = makeMoveF

    def setQ(self, Q):
        self.Q = Q
        
    def takeAction(self, state):
        '''Takes best action for agent given passed state, returning new state as a result, using the 
        trained Q neural network set for this agent.
        
        ARGUMENTS:
            state - current game state
        '''
        return self.makeMove(state)
        
    def makeMove(self, state):
        '''
        '''
        # Re-use select a move by epsilon greedy policy, with epsilon set to 0, so that action
        # will be picked from the Q neural net model
        action = selectMoveByEpsilonGreedyPolicy(epsilon=0., 
                                                 state=state, 
                                                 Q=self.Q, 
                                                 validMovesF=self.validMovesF)

        new_state = self.makeMoveF(state, action)
        return new_state

    def __str__(self):
        '''String representation of agent.
        '''
        return "{{<{}>}}".format(self.__class__.__name__)



class RandomAgent:
    def __init__(self, validMovesF, makeMoveF):
        self.validMovesF = validMovesF
        self.makeMoveF = makeMoveF

    def takeAction(self, state):
        '''Takes best action for agent given passed state, returning new state as a result, using the 
        trained Q neural network set for this agent.
        
        ARGUMENTS:
            state - current game state
        '''
        actions = self.validMovesF(state)
        action = actions[np.random.randint(0,len(actions))]
        return self.makeMoveF(state, action) 

    def __str__(self):
        '''String representation of agent.
        '''
        return "{{<{}>}}".format(self.__class__.__name__)


