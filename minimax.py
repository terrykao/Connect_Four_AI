import time

from connectfour import *
from connect4heuristics import *

class MiniMaxConstants:
    '''MiniMax constant values.
    '''
    MAX = 1
    MIN = 2
    INFINITY = float('inf')


def isMax(state):
    '''Test if Max has current turn given state.
    '''
    return player(state) == MiniMaxConstants.MAX

def isMin(state):
    '''Test if Min has current turn given state.
    '''
    return player(state) == MiniMaxConstants.MIN

def initialBestValue(state):
    '''Returns initial best value, depending on if current state indicates next to move is MAX or MIN. 
    Basically returns +INF or -INF depending on if next move is MIN or MAX respectively.
    '''
    v = MiniMaxConstants.INFINITY
    if isMax(state):
        v *= -1
    return v


class SearchMetrics:
    '''Helper class to track metrics in search algorithms. Model for tracking current depth, nodes 
    expanded or explored, and depth where solution was found, as well timer start/stop functions to
    track how much time a search function took to execute.
    '''
    def __init__(self, nodesExpanded=0, depth=0, solutionDepth=0):
        '''Constructor. Initializes everything to 0 value by default.
        '''
        self.nodesExpanded = nodesExpanded
        self.depth = depth
        self.solutionDepth = solutionDepth
        self.startTime = 0
        self.stopTime = 0
                
    def start(self):
        '''Called before start of search function entry to record start time in seconds since epoch.
        '''
        self.startTime = time.time()
    
    def stop(self):
        '''Called after search function exit, to record stop time in seconds since epoch.
        '''
        self.stopTime = time.time()
        
    def runTime(self):
        '''Computes time in seconds taken by search function.
        '''
        return self.stopTime - self.startTime
        
    def __repr__(self):
        return "{{<{}> nodesExpanded: {}, solutionDepth: {}, depth: {}, runTime: {}}}".format(self.__class__.__name__,
                                                                                 self.nodesExpanded,
                                                                                 self.solutionDepth,
                                                                                 self.depth,
                                                                                 self.runTime())


def minimaxValueAB(state, alpha, beta, playerWinsF, depth, searchMetrics=SearchMetrics()):
    # If game is over then return utility based on who won, or if we had a draw
    if terminalTest(state):
        return utility(state, MiniMaxConstants.MAX, playerWinsF)
    
    # If we reached the maximum depth limit for game tree search then return an estimated utility
    # based on current game state, which player has a higher strategic board position based on current
    # pieces on board.
    if depth == 0:
        return utilityEstimated(state, MiniMaxConstants.MAX, playerWinsF)
    
    # Track search metrics, the current depth and number of nodes searched
    searchMetrics.nodesExpanded += 1
    searchMetrics.depth += 1
    
    # Intiial best utility value based on current state, which indicates if next move is MAX or MIN.
    # Value will be -INF or +INF, for MAX or MIN respectively.
    bestValue = initialBestValue(state)
    
    # Loop through all valid actions for current game state; i.e. valid moves for player with turn
    for a in actions(state):
        value = minimaxValueAB(result(state, a), alpha, beta, playerWinsF, depth-1, searchMetrics=searchMetrics)
        if isMax(state):
            bestValue = max(bestValue, value)
            if bestValue >= beta:
                break
            alpha = max(alpha, bestValue)
        elif isMin(state):
            bestValue = min(bestValue, value)
            if bestValue <= alpha:
                break
            beta = min(beta, bestValue)
 
    searchMetrics.depth -= 1

    return bestValue
 
def alphaBetaSearch(state, actionsF, playerWinsF, maxDepth=3, printStateF=None, verbose=False):
    '''Searches for best move in game tree, using MiniMax algorithm limiting search by maximum tree depth limit, 
    and optimization using Alpha-Beta pruning of subtrees.  If depth limit is reached, an estimated utility
    is returned, based on the strategic value of current board state relative to MIN or MAX player. Otherwise,
    utility based on win, lose, or tie is used.
    
    ARGUMENTS:
        playerWinsF - Function that test is player wins in given state -> playerWinsF(state, player)
        maxDepth - Maximum search depth, for AI algorithms that needs it. Defaults to 3.    
    RETURNS:
        best action for player with current turn based on state
    
    '''
    bestValue = initialBestValue(state)
    bestAction = None
    metrics = SearchMetrics()
    metrics.start()
    
    if verbose:
        print("alphaBetaSearch() initial bestValue -> {}".format(bestValue))
    
    for a in actionsF(state):
        value = minimaxValueAB(result(state, a), 
                               -MiniMaxConstants.INFINITY, MiniMaxConstants.INFINITY, 
                               playerWinsF, maxDepth,
                               searchMetrics=metrics)

        if isMax(state) and value > bestValue:
            bestValue = value
            bestAction = a
        elif isMin(state) and value < bestValue:
            bestValue = value
            bestAction = a

        if verbose:
            print("==== minimaxValueAB: {}, bestAction so far: {} =====".format(value, bestAction))
            if printStateF is not None:
                printStateF(result(state, a))            
       
    metrics.stop()
    
    if verbose:
        print("alphaBetaSearch() metrics -> {}".format(metrics))
    
    return bestAction

    
def utility(state, player, playerWinsF):
    '''Payoff for player on reaching state.
    
    ARGUMENTS:
        state: game state
        player (int): 1 or 2
        playerWinsF: function that tests if player wins with given state - playerWinsF(state, player) -> boolean
        
    RETURNS:
        int
    '''
    if playerWinsF(state, player):
        value = 100000
    elif playerWinsF(state, otherPlayer(player)):
        value = -100000
    else:
        value = 0
    
    return value


