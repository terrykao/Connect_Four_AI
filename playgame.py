import time
from IPython.display import clear_output

from connectfour import *
from minimax import *

def letsPlayAGame(agent1, agent2, terminalTestF, playerWinsF, printStateF, initialGameStateF, verbose=True, clear=True):
    '''Plays a single Connect Four game between two different AI agents. It's actually more generic than that,
    in it is passed references to specific game functions of terminalTest, playerWins, and initialGameState.
    
    ARGUMENTS:
        agent1
        agent2
        terminalTestF
        playerWinsF
        printStateF
        initialGameStateF
    RETURNS:
        nothing
    '''
    # Initialize s, the list of games states.
    s = []
    
    # Push the initial game state, an empty board.  Set move count to 0.
    s.append(initialGameStateF())
    moveCnt = 0
    
    thinkTimeAgent1 = 0.
    thinkTimeAgent2 = 0.
    startGameTime = time.time()
    
    # Loop until game is done, with either agent winning or a draw
    while not terminalTest(s[-1]):
        # Each agent takes turns making a move, based on the move count. Agent 1 goest first, followed by
        # Agent 2, and so on.  On each action taken by an agent, we get a new state for the game.
        thinkStartTime = time.time()
        newState = agent1.takeAction(s[-1]) if moveCnt % 2 == 0 else agent2.takeAction(s[-1])
        thinkTime = time.time() - thinkStartTime
        
        if moveCnt % 2 == 0:
            thinkTimeAgent1 += thinkTime
        else:
            thinkTimeAgent2 += thinkTime

        # Append new game state to list
        s.append(newState)

        # Increment move count and restart game loop
        moveCnt += 1
        
        # Pretty print the game state if verbose is True
        if verbose:
            if clear:
                clear_output(wait=True)
            print("Agent {} took {:.3f}s to make move.".format(int(moveCnt % 2 + 1), thinkTime))
            printStateF(s[-1])
            print()
    
    if verbose:
        if playerWinsF(s[-1], MiniMaxConstants.MAX):
            print("X won.")
        elif playerWinsF(s[-1], MiniMaxConstants.MIN):
            print("O won.")
        else:
            print("Tie game.") 
            #print("validMovess[-1] -> {}".format(validMoves(s[-1])))
            #print("terminalTest(s[-1]) -> {}".format(terminalTest(s[-1])))
            #print("s[-1] -> {}".format(s[-1]))
            #player1PiecesCnt, player2PiecesCnt = pieces(s[-1])
            #print("player1PiecesCnt -> {}, player2PiecesCnt -> {}".format(player1PiecesCnt, player2PiecesCnt))
            #print("player1PiecesCnt + player2PiecesCnt == 6*7 -> {}".format(player1PiecesCnt + player2PiecesCnt == 6*7)) 
            #print("playerWins(state, 1) -> {}".format(playerWins(s[-1], 1, verbose=True)))
            #print("playerWins(state, 2) -> {}".format(playerWins(s[-1], 2, verbose=True)))
            #printState(s[-1])
            
        print("Total game time was {:.3f}s, with {} plies.".format(time.time() - startGameTime, int(moveCnt/2)))
        print("Agent 1 ({}) average move time is {:.3f}s, taking a total of {:.3f}s".format(agent1, thinkTimeAgent1/int(moveCnt/2), thinkTimeAgent1))
        print("Agent 2 ({}) average move time is {:.3f}s, taking a total of {:.3f}s".format(agent2, thinkTimeAgent2/int(moveCnt/2), thinkTimeAgent2))

    if playerWinsF(s[-1], MiniMaxConstants.MAX):
        return 1
    elif playerWinsF(s[-1], MiniMaxConstants.MIN):
        return 2
    else:
        return 0
               

def contestPlay(agent1, agent2, total_games):
    agent1Wins = 0
    agent2Wins = 0
    ties = 0
    
    for i in range(total_games):
        gameResult = letsPlayAGame(agent1=agent1,
                                   agent2=agent2,
                                   terminalTestF=terminalTest,
                                   playerWinsF=playerWins,
                                   printStateF=printState,
                                   initialGameStateF=createInitialGameState,
                                   verbose=False)
        if gameResult == 1:
            agent1Wins += 1
        elif gameResult == 2:
            agent2Wins += 1
        elif gameResult == 0:
            ties += 1
            
        clear_output(wait=True)
        print("Game #{}. Agent1 ({}): {}. Agent2 ({}): {}. Ties: {}.".format(i+1, agent1, agent1Wins, agent2, agent2Wins, ties))

    print("Played {} games. Agent1({}) won {} ({}%). Agent2({}) won {} ({}%). Total ties: {} ({}%).".format(total_games, agent1, agent1Wins, int(agent1Wins/total_games*100), agent2, agent2Wins, int(agent2Wins/total_games*100), ties, int(ties/total_games*100)))



