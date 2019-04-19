import numpy as np
import copy

def createInitialGameState(rows=6, cols=7):
    '''Creates an initial start game state for Connect Four.  Defatuls to the standard classic game with 
    6 rows and 7 columns.  However, this function allows for initialization of other variations
    
    ARGUMENTS:
        rows - number of rows in game board, defatuls to 6
        cols - number of columns in game board, defaults to 7
        
    RETURNS:
        game state
    '''
    state = []
    for i in range(cols):
        state.append([])
    
    return state


def pieces(state):
    '''Returns the count of player 1 and player 2 pieces as a tuple, in that order.
    
    ARGUMENTS:
        state - game state
        
    RETURNS:
        tuple (player1_cnt, player2_cnt)
    '''
    one_cnt = 0
    two_cnt = 0
    for col in range(len(state)):
        column = state[col]
        for row in range(len(column)):
            if column[row] == 1:
                one_cnt += 1
            elif column[row] == 2:
                two_cnt += 1
    
    return one_cnt, two_cnt


def playerToMove(state):
    '''Returns player to move based on given state. 
    
    ARGUMENTS:
        state - game state    
    
    RETURNS:
        integer of player to move, 1 or 2
    '''
    player1_cnt, player2_cnt = pieces(state)
    return 1 if (player1_cnt <= player2_cnt) else 2
    
    
def makeMove(state, move):
    '''Returns the new state resulting from making move of current player from current state.
    
    ARGUMENTS:
        state - current game state
        move - column to drop piece for current player
    '''
    # Get the current player to move - 1 or 2
    player = playerToMove(state)
    
    # Make deep copy of current state, which we will modify as new state after making move
    new_state = copy.deepcopy(state)
    new_state[move].append(player)
    return new_state


def validMoves(state, rows=6):
    '''Returns a list of valid moves that the current player can take. In Connect Four, this is just a list of
    all columns that are not full, and can accept another piece.
    
    ARGUMENTS:
        state - game state   
        rows - number of rows in game board, defatuls to 6

    RETURNS:
        [int] list of valid moves
    '''
    moves = []
    for col in range(len(state)):
        if len(state[col]) < rows:
            moves.append(col)
    
    return moves
    

def player2Str(player_num, map_repr={'1': '1', '2': '2'}):
    '''Converts player number (integer) to a string representation given by map representation.
    
    ARGUMENTS:
        player_num - player number, 1 or 2
        map_repr - dictionary to map player number to a string representation
        
    RETURNS:
        string representation of player
    '''
    return map_repr[str(player_num)]
    

def printState(state, rows=6, spacing=0, empty='.', player_map_repr={'1': 'X', '2': 'O'}):
    '''Pretty prints the current Connect Four game state to stdout.
    
    ARGUMENTS:
        rows - number of rows in game board, defatuls to 6
        spacing - white space between cells in all four directions, defaults to 0
        empty - character representing an empty cell, defaults to '.'
    '''
    space = ' ' * spacing
    for row in range(rows-1, -1, -1):
        line = ''
        for col in range(len(state)):
            column = state[col]
            if len(column)-1 >= row:
                line += space + player2Str(column[row], player_map_repr) + space
            else:
                line += space + empty + space
    
        print(line)

        if spacing > 0:
            line = ''
            for i in range(spacing):
                line += "\n"
            print(line)
                

def stateAsNdarray(state, rows=6):
    '''Helper to conver state represented as list of list to a 2d numpy array
    '''
    cols = len(state)
    ndarr = np.zeros((rows, cols))

    for row in range(rows-1, -1, -1):
        for col in range(cols):
            column = state[col]
            if len(column)-1 >= row:
                r = rows - row - 1
                c = col
                ndarr[r, c] = column[row]                         
    
    return ndarr



def playerWins(state, player, rows=6, verbose=False):
    '''Tests if state has given player win.
    
    ARGUMENTS:
        state - game state   
        player - player number to check
    
    RETURNS:    
        boolean
    '''
    cols = len(state)
    
    # Convert state to numpy ndarray which is easier to check for winning state
    s = stateAsNdarray(state)
    
    # Check for horizontal four in a row. 
    for r in range(6):
        for c in range(4):
            if s[r,c] == player:
                seq_cnt = 0
                for x in range(4):
                    if c+x < cols and s[r,c+x] == player:
                        seq_cnt += 1
                    else:
                        break                
    
                if seq_cnt == 4:
                    if verbose:
                        print("Horizontal win at {}".format((r,c)))
                    return True         

    # Check for vertical four in a row. 
    for c in range(7):
        for r in range(3):
            if s[r,c] == player:
                seq_cnt = 0
                for x in range(4):
                    if r+x < rows and s[r+x,c] == player:
                        seq_cnt += 1
                    else:
                        break                
    
                if seq_cnt == 4:
                    if verbose:
                        print("Vertical win at {}".format((r,c)))
                    return True  


    # Check for diagonal down four in a row. Rows 0 to 2.    
    for r in range(3):
        for c in range(4):
            if s[r,c] == player:
                seq_cnt = 0
                for x in range(4):
                    if c+x < cols and r+x < rows and s[r+x,c+x] == player:
                        seq_cnt += 1
                    else:
                        break                
    
                if seq_cnt == 4:
                    if verbose:
                        print("Diagonal down win at {}".format((r,c)))
                        printState(state)
                        print(s)
                    return True                
    
    # Check for diagonal up four in a row. Rows 3 to 5.    
    for r in range(3,6):
        for c in range(4):
            if s[r,c] == player:
                seq_cnt = 0
                for x in range(4):
                    if c-x < cols and r-x < rows and s[r-x,c-x] == player:
                        seq_cnt += 1
                    else:
                        break                
    
                if seq_cnt == 4:
                    if verbose:
                        print("Diagonal up win at {}".format((r,c)))
                    return True              
    
    # Scanned entire grid and no winning combinations found
    return False




def player(state):
    '''Returns the player that can make a move for current state.
    
    ARGUMENTS:
        state: a Connect Four game state
    
    RETURNS:
        1 or 0
    '''    
    return playerToMove(state)

def otherPlayer(player):
    '''Returns the other player value.

    ARGUMENTS:
        player (int): 1 or 2

    RETURNS:
        int
    '''
    return int(player % 2 + 1)

    
def terminalTest(state):
    '''Tests if the game is over.
    
    ARGUMENTS:
        state: a Connect Four game state
        
    RETURNS:
        boolean
    '''
    player1PiecesCnt, player2PiecesCnt = pieces(state)
    return (player1PiecesCnt + player2PiecesCnt == 6*7 or 
            playerWins(state, 1) or 
            playerWins(state, 2))
        
    
def actions(state):
    '''Returns list of all valid actions for given game state.
    
    ARGUMENTS:
        state (list[int]): a tic-tac-toe game state   
        
    RETURNS:
        list[(player, position)]
    '''
    return validMoves(state)       

def result(state, action):
    '''Makes a Connect Four move given by action for current state of game. Action is provided as a 
    tuple (player, position).  Where, player is either 'X' or 'O', and position if (row, col), 0 based
    starting at top-left corner.
    '''
    return makeMove(state, action)
    

def winner(state, rows=6):
    '''Returns winning player number if state is a win for either player 1 or 2, or 0 is no winner.

    ARGUMENTS:
        state - game state    
    
    RETURNS:
        integer of winning player or 0 for none
    '''
    if playerWins(state, 1):
        win = 1
    elif playerWins(state, 2):
        win = 2
    else:
        win = 0
    
    return win


