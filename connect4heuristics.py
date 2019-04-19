from connectfour import *



def winningDiagonalUpCombination(state, player):
    '''Scans for diagonal up, winning combinations, three in a row, where opponent can only block one end, 
    forcing win for player.
    
    ARGUMENTS:
        state - game state   
        player - player number to check
    
    RETURNS:    
        int - count of winning combinations
    '''
    cnt = 0

    # Convert state to numpy ndarray which is easier to check for winning state
    s = stateAsNdarray(state)
    rows = s.shape[0]
    cols = s.shape[1]
    
    # Check for diagonal down consecutive pieces for player. 
    # Go across columns of the board from 1 to 6, and scan diagonally downward for lines
    r = 0
    for c in range(1, cols):
        # Reset diagonal counter and sequential count
        x = 0
        seq_cnt = 0
        
        # Start scanning diagonally down
        while x < rows-1:   # This is the max cells we go diagonally, but may break early
            # If cell is target player value then increment sequential count
            if s[r+x,c-x] == player:
                seq_cnt += 1
                
                
            if seq_cnt == 3:
                if ((c-x-1 >= 0 and r+x+1 < rows and s[r+x+1,c-x-1] == 0) and
                    (r+x+1 == rows-1 or (r+x+2 < rows and (s[r+x+2,c-x-1] == 1 or s[r+x+2,c-x-1] == 2))) and
                    (c-x+3 < cols and r+x-3 >= 0 and s[r+x-3,c-x+3] == 0) and
                    (r+x-2 >= 0 and (s[r+x-2,c-x+3] == 1 or s[r+x-2,c-x+3] == 2))):
                        cnt += 1

                seq_cnt = 0
                
            # increment diagonal counter
            x += 1
            
            # check if we reached out diagonal limits and need to break
            if r+x >= rows or c-x < 0:
                break
    
                    
    # Check for diagonal down lines
    # Go down rows of board from 1 to 4, and scan diagonally downward for lines
    c = cols - 1
    for r in range(1, rows-1):
        # Reset diagonal counter and sequential count
        x = 0
        seq_cnt = 0
        
        # Start scanning diagonally down
        while x < rows-1:  # This is the max cells we go diagonally, but may break early
            # If cell is target player value then increment sequential count
            if s[r+x,c-x] == player:
                seq_cnt += 1
                
            if seq_cnt == 3:
                if ((r+x+1 < rows  and c-x-1 >= 0 and s[r+x+1,c-x-1] == 0) and
                    (r+x+1 == rows-1 or (r+x+2 < rows and (s[r+x+2,c-x-1] == 1 or s[r+x+2,c-x-1] == 2))) and
                    (r+x-3 >= 0 and c-x+3 < cols and s[r+x-3,c-x+3] == 0) and
                    (r+x-2 >= 0 and (s[r+x-2,c-x+3] == 1 or s[r+x-2,c-x+3] == 2))):
                        cnt += 1

                seq_cnt = 0          

            # increment diagonal counter
            x += 1
            
            # check if we reached out diagonal limits and need to break
            if r+x >= rows:  # checking row is sufficient
                break    
    
    return cnt


def winningDiagonalDownCombination(state, player):
    '''Scans for diagonal down, winning combinations, three in a row, where opponent can only block one end, 
    forcing win for player.
    
    ARGUMENTS:
        state - game state   
        player - player number to check
    
    RETURNS:    
        int - count of winning combinations
    '''
    cnt = 0
    
    # Convert state to numpy ndarray which is easier to check for winning state
    s = stateAsNdarray(state)
    rows = s.shape[0]
    cols = s.shape[1]
    
    # Check for diagonal down consecutive pieces for player. 
    # Go across columns of the board from 0 to 5, and scan diagonally downward for lines
    r = 0
    for c in range(cols-1):
        # Reset diagonal counter and sequential count
        x = 0
        seq_cnt = 0
        
        # Start scanning diagonally down
        while x < rows-1:   # This is the max cells we go diagonally, but may break early
            # If cell is target player value then increment sequential count
            if s[r+x,c+x] == player:
                seq_cnt += 1
                
            if seq_cnt == 3:
                if ((c+x-3 >= 0 and r+x-3 >= 0 and s[r+x-3,c+x-3] == 0) and
                    (r+x-2 < rows and s[r+x-2,c+x-3] == 1 or s[r+x-2,c+x-3] == 2) and
                    (c+x < cols-1 and r+x < rows-1 and s[r+x+1,c+x+1] == 0) and
                    (r+x+2 == rows-1 or (r+x+2 < rows-1 and (s[r+x+2,c+x+1] == 1 or s[r+x+2,c+x+1] == 2)))):
                        cnt += 1
                seq_cnt = 0

                
            # If last row or last col or we got a 'gap'
            #if r+x == rows-1 or c+x == cols-1 or s[r+x,c+x] != player:
            #    if seq_cnt > 1:
            #        results[seq_cnt] += 1
            #    seq_cnt = 0
                
            # increment diagonal counter
            x += 1
            
            # check if we reached out diagonal limits and need to break
            if r+x >= rows or c+x >= cols:
                break
    

    
    # Check for diagonal down lines
    # Go down rows of board from 1 to 4, and scan diagonally downward for lines
    c = 0
    for r in range(1, rows-1):
        # Reset diagonal counter and sequential count
        x = 0
        seq_cnt = 0
        
        # Start scanning diagonally down
        while x < rows-1:  # This is the max cells we go diagonally, but may break early
            # If cell is target player value then increment sequential count
            if s[r+x,c+x] == player:
                seq_cnt += 1
                
            if seq_cnt == 3:
                if ((c+x-3 >= 0 and r+x-3 >= 0 and s[r+x-3,c+x-3] == 0) and
                    (r+x-2 < rows and (s[r+x-2,c+x-3] == 1 or s[r+x-2,c+x-3] == 2)) and
                    ( (c+x+1 < cols) and (r+x+1 < rows) and (s[r+x+1,c+x+1] == 0) ) and
                    ( (r+x+1 == rows-1) or ( (r+x+2 < rows) and ( (s[r+x+2,c+x+1] == 1) or (s[r+x+2,c+x+1] == 2) ) ) ) ):            
                     cnt += 1            

            # increment diagonal counter
            x += 1
            
            # check if we reached out diagonal limits and need to break
            if r+x >= rows:  # checking row is sufficient
                break        
    
    return cnt



def winningHorizontalCombination(state, player):
    '''Scans for horizontal, winning combinations, three in a row, where opponent can only block one end, 
    forcing win for player.
    
    ARGUMENTS:
        state - game state   
        player - player number to check
    
    RETURNS:    
        int - count of winning combinations
    '''
    cnt = 0
    
    # Convert state to numpy ndarray which is easier to check for winning state
    s = stateAsNdarray(state)
    rows = s.shape[0]
    cols = s.shape[1]
    
    # Check for horizontal 3 consecutive pieces in a row with blank pieces to the left and right.
    # Go down rows of the board from top to bottom
    for r in range(rows):
        # Reset column and sequential count 
        c = 0
        seq_cnt = 0
        
        # Start scanning horizontally across the row
        while c < cols:
            # If cell is target player value then increment sequential count
            if s[r,c] == player:
                seq_cnt += 1
            
            
            if seq_cnt == 3:
                if ((c-3 >= 0 and s[r,c-3] == 0) and
                    (r == rows-1 or (r < rows-1 and (s[r+1,c-3] == 1 or s[r+1,c-3] == 2))) and
                    (c < cols-1 and s[r,c+1] == 0) and
                    (r == rows-1 or (r < rows-1 and (s[r+1,c+1] == 1 or s[r+1,c+1] == 2)))):
                        cnt += 1
                seq_cnt = 0
            # If last column or we got a gap
            elif c == cols-1 or s[r,c] != player:
                seq_cnt = 0

                
            # increment to next column
            c += 1    
    
    return cnt

def horizontalLines(state, player):
    '''Scans for counts of horizontal lines for player pieces of 2 or more.
    
    ARGUMENTS:
        state - game state   
        player - player number to check
    
    RETURNS:    
        dictionary
    '''
    results = {key: 0 for key in range(2,8)}

    # Convert state to numpy ndarray which is easier to check for winning state
    s = stateAsNdarray(state)
    rows = s.shape[0]
    cols = s.shape[1]
    
    # Check for horizontal consecutive pieces in a row. 
    # Go down rows of the board from top to bottom
    for r in range(rows):
        # Reset column and sequential count
        c = 0
        seq_cnt = 0
        
        # Start scanning horizontally across the row
        while c < cols:
            # If cell is target player value then increment sequential count
            if s[r,c] == player:
                seq_cnt += 1
            
            # If last column or we got a gap
            if c == cols-1 or s[r,c] != player:
                if seq_cnt > 1:
                    results[seq_cnt] += 1
                seq_cnt = 0
                
            # increment to next column
            c += 1
                               
    return results

def verticalLines(state, player):
    '''Scans for counts of vertical lines for player pieces of 2 or more.
    
    ARGUMENTS:
        state - game state   
        player - player number to check
    
    RETURNS:    
        dictionary
    '''
    results = {key: 0 for key in range(2,7)}

    # Convert state to numpy ndarray which is easier to check for winning state
    s = stateAsNdarray(state)
    rows = s.shape[0]
    cols = s.shape[1]
    
    # Check for vertical consecutive pieces in a row. 
    # Go across columns of the board from left to right
    for c in range(cols):
        # Reset row and sequential count
        r = 0
        seq_cnt = 0
        
        # Start scanning vertically down the column
        while r < rows:
            # If cell is target player value then increment sequential count
            if s[r,c] == player:
                seq_cnt += 1
                
            # If last row or we got a 'gap'
            if r == rows-1 or s[r,c] != player:
                if seq_cnt > 1:
                    results[seq_cnt] += 1
                seq_cnt = 0
                
            # increment to next row
            r += 1
                               
    return results



def diagonalDownLines(state, player):
    '''Scans for counts of diagonal down lines for player pieces of 2 or more.
    
    ARGUMENTS:
        state - game state   
        player - player number to check
    
    RETURNS:    
        dictionary
    '''
    results = {key: 0 for key in range(2,7)}

    # Convert state to numpy ndarray which is easier to check for winning state
    s = stateAsNdarray(state)
    rows = s.shape[0]
    cols = s.shape[1]
    
    # Check for diagonal down consecutive pieces for player. 
    # Go across columns of the board from 0 to 5, and scan diagonally downward for lines
    r = 0
    for c in range(cols-1):
        # Reset diagonal counter and sequential count
        x = 0
        seq_cnt = 0
        
        # Start scanning diagonally down
        while x < rows-1:   # This is the max cells we go diagonally, but may break early
            # If cell is target player value then increment sequential count
            if s[r+x,c+x] == player:
                seq_cnt += 1
                
            # If last row or last col or we got a 'gap'
            if r+x == rows-1 or c+x == cols-1 or s[r+x,c+x] != player:
                if seq_cnt > 1:
                    results[seq_cnt] += 1
                seq_cnt = 0
                
            # increment diagonal counter
            x += 1
            
            # check if we reached out diagonal limits and need to break
            if r+x >= rows or c+x >= cols:
                break
    
    # Check for diagonal down lines
    # Go down rows of board from 1 to 4, and scan diagonally downward for lines
    c = 0
    for r in range(1, rows-1):
        # Reset diagonal counter and sequential count
        x = 0
        seq_cnt = 0
        
        # Start scanning diagonally down
        while x < rows-1:  # This is the max cells we go diagonally, but may break early
            # If cell is target player value then increment sequential count
            if s[r+x,c+x] == player:
                seq_cnt += 1
                
            # If last row or we got a 'gap'
            if r+x == rows-1 or s[r+x,c+x] != player:
                if seq_cnt > 1:
                    results[seq_cnt] += 1
                seq_cnt = 0                

            # increment diagonal counter
            x += 1
            
            # check if we reached out diagonal limits and need to break
            if r+x >= rows:  # checking row is sufficient
                break    
    
    return results


def diagonalUpLines(state, player):
    '''Scans for counts of diagonal up lines for player pieces of 2 or more.
    
    ARGUMENTS:
        state - game state   
        player - player number to check
    
    RETURNS:    
        dictionary
    '''
    results = {key: 0 for key in range(2,7)}

    # Convert state to numpy ndarray which is easier to check for winning state
    s = stateAsNdarray(state)
    rows = s.shape[0]
    cols = s.shape[1]
    
    # Check for diagonal down consecutive pieces for player. 
    # Go across columns of the board from 0 to 5, and scan diagonally downward for lines
    r = 0
    for c in range(1, cols):
        # Reset diagonal counter and sequential count
        x = 0
        seq_cnt = 0
        
        # Start scanning diagonally down
        while x < rows-1:   # This is the max cells we go diagonally, but may break early
            # If cell is target player value then increment sequential count
            if s[r+x,c-x] == player:
                seq_cnt += 1
                
            # If last row or we got a 'gap'
            if r+x == rows-1 or s[r+x,c-x] != player:
                if seq_cnt > 1:
                    results[seq_cnt] += 1
                seq_cnt = 0
                
            # increment diagonal counter
            x += 1
            
            # check if we reached out diagonal limits and need to break
            if r+x >= rows or c-x < 0:
                break
    
    # Check for diagonal down lines
    # Go down rows of board from 1 to 4, and scan diagonally downward for lines
    c = cols - 1
    for r in range(1, rows-1):
        # Reset diagonal counter and sequential count
        x = 0
        seq_cnt = 0
        
        # Start scanning diagonally down
        while x < rows-1:  # This is the max cells we go diagonally, but may break early
            # If cell is target player value then increment sequential count
            if s[r+x,c-x] == player:
                seq_cnt += 1
                
            # If last row or we got a 'gap'
            if r+x == rows-1 or s[r+x,c-x] != player:
                if seq_cnt > 1:
                    results[seq_cnt] += 1
                seq_cnt = 0                

            # increment diagonal counter
            x += 1
            
            # check if we reached out diagonal limits and need to break
            if r+x >= rows:  # checking row is sufficient
                break    
    
    return results


def boardScore(state, player):
    v_lines = verticalLines(state, player)
    h_lines = horizontalLines(state, player)
    dd_lines = diagonalDownLines(state, player)
    du_lines = diagonalUpLines(state, player)
    
    score = linesScore(v_lines) + linesScore(h_lines) + linesScore(dd_lines) + linesScore(du_lines)
    
    return min(score, 1.)


def linesScore(lines):
    return (lines[2] * 0.01 + lines[3] * 0.05) if lines[4] <= 0 else 1. 
    #return lines[2] * 10 + lines[3] * 100 + lines[4] * 1000


def individualPiecesPlacementScore(state, player):
    s = stateAsNdarray(state)
    player_move_cnt = 0
    player_score = 0.
    other_player_move_cnt = 0
    other_player_score = 0.
    
    rows, cols = s.shape
    
    # Scoring utility based on strategic placement of individual pieces on board, with higher utility given for 
    # more pieces placed in the center of board, and decreasing as we move out in a radius.
    for row in range(rows):
        for col in range(cols):
            score = ((1 - abs(col - 3.)/3.) + (1 - abs(row - 3)/3.)) / 2.
            score *= .1
            if s[row, col] == player:
                player_score += score
                player_move_cnt += 1
            elif s[row, col] == otherPlayer(player):
                other_player_score += score
                other_player_move_cnt += 1
    
    # Compute the average score of pieces placement for each player
    player_score_avg = player_score / player_move_cnt
    other_player_score_avg = other_player_score / other_player_move_cnt    
    
    return player_score_avg, other_player_score_avg


def utilityEstimated(state, player, playerWinsF):
    '''Estimated payoff for player on given state, based on strategic score of pieces on board.
    
    ARGUMENTS:
        state: game state
        player (int): 1 or 2
        playerWinsF: function that tests if player wins with given state - playerWinsF(state, player) -> boolean
        
    RETURNS:
        int
    '''
    #if playerWinsF(state, player):
    #    value = 1
    #elif playerWinsF(state, otherPlayer(player)):
    #    value = -1
    #else:
    #    value = 0
    
    player_inv_pieces_score, other_player_inv_pieces_score = individualPiecesPlacementScore(state, player)

    player_win_combos = (winningDiagonalUpCombination(state, player) +
                         winningDiagonalDownCombination(state, player) +
                         winningHorizontalCombination(state, player) )
    other_player_win_combos = (winningDiagonalUpCombination(state, otherPlayer(player)) +
                               winningDiagonalDownCombination(state, otherPlayer(player)) +
                               winningHorizontalCombination(state, otherPlayer(player)) )
    
    player_score = min(1, max(boardScore(state, player) + player_inv_pieces_score, player_win_combos))
    other_player_score = min(1, max(boardScore(state, otherPlayer(player)) + other_player_inv_pieces_score, other_player_win_combos))
    
    #player_score = min(boardScore(state, player) + player_inv_pieces_score, 1.)
    #other_player_score = min(boardScore(state, otherPlayer(player)) + other_player_inv_pieces_score, 1.)
                                    
    if player_score > other_player_score:
        return player_score
    else:
        return -1 * other_player_score



