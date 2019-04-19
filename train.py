import torch
import random
from IPython.display import clear_output
from collections import OrderedDict
from trainhistory import *
from connectfour import *
from minimax import otherPlayer
from util import *

def createNNBuildingBlocks(nn_layers, learning_rate=1e-3):
    '''Creates and returns an untrained Q neural network model, cloned Q target model, a loss function, and
    an optimizer that can be used for training the Q model.
    
    INPUTS:
        nn_layers - list of integers specifying the number of nodes in each layer
        learning_rate - the learning rate used to initialize the optimizer, defaults to 1x10^-3
        
    RETURNS:
        model, model_target, loss_f, optimizer
    '''

    nn_modules = []
    last_i = 0
    for i in range(1, len(nn_layers)-1):
        nn_modules.append( ('linear' +  str(i), torch.nn.Linear(nn_layers[last_i], nn_layers[i])) )
        nn_modules.append( ('relu' + str(i), torch.nn.ReLU()) )
        last_i = i
                       
    nn_modules.append( ('linear' + str(len(nn_layers)-1), torch.nn.Linear(nn_layers[len(nn_layers)-2], nn_layers[len(nn_layers)-1])) )
        
    print(nn_modules)
    
    model = torch.nn.Sequential(OrderedDict(nn_modules))
    
        
    # Create the target network with a deep copy
    model_target = copy.deepcopy(model) 

    # Copy to GPU memory
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = getDevice()
    model.to(device)
    model_target.to(device)    
    
    # Create the objective loss function
    loss_f = torch.nn.MSELoss(reduction='elementwise_mean')     # size_average=True 

    # Create the optimizer, passing in reference to parmeter weights of the model
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    
    return model, model_target, loss_f, optimizer


def selectMoveByEpsilonGreedyPolicy(epsilon, state, Q, validMovesF=None):
    '''Selects a move using Epsilon Greedy policy, either randomly or action with best Q value from Q model.
    
    INPUTS:
        epsilon - current epsilon that controls probability of random action selection versus max Q value
        state - current game state
        Q - current learned Q neural network model
        validMovesF - function pointer that returns valid moves for given game state
    
    RETURNS:
        action (int) - selected action representing column 
    '''
    # Convert game state to a tensor
    state_tensor = torch.tensor(stateAsNdarray(state).reshape(1, 6*7), dtype=torch.float).to(getDevice())

    # Forward fee to get Q value for state, converting to numpy ndarray
    qval = Q(state_tensor)
    qval_np = qval.cpu().data.numpy()
    
    # Get the list of valid moves for state
    actions = validMovesF(state)
    
    if (random.random() < epsilon):
        # Randomly select an action from list of valid moves
        rnd_indx = np.random.randint(0, len(actions))
        action = actions[rnd_indx]
    else:
        # Set invalid actions (game columns) to -INF so they won't be selected
        valid_moves = validMovesF(state)
        for col in range(qval_np.shape[1]):
            if col not in valid_moves:
                qval_np[0, col] = -float('inf')
        
        # Select action based on largest Q value
        action = np.argmax(qval_np)

    return action    


def buildTrainingMiniBatch(replayBuffer, batchSize, gamma, Q, Q_target, currentReward):
    '''Randomly selects batch size training samples from the experience replay buffer, and computes the target vslues,
    returned in X_train and Y_train tensors on the GPU if available. 
    
    INPUTS:
        replayBuffer
        batchSize
        gamma
        Q
        Q_target
        reward
    
    RETURNS:
        X_train, Y_train 
    
    '''
    # Number of columns in Connect Four which translates into possible game actions
    cols = 7
    
    # Allocate empty tensors on GPU to store our batch of samples to train on 
    X_train = torch.empty(batchSize, cols, dtype=torch.float).to(getDevice())
    Y_train = torch.empty(batchSize, cols, dtype=torch.float).to(getDevice())

    # Select random samples from experience replay buffer
    minibatch = random.sample(replayBuffer, batchSize)

    h = 0
    for old_state, action_m, reward_m, new_state_m in minibatch:
        old_qval = Q(old_state) 

        newQ = Q(new_state_m)   #Q_target(new_state_m)
        maxQ = torch.max(newQ)
        
        y = torch.zeros((1, cols)).to(getDevice())
        y[:] = old_qval[:]
        
        if currentReward == 0:
            update = (reward_m + (gamma * maxQ))
        else:
            update = currentReward  #reward_m
            
        y[0][action_m] = update
        X_train[h] = old_qval
        Y_train[h] = y  #Variable(y)
        h+=1    
    
    return X_train, Y_train


def trainQLearningNN(nn_layers,
                     agent,
                     terminalTestF, playerWinsF, validMovesF, makeMoveF,
                     initialGameStateF, stateNdarrayConverterF, printStateF, 
                     epochs=200, 
                     batchSize=250, bufferSize=1000, 
                     epsilon=1.0, min_epsilon=0.1, gamma=0.9, 
                     verbose=True):
    '''Train neural network as the Q function for Reinforcement Learning.
    
    INPUTS:
        nn_layers - list of integers that specifies size of each NN layer, including input and output layers
        agent - opponent agent used for training
        epochs - total full game iterations to train for
        batchSize - number of samples to randomly select from buffer for training
        buffer - size of experience replay buffer
        epsilon - startig propability of selecting a random move per episilon greedy policy
        gamma - 
        
    RETURNS:
        model - trained NN approximating the Q function
        losses - list of loss value at end of each epoch
        accuracies - accuracy of model at end of each epoch
        
    '''
    trainHist = createTrainingHistoryData(nn_layers, epochs=epochs, batchSize=batchSize, bufferSize=bufferSize, 
                                          startEpsilon=epsilon, minEpsilon=min_epsilon, gamma=gamma)
    
    device = getDevice()

    train_start_time = time.time()

    # Train Q network for player 1
    q_player = 1
    
    # Construct the NN building blocks: NN model, NN target model, loss function, and optimizer
    model, model_target, loss_f, optimizer = createNNBuildingBlocks(nn_layers)
    
    if agent.Q is None:
        agent.setQ(model)
    
    accuracies = []    # historical accuracy after each epoch
    losses = []        # historical loss after each epoch
    replay = []        # the experience replay buffer
    c = 500            # target network update step size
    c_step = 0
    h = 0  
    epoch_comp = 0     # don't count epoch iterations where we're just building experience reply buffer
    epoch_time = 0.
    epoch_start_time = 0.
    epoch_total_time = 0.
    epoch_avg_time = 0.
    epoch_iteration = 1
    est_training_time_left = -1
    
    # REMOVE: Gamegrid specific detail
    #max_moves = 9

    # Local var to track performance of model
    acc = 0. 
    avg_loss = 0.
    curr_loss = 0.
    
    done = False
    while epoch_iteration <= epochs:
        epoch_start_time = time.time()
        
        # CHANGE: Start a new game
        #game = Gridworld(size=4, mode='random')
        
        # CHANGE: Get initial state of game
        #state_np = game.board.render_np().reshape(1,64) + np.random.rand(1,64)/100.0
        #state = torch.tensor(state_np, dtype=torch.float).to(device)
        
        # Start a new Connect 4 game
        state_l = initialGameStateF()
        state_np = stateNdarrayConverterF(state_l).reshape(1,6*7) + np.random.rand(1,6*7)/1000.0
        state = torch.tensor(state_np, dtype=torch.float).to(device)
        
        # Indicator for when current game is over
        status = 1  
        
        moves_total = 0
        mov = 0
        loss_total = 0.        
        
        # The game loop for this epoch. Continue playing until current game is over.
        while not terminalTestF(state_l):
            # Synch Q network model with the target network model every c steps
            c_step += 1
            if c_step > c:
                model_target.load_state_dict(model.state_dict())
                c_step = 0
                
                
            mov += 1
            
            # Select a move using epsilon greedy policy and upate game board using the action         
            action = selectMoveByEpsilonGreedyPolicy(epsilon=epsilon, 
                                                     state=state_l, 
                                                     Q=model, 
                                                     validMovesF=validMovesF)
            #game.makeMove(action)
            
            # Get new game state resulting from move taken
            new_state_l = makeMoveF(state_l, action)
            new_state_np = stateNdarrayConverterF(new_state_l).reshape(1,6*7) + np.random.rand(1,6*7)/1000.0
            new_state = torch.tensor(new_state_np, dtype=torch.float).to(device)
            #new_state_np = game.board.render_np().reshape(1,64) + np.random.rand(1,64)/100.0
            #new_state = torch.tensor(new_state_np, dtype=torch.float).to(device)
            
            # Get reward
            if playerWinsF(new_state_l, q_player):
                reward = 1
            elif terminalTestF(new_state_l):
                reward = 0
            else:
                next_new_state_l = agent.takeAction(new_state_l)
                
                if playerWinsF(next_new_state_l, otherPlayer(q_player)):
                    reward = -1
                elif terminalTestF(next_new_state_l):
                    reward = 0
                else:
                    reward = 0
                    
                new_state_l = next_new_state_l
                new_state_np = stateNdarrayConverterF(new_state_l).reshape(1,6*7) + np.random.rand(1,6*7)/1000.0
                new_state = torch.tensor(new_state_np, dtype=torch.float).to(device)                

                
                    
            #reward = game.reward()
            
            #if mov > max_moves:
            #    reward = -5
                
            # Add experience to the replay buffer while its not full yet
            # Will add up to buffer size before we start any batch training
            if len(replay) < bufferSize: 
                replay.append((state, action, reward, new_state))
                
                clear_output(wait=True)
                print("Filling experience replay memory: {}/{}".format(len(replay), bufferSize))
                printStateF(new_state_l)
                
            # Experience replay buffer is now full, so we will overwrite least recently added value
            # with new experience value.  We will also start the training by selecting random batch
            # of samples from the the replay buffer for training (backprop gradient update using optimizer)
            else: 
                # Remove the first item from replay buffer (oldest added item) and append current experience
                # to end of buffer
                replay.pop(0)
                replay.append((state, action, reward, new_state))

                # Randomly select batchSize number of samples from the experience replay buffer for use as
                # mini batch training data. X_train and Y_train returned are GPU tensors.
                X_train, Y_train = buildTrainingMiniBatch(replayBuffer=replay, 
                                                          batchSize=batchSize, 
                                                          gamma=gamma, 
                                                          Q=model, 
                                                          Q_target=model_target,
                                                          currentReward=reward)
                
                # Compute difference (loss) between predicted (X_train) and target (Y_train)  
                loss = loss_f(X_train, Y_train)
                
                # Used to calculate average loss when this epoch is done
                curr_loss = loss.item()
                loss_total += curr_loss
                moves_total += 1
                
                if verbose and moves_total > 0:
                    clear_output(wait=True)
                    print("epoch: {}/{}, accuracy: {}%, average loss: {:.2f}, current loss: {:.2f}, average epoch time: {:.1f}, last epoch time: {:.1f}, estimated training time left: {:.1f}, elapsed: {:.1f}, epsilon: {:.4f}".format(epoch_iteration, epochs, int(acc * 100), avg_loss, loss.item(), epoch_avg_time, epoch_time, est_training_time_left, time.time() - train_start_time, epsilon))        
                    printStateF(new_state_l)
                                 
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


            #if terminalTestF(new_state_l):
            #    status = 1
                
            state_l = new_state_l

            #if reward != -1:
            #    status = 0
            #    mov = 0
        
        

        # Update historical accuracy and loss, after we have started actual training when the buffer is full
        if len(replay) == bufferSize:
            epoch_time = time.time() - epoch_start_time
            epoch_iteration += 1
            epoch_total_time += epoch_time
            epoch_avg_time = epoch_total_time / epoch_iteration
            est_training_time_left = (epochs - epoch_iteration) * epoch_avg_time
            if est_training_time_left < 0.:
                est_training_time_left = 0.
                
            gameResult = 0
            if playerWinsF(state_l, 1):
                gameResult = 1
            elif playerWinsF(state_l, 2):
                gameResult = 2

            
            
            # Compute average loss from this epoch
            if moves_total > 0:
                avg_loss = loss_total / moves_total
                losses.append(avg_loss)

            # Update and save training history after each epoch
            updateTrainingHistoryData(trainHist, 
                                      loss=avg_loss, 
                                      acc=acc, 
                                      epsilon=epsilon, 
                                      trainTime=epoch_time, 
                                      moves=moves_total,
                                      gameResult=gameResult,
                                      model=model,
                                      epochAvgTrainTime=epoch_avg_time, 
                                      totalTrainTime=epoch_total_time)

            saveTrainingHistory(trainHist)

            if verbose and epoch_iteration > 0:
                clear_output(wait=True)
                print("epoch: {}/{}, accuracy: {}%, average loss: {:.2f}, current loss: {:.2f}, average epoch time: {:.1f}, last epoch time: {:.1f}, estimated training time left: {:.1f}, elapsed: {:.1f}, epsilon: {:.4f}".format(epoch_iteration, epochs, int(acc * 100), avg_loss, curr_loss, epoch_avg_time, epoch_time, est_training_time_left, time.time() - train_start_time, epsilon))        
                printStateF(new_state_l)
                #clear_output(wait=True)
                #print("epoch: {}/{}, accuracy: {}%, average loss: {}, current loss: {}, average epoch time: {}, last epoch time: {}, estimated training time left: {}".format(epoch_iteration, epochs, int(acc * 100), avg_loss, -1, epoch_avg_time, epoch_time, est_training_time_left))        
                #print("epoch: {}/{}, accuracy: {}%, average loss: {:.2f}, current loss: {:.2f}, average epoch time: {:.1f}, last epoch time: {:.1f}, estimated training time left: {:.1f}, elapsed: {:.1f}, epsilon: {:.4f}".format(epoch_iteration, epochs, int(acc * 100), avg_loss, -1, epoch_avg_time, epoch_time, est_training_time_left, time.time() - train_start_time, epsilon))        

            # Compute accuracy form this epoch, by playing 50 games 
            #accuracies.append(test_win_perc(copy.deepcopy(model).to('cpu'), max_games=50, verbose=False))
            #acc = accuracies[-1]
        else:
            epoch_comp += 1
        
     
            
        # Decay epsilon after every training epoch
        if len(replay) == bufferSize and epsilon > min_epsilon:
            epsilon = epsilon - (1. / epochs)
            if epsilon < min_epsilon:
                epsilon = min_epsilon
            
    return model, trainHist




