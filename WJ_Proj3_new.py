import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers


class player():
    # class initiation
    def __init__(self, name="PlayerName", ball = None):
        self.name = name
        self.score = 0
        ## player's position in x-axis: 1 or 2
        self.posX = 2
        ## player's position in y-axis: 0 or 1
        self.posY = 0
        ## player's position in flattened 1D array: 1/2/5/6
        self.pos = 2
        ## ball=0 means playerA has ball, ball=1 means playerB has ball
        self.ball = ball
        ## reward
        self.reward = 0
        print("Player {} Created".format(name))

    # calculate flattened position from 2D to 1D array
    def flattern_pos():
        self.pos = self.posX + self.posY * self.cols

    def has_ball(self):
        return self.ball


class soccer_env():
    # class initiation
    def __init__(self, playerA, playerB, rows=2, cols=4):
        self.playerA = playerA
        self.playerB = playerB
        self.rows = rows
        self.columns = cols
        ## A to score a goal: must reach either (0,0) or (0,1)
        self.goal_posA = [i*self.columns for i in range(self.rows)]
        ## B to score a goal: must reach either (3,0) or (3,1)
        self.goal_posB = [(i+1)*self.columns-1 for i in range(self.rows)]
        ## initialize B has ball
        self.ball = playerB.ball
        self.ball_pos = playerB.pos

    # follow notation in paper, state includes players' positions and who owns the ball
    def state(self):
        return [playerA.pos, playerB.pos, game.ball]

    # flattern 2D array (x,y) into 1D array (x + y*cols)
    def flattern_pos(self, x, y):
        return (x + y*self.columns)

    def create_new_env(self):
        ## playerA: (2,0)
        self.playerA.posX = 2
        self.playerA.posY = 0
        self.playerA.pos = self.flattern_pos(self.playerA.posX, self.playerA.posY)
        ## playerB: (1,0)
        self.playerB.posX = 1
        self.playerB.posY = 0
        self.playerB.pos = self.flattern_pos(self.playerB.posX, self.playerB.posY)
        ## initialize B has ball
        self.ball = playerB.ball
        self.ball_pos = playerB.pos

    # move single player
    # not move conditions:
    ## 1) already at boarder
    ## 2) will reach goal position
    def move_single_player(self, player, action):
        ## Move North
        if ( (action == 0) and (player.pos >= self.columns) ):
            player_new_pos = player.pos - self.columns
        ## Move East
        elif ( (action == 1) and (player.pos not in self.goal_posB) ):
            player_new_pos = player.pos + 1
        ## Move South
        elif ( (action == 2) and (player.pos < self.columns) ):
            player_new_pos = player.pos + self.columns
        ## Move West
        elif ( (action == 3) and (player.pos not in self.goal_posA) ):
            player_new_pos = player.pos - 1
        ## Stick
        else:
            player_new_pos = player.pos
        ## return player's new position
        return player_new_pos

    # move both players, including ball exchange
    def player_action_pair(self, player1, player2, action1, action2):
        ## calculate players' new positions
        player1_new_pos = self.move_single_player(player1, action1)
        player2_new_pos = self.move_single_player(player2, action2)
        ## player1 is the one who moves first
        ### Case 1.1: no collision -> move
        if player1_new_pos != player2.pos:
            player1.pos = player1_new_pos
        ### Case 1.2: collision -> ball changes possession
        else:
            self.ball = player2.ball
        ## player2 is the one who moves second
        ### Case 2.1: no collision -> move
        if player2_new_pos != player1.pos:
            player2.pos = player2_new_pos
        ### Case 2.2: collision -> ball changes possession
        else:
            self.ball = player1.ball

        ## assign ball's position to player A/B based on the value of self.ball
        if self.ball:
            self.ball_pos = playerB.pos
        else:
            self.ball_pos = playerA.pos

    # move both players
    def move_both_players(self, actionA, actionB):
        ## Randomly choose playerA / playerB to act first
        if np.random.randint(2) == 0:
            self.player_action_pair(self.playerA, self.playerB, actionA, actionB)
        else:
            self.player_action_pair(self.playerB, self.playerA, actionB, actionA)

        ## Case 1: playerA scores a goal (ball's position in [0, 4])
        if (self.ball_pos in self.goal_posA):
            playerA.reward = 100
            playerB.reward = -100
            done = 1
        ## Case 2: playerB scores a goal (ball's position in [3, 7])
        elif (self.ball_pos in self.goal_posB):
            playerA.reward = -100
            playerB.reward = 100
            done = 1
        ## Case 3: no player socres a goal
        else:
            playerA.reward = 0
            playerB.reward = 0
            done = 0

        ## return playerA's position, playerB's position, ball's possession, and if a game finished
        return self.state(), done


# print to replicate Fig. 3 in paper
def plot_Q_Diff(Q_diff, iter_list, name="Q-learner"):
    plt.plot(iter_list, Q_diff, color='black', linewidth=0.5)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.title(name)
    plt.xlabel('Simulation Iteration')
    plt.ylabel('Q-value Difference')
    plt.ylim(0, 0.5)
    plt.show()
    plt.gcf().clear()


# Q-Learning
def Q_learning(game, playerA, playerB, iterations=10**6, gamma=0.9):
    ## exploration rate
    epsilon = 0.5
    ## alpha
    alpha = 1.0
    alpha_min = 0.10
    alpha_decay = (alpha - alpha_min) / iterations
    ## 8 possible possible positions in 1D flattened array for both players
    playerA_pos_array = game.rows*game.columns
    playerB_pos_array = game.rows*game.columns
    ## 2 possible ball possessions (belongs to playerA / playerB)
    ball_array = 2
    ## 5 possible actions for playerA (NESW, stick)
    playerA_action_array = 5
    ## Q table for playerA
    playerA_Q_table = np.zeros([playerA_pos_array, playerB_pos_array, ball_array, playerA_action_array])

    ## create new game environment, and set done to be 0 to begin with
    game.create_new_env()
    done = 0

    ## record difference in Q table for successful goal
    Q_diff = []
    iter_list = []

    ## loop for 1e6 times
    for i in range(iterations):
        ### check if a successful goal has finished -> reset
        if done == 1:
            game.create_new_env()
            done = 0

        ### record last time's Q value
        Q_backup = playerA_Q_table[2, 1, 1, 2]

        ### initialize state
        playerA_pos = playerA.pos
        playerB_pos = playerB.pos
        who_owns_ball = game.ball
        curr_state = [playerA_pos, playerB_pos, who_owns_ball]

        # Epsilon-Greedy Search for Q-Learning, as suggested in paper
        if epsilon > np.random.random():
            actionA = np.random.choice(playerA_action_array)
        else:
            actionA = np.argmax(playerA_Q_table[playerA_pos, playerB_pos, who_owns_ball])
        # actionB is rancomly chosen from the 5 possible actions
        actionB = np.random.choice(playerA_action_array)

        ### update state, after moving both players for 1 step
        new_state, done = game.move_both_players(actionA, actionB)
        playerA_new_pos, playerB_new_pos, who_owns_ball_new = new_state
        
        ### update Q-table for playerA only
        playerA_Q_table[playerA_pos, playerB_pos, who_owns_ball, actionA] = \
                                (1 - alpha) * playerA_Q_table[playerA_pos, playerB_pos, who_owns_ball, actionA] + \
                                alpha * ((1 - gamma) * playerA.reward + \
                                gamma * np.max(playerA_Q_table[playerA_new_pos, playerB_new_pos, who_owns_ball_new]))

        ### only calculate difference in Q when reaches initial state again
        if [playerA_pos, playerB_pos, who_owns_ball, actionA, actionB] == [2, 1, 1, 2, 4]:
            Q_diff.append(abs(playerA_Q_table[2, 1, 1, 2] - Q_backup))
            iter_list.append(i)
            print("Iteration=", i, ", alpha=", alpha)

        ### decay alpha
        alpha -= alpha_decay

    # graph Q_diff
    plot_Q_Diff(Q_diff, iter_list, name="Q-Learner")


# Friend-Q Learning
def Friend_Q_learning(game, playerA, playerB, iterations=10**6, gamma=0.9):
    ## alpha decay
    alpha = 1.0
    alpha_min = 0.001
    alpha_decay = (alpha - alpha_min) / iterations
    ## 8 possible possible positions in 1D flattened array for both players
    playerA_pos_array = game.rows*game.columns
    playerB_pos_array = game.rows*game.columns
    ## 2 possible ball possessions (belongs to playerA / playerB)
    ball_array = 2
    ## 5 possible actions for both players (NESW, stick)
    playerA_action_array = 5
    playerB_action_array = 5
    ## Q table for both players
    players_Q_table = np.zeros([playerA_pos_array, playerB_pos_array, ball_array, playerA_action_array, playerB_action_array])

    ## create new game environment, and set done to be 0 to begin with
    game.create_new_env()
    done = 0

    ## record difference in Q table for successful goal
    Q_diff = []
    iter_list = []

    ## loop for 1e6 times
    for i in range(iterations):
        ### check if a successful goal has finished -> reset
        if done == 1:
            game.create_new_env()
            done = 0

        ### record last time's Q value
        Q_backup = players_Q_table[2, 1, 1, 2, 4]

        ### initialize state
        playerA_pos = playerA.pos
        playerB_pos = playerB.pos
        who_owns_ball = game.ball
        curr_state = [playerA_pos, playerB_pos, who_owns_ball]

        ### randomly chosen actions for both players
        actionA = np.random.randint(playerA_action_array)
        actionB = np.random.randint(playerB_action_array)

        ### update state, after moving both players for 1 step
        new_state, done = game.move_both_players(actionA, actionB)
        playerA_new_pos, playerB_new_pos, who_owns_ball_new = new_state

        ### update joint Q-table for both players
        players_Q_table[playerA_pos, playerB_pos, who_owns_ball, actionA, actionB] = \
                    (1 - alpha) * players_Q_table[playerA_pos, playerB_pos, who_owns_ball, actionA, actionB] + \
                    alpha * ((1 - gamma) * playerA.reward + \
                    gamma * np.max(players_Q_table[playerA_new_pos, playerB_new_pos, who_owns_ball_new]))

        ### only calculate difference in Q when reaches initial state again
        if [playerA_pos, playerB_pos, who_owns_ball, actionA, actionB] == [2, 1, 1, 2, 4]:
            Q_diff.append(abs(players_Q_table[2, 1, 1, 2, 4] - Q_backup))
            iter_list.append(i)
            print("Iteration=", i, ", alpha=", alpha)

        ### alpha decay
        alpha -= alpha_decay

    # graph Q_diff
    plot_Q_Diff(Q_diff, iter_list, name="Friend-Q")


# Foe-Q Learning
def Foe_Q_learning(game, playerA, playerB, iterations=10**6, gamma=0.9):
    ## alpha
    alpha = 1.0
    alpha_min = 0.001
    alpha_decay = (alpha - alpha_min) / iterations
    ## 8 possible possible positions in 1D flattened array for both players
    playerA_pos_array = game.rows*game.columns
    playerB_pos_array = game.rows*game.columns
    ## 2 possible ball possessions (belongs to playerA / playerB)
    ball_array = 2
    ## 5 possible actions for both players (NESW, stick)
    playerA_action_array = 5
    playerB_action_array = 5
    ## Q table for both players
    players_Q_table = np.zeros([playerA_pos_array, playerB_pos_array, ball_array, playerA_action_array, playerB_action_array])

    ## create new game environment, and set done to be 0 to begin with
    game.create_new_env()
    done = 0

    ## record difference in Q table for successful goal
    Q_diff = []
    iter_list = []

    ## loop for 1e6 times
    for i in range(iterations):
        ### check if a successful goal has finished -> reset
        if done == 1:
            game.create_new_env()
            done = 0
        
        ### record last time's Q value
        Q_backup = players_Q_table[2, 1, 1, 2, 4]
        
        ### initialize state
        playerA_pos = playerA.pos
        playerB_pos = playerB.pos
        who_owns_ball = game.ball
        curr_state = [playerA_pos, playerB_pos, who_owns_ball]
        ### current Q-table
        current_Q = players_Q_table[playerA_pos, playerB_pos, who_owns_ball]

        ### randomly chosen actions for both players
        actionA = np.random.randint(playerA_action_array)
        actionB = np.random.randint(playerB_action_array)

        ### update state, after moving both players for 1 step
        new_state, done = game.move_both_players(actionA, actionB)
        #playerA_new_pos, playerB_new_pos, who_owns_ball_new = new_state

        ### LP to find equilibrium
        #### build matrix A
        M = matrix(current_Q).trans()
        n_col = M.size[1]
        A = np.hstack((np.ones((M.size[0], 1)), M))
        A = np.vstack((A, np.hstack((np.zeros((n_col, 1)), -np.eye(n_col)))))
        A = matrix(np.vstack((A, np.hstack((0,np.ones(n_col))), np.hstack((0,-np.ones(n_col))))))
        #### build vector b
        b = matrix(np.hstack((np.zeros(A.size[0] - 2), [1, -1])))
        #### build vector c
        c = matrix(np.hstack(([-1], np.zeros(n_col))))
        #### LP solver
        solvers.options['glpk'] = {'msg_lev': 'GLP_MSG_OFF'}
        sol = solvers.lp(c,A,b, solver='glpk')
        equilibrium = sol['primal objective']

        ### update joint Q-table for both players
        players_Q_table[playerA_pos, playerB_pos, who_owns_ball, actionA, actionB] = \
                    (1 - alpha) * players_Q_table[playerA_pos, playerB_pos, who_owns_ball, actionA, actionB] + \
                    alpha * ((1 - gamma) * playerA.reward + gamma * equilibrium)

        ### only calculate difference in Q when reaches initial state again
        if [playerA_pos, playerB_pos, who_owns_ball, actionA, actionB] == [2, 1, 1, 2, 4]:
            Q_diff.append(abs(players_Q_table[2, 1, 1, 2, 4] - Q_backup))
            iter_list.append(i)
            print("Iteration=", i, ", alpha=", alpha)

        ### alpha decay
        alpha -= alpha_decay

    # graph Q_diff
    plot_Q_Diff(Q_diff, iter_list, name="Foe-Q")



# Returns the expected returns of playerA_Q_table_state and playerB_Q_table_state
def LP_Solver_uCEQ(playerA_Q_table_state, playerB_Q_table_state):
    ## build matrix A
    M = matrix(playerA_Q_table_state).trans()
    n = M.size[1]
    A = np.zeros((2 * n * (n - 1), (n * n)))
    playerA_Q_table_state = np.array(playerA_Q_table_state)
    playerB_Q_table_state = np.array(playerB_Q_table_state)
    row = 0
    ## Correlation matrix
    for i in range(n):
        for j in range(n):
            if i != j:
                A[row, i * n:(i + 1) * n] = playerA_Q_table_state[i] - playerA_Q_table_state[j]
                A[row + n * (n - 1), i:(n * n):n] = playerB_Q_table_state[:, i] - playerB_Q_table_state[:, j]
                row += 1
    A = matrix(A)
    A = np.hstack((np.ones((A.size[0], 1)), A))
    A = np.vstack((A, np.hstack((np.zeros((n*n, 1)), -np.eye(n*n)))))
    A = matrix(np.vstack((A, np.hstack((0,np.ones(n*n))), np.hstack((0,-np.ones(n*n))))))
    ## build vector b
    b = matrix(np.hstack((np.zeros(A.size[0] - 2), [1, -1])))
    ## build vector c
    c = matrix(np.hstack(([-1.], -(playerA_Q_table_state+playerB_Q_table_state).flatten())))
    ## LP solver
    solvers.options['glpk'] = {'msg_lev': 'GLP_MSG_OFF'}
    sol = solvers.lp(c,A,b, solver='glpk')
    ## Corner case: no solution -> set to 0
    if sol['x'] is None:
        return 0, 0
    else:
        playerA_reward_expect = np.matmul(playerA_Q_table_state.flatten(), sol['x'][1:])[0]
        playerB_reward_expect = np.matmul(playerB_Q_table_state.transpose().flatten(), sol['x'][1:])[0]
        return playerA_reward_expect, playerB_reward_expect


# uCE-Q Learning
def uCE_Q_Learning(game, playerA, playerB, iterations=10**6, gamma=0.9):
    ## alpha
    alpha = 1.0
    alpha_min = 0.001
    alpha_decay = (alpha - alpha_min) / iterations
    ## 8 possible possible positions in 1D flattened array for both players
    playerA_pos_array = game.rows*game.columns
    playerB_pos_array = game.rows*game.columns
    ## 2 possible ball possessions (belongs to playerA / playerB)
    ball_array = 2
    ## 5 possible actions for both players (NESW, stick)
    playerA_action_array = 5
    playerB_action_array = 5
    ## Q table for both players
    playerA_Q_table = np.zeros([playerB_pos_array, playerA_pos_array, ball_array, playerA_action_array, playerB_action_array])
    playerB_Q_table = np.zeros([playerB_pos_array, playerA_pos_array, ball_array, playerA_action_array, playerB_action_array])

    ## create new game environment, and set done to be 0 to begin with
    game.create_new_env()
    done = 0

    ## record difference in Q table for successful goal
    Q_diff = []
    iter_list = []

    ## loop for 1e6 times
    for i in range(iterations):
        ### check if a successful goal has finished -> reset
        if done == 1:
            game.create_new_env()
            done = 0

        ### record last time's Q value
        Q_backup = playerA_Q_table[2, 1, 1, 2, 4]
        
        ### initialize state
        playerA_pos = playerA.pos
        playerB_pos = playerB.pos
        who_owns_ball = game.ball
        curr_state = [playerA_pos, playerB_pos, who_owns_ball]
        ### current Q-table
        playerA_Q_table_state = playerA_Q_table[playerA_pos, playerB_pos, who_owns_ball]
        playerB_Q_table_state = playerB_Q_table[playerA_pos, playerB_pos, who_owns_ball]

        ### randomly chosen actions for both players
        actionA = np.random.choice(playerA_action_array)
        actionB = np.random.choice(playerA_action_array)

        ### update state, after moving both players for 1 step
        new_state, done = game.move_both_players(actionA, actionB)

        ### Solve Correlated Equilibirum, using Linear Programming (LP)
        playerA_reward_expect, playerB_reward_expect = LP_Solver_uCEQ(playerA_Q_table_state, playerB_Q_table_state)

        ### update Q-tables for both players
        playerA_Q_table[playerA_pos, playerB_pos, who_owns_ball, actionA, actionB] = \
                (1 - alpha) * playerA_Q_table[playerA_pos, playerB_pos, who_owns_ball, actionA, actionB] + \
                alpha * ((1 - gamma) * playerA.reward + gamma * playerA_reward_expect)

        playerB_Q_table[playerA_pos, playerB_pos, who_owns_ball, actionA, actionB] = \
                (1 - alpha) * playerB_Q_table[playerA_pos, playerB_pos, who_owns_ball, actionA, actionB] + \
                alpha * ((1 - gamma) * playerB.reward + gamma * playerB_reward_expect)

        ### only calculate difference in Q when reaches initial state again
        if [playerA_pos, playerB_pos, who_owns_ball, actionA, actionB] == [2, 1, 1, 2, 4]:
            Q_diff.append(abs(playerA_Q_table[2, 1, 1, 2, 4] - Q_backup))
            iter_list.append(i)
            print("Iteration=", i, ", alpha=", alpha)

        ### alpha decay
        alpha -= alpha_decay

    plot_Q_Diff(Q_diff, iter_list, name="Correlated-Q")



# main function
## initialize player A and B
playerA = player(name="A", ball = 0)
playerB = player(name="B", ball = 1)
## initialize soccer game environment
game = soccer_env(playerA, playerB)
## run Q-learning
Q_learning(game, playerA, playerB)
## run friend-Q learning
Friend_Q_learning(game, playerA, playerB)
## run foe-Q learning
Foe_Q_learning(game, playerA, playerB)
## run uCE-Q learning
uCE_Q_Learning(game, playerA, playerB)

