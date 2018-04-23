import numpy as np
import matplotlib.pyplot as plt
import time
from numpy import array, eye, hstack, ones, vstack, zeros
from enum import Enum

class Action(Enum):
    North = 0
    East = 1
    South = 2
    West = 3
    Stick = 4

class OwnBall(Enum):
    A = 0
    B = 1


class Player():
    #if ball = 0 means do not have has has ball
    #if ball = 1 means have ball
    #
    def __init__(self, name="PlayerX", ball = None):
        self.name = name
        self.xPos = 0
        self.yPos = 0
        self.ball = 0

    def clone(self):
        playerClone = Player(self.name+"Clone")
        playerClone.setPos(self.xPos, self.yPos)
        playerClone.ball = self.ball
        return playerClone

    def isCollion(self, player):
        return self.xPos == player.xPos and self.yPos == player.yPos

    def hasBall(self):
        return self.ball == 1
    def loseBall(self):
        self.ball = 0
    def getBall(self):
        self.ball = 1
    def setPos(self, x, y):
        self.xPos = x
        self.yPos = y
    def toNorth(self):
        self.yPos -= 1
    def toEast(self):
        self.xPos += 1
    def toSouth(self):
        self.yPos += 1
    def toWest(self):
        self.xPos -= 1



class SoccerEnv():

    # we will use 1d array two present the pos, in this way we can save the dimension of Q table
    def __init__(self, row = 2, col = 4):
        self.playerA = Player("PlayerA")
        self.playerB = Player("PlayerB")
        self.row = row
        self.col = col
        self.goalFieldA = []
        self.goalFieldB = []
        # this pos is used to test whether is goal
        self.ballPos = 0
        # As a state for Q table
        self.whoOwnBall = OwnBall.B
    # mapping between the 2d to 1d
    def envPos(self,x,y):
        return x+y*self.col

    def createEnv(self):
        # the initialization state is same every time
        # the position of A is (3,0) B is(2, 0)
        #map the two dimension space to an array
        self.goalFieldA = [0+i*self.col for i in range(0, self.row)]
        self.goalFieldB = [self.col*(i+1)-1 for i in range(0, self.row)]
        self.playerA.setPos(2,0)
        self.playerB.setPos(1,0)
        self.playerB.ball = 1
        self.playerA.ball = 0
        #player B has the ball in the default setting
        self.ballPos = self.envPos(self.playerB.xPos, self.playerB.yPos)
        self.whoOwnBall = OwnBall.B

    def moveSinglePlayer(self, player, action):
        if action == Action.North and player.yPos > 0:
            player.toNorth()
        elif action == Action.East and player.xPos < self.col-1:
            player.toEast()
        elif action == Action.South and player.yPos < self.row-1:
            player.toSouth()
        elif action == Action.West and player.xPos > 0:
            player.toWest()
        return

    def movePlayers(self, player1, player2, action1, action2):

        player1Clone = player1.clone()
        player2Clone = player2.clone()

        self.moveSinglePlayer(player1Clone, action1)
        self.moveSinglePlayer(player2Clone, action2)
        #always player1 first


        if player1Clone.isCollion(player2):

            if(player1Clone.hasBall()):
                player2.getBall()
                player1.loseBall()

        else:
            # can move the empty place
            player1.setPos(player1Clone.xPos, player1Clone.yPos)

        # then move the second player
        if player2Clone.isCollion(player1):

            if (player2Clone.hasBall()):
                player2.loseBall()
                player1.getBall()
        else:
            player2.setPos(player2Clone.xPos, player2Clone.yPos)
            #print(player2.xPos, player2.yPos)

    def calcReward(self):
        if self.ballPos in self.goalFieldA:
            return (100, -100, True)
        elif self.ballPos in self.goalFieldB:
            return (-100, 100, True)
        return (0, 0, False)
    # this state is used to record which player own the ball
    def updateBallState(self):
        if (self.playerA.hasBall()):
            self.whoOwnBall = OwnBall.A
            self.ballPos = self.envPos(self.playerA.xPos, self.playerA.yPos)
            #print("ballPos:", self.ballPos)
        elif(self.playerB.hasBall()):
            self.whoOwnBall = OwnBall.B
            self.ballPos = self.envPos(self.playerB.xPos, self.playerB.yPos)
            #print("ballPos:", self.ballPos)
        else:
            print("something may be wrong!!!!")

    def nextStep(self, actionA, actionB):
        #print("before: ", self.envPos(self.playerA.xPos, self.playerA.yPos))
        if np.random.randint(2) == 0:
            #player A first
            self.movePlayers(self.playerA, self.playerB, actionA, actionB)
        else:
            self.movePlayers(self.playerB, self.playerA, actionB, actionA)
        #print("after: ", self.envPos(self.playerA.xPos, self.playerA.yPos))
        #print("playerA ball:",self.playerA.hasBall())
        #print("playerB ball:",self.playerB.hasBall())
        self.updateBallState()
        rewardA, rewardB, done = self.calcReward()

        return self.envPos(self.playerA.xPos, self.playerA.yPos), self.envPos(self.playerB.xPos, self.playerB.yPos),\
                self.whoOwnBall, rewardA, rewardB, done



class MarkovGame:
    def __init__(self, env):
        self.env = env
        self.iter = 10**6
        self.eps = 1
        self.epsMin = 0.01
        self.epsDecay = (self.eps - self.epsMin) / self.iter
        self.alpha = 0.5
        self.alphaMin = 0.001
        self.alphaDecay = (self.alpha - self.alphaMin) / self.iter
        self.gamma = 0.9
        self.recordAState = [2, 1, OwnBall.B, Action.South]
        self.recordState = [2, 1, OwnBall.B, Action.South, Action.Stick]
        self.Qrecord = []
        self.indexRecord=[]
    def initialQTable(self):
        numPos = self.env.row * self.env.col
        numBallState = 2# A or B
        numAction = len(Action)
        self.Qa = np.zeros([numPos, numPos, numBallState, numAction])
        self.Qb = np.zeros([numPos, numPos, numBallState, numAction])
        self.Qff = np.zeros([numPos, numPos, numBallState, numAction, numAction])

    def plotQDiff(self):
        res = np.array(self.Qrecord)
        diff = np.abs(res[1:] - res[0:-1])
        plt.plot(self.indexRecord[0:-1], diff)
        plt.xlabel("Iter")
        plt.ylabel("Q-Diff")
        plt.show()

    def simulateQ(self):
        self.env.createEnv()
        self.initialQTable()
        done = False
        self.Qrecord = []
        self.indexRecord=[]
        for i in range(self.iter):
            if done:
                self.env.createEnv()
            aState = self.env.envPos(self.env.playerA.xPos, self.env.playerA.yPos)
            bState = self.env.envPos(self.env.playerB.xPos, self.env.playerB.yPos)
            ballState = self.env.whoOwnBall

            if self.eps > np.random.random():
                actionA = np.random.choice(Action)
                actionB = np.random.choice(Action)
            else:
                actionA = Action(np.argmax(self.Qa[aState, bState, ballState.value]))
                actionB = np.random.choice(Action)

            aStateNew, bStateNew,ballStateNew, rewardA, rewardB, done = self.env.nextStep(actionA, actionB)
            self.Qa[aState, bState, ballState.value, actionA.value] = (1 - self.alpha) * self.Qa[aState, bState, ballState.value, actionA.value]\
                                                     + self.alpha *((1-self.gamma)*rewardA + self.gamma*np.max(self.Qa[aStateNew, bStateNew,ballStateNew.value]))
            self.alpha -= self.alphaDecay
            self.eps -= self.epsDecay
            #print(aState, bState, ballState, actionA)
            if [aState, bState, ballState, actionA] == self.recordAState :
                #print("count:", i)
                #print(aState, bState, ballState, actionA)
                self.Qrecord.append(self.Qa[aState, bState, ballState.value, actionA.value])
                self.indexRecord.append(i)
        self.plotQDiff()
        return

    def simulateFriendQ(self):
        np.random.seed(1024)
        self.env.createEnv()
        self.initialQTable()
        done = False
        self.Qrecord = []
        self.indexRecord=[]
        for i in range(self.iter):
            if done:
                self.env.createEnv()
            aState = self.env.envPos(self.env.playerA.xPos, self.env.playerA.yPos)
            bState = self.env.envPos(self.env.playerB.xPos, self.env.playerB.yPos)
            ballState = self.env.whoOwnBall
            actionA = Action(np.random.randint(5))

            actionB = Action(np.random.randint(5))

            #print(aState, bState, ballState.value, actionA.value,actionB.value)
            aStateNew, bStateNew,ballStateNew, rewardA, rewardB, done = self.env.nextStep(actionA, actionB)


            #print (aStateNew, bStateNew,ballStateNew, rewardA, rewardB, done)
            self.Qff[aState, bState, ballState.value, actionA.value, actionB.value] = (1 - self.alpha) * self.Qff[aState, bState, ballState.value, actionA.value, actionB.value]\
                                                     + self.alpha *((1-self.gamma)*rewardA + self.gamma*np.max(self.Qff[aStateNew, bStateNew,ballStateNew.value]))
            self.alpha -= self.alphaDecay

            if [aState, bState, ballState, actionA, actionB] == self.recordState :
                #print(aState, bState, ballState.value, actionA.value, actionB.value)
                self.Qrecord.append(self.Qff[aState, bState, ballState.value, actionA.value, actionB.value])
                self.indexRecord.append(i)
        self.plotQDiff()
        return

    

def main():
    env = SoccerEnv()
    game = MarkovGame(env)
    game.simulateQ()
