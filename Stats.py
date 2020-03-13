# Helps to view the battle statistics
import numpy as np
from collections import deque

MEMORY_SIZE = 10

class Stats:
    def __init__(self):
        self.numGames = 0
        self.numWins = 0
        self.absWinRate = 0
        self.relWinRate = 0
        self.window = []
        self.windowSum = 0
        self.winRate = 0
        self.history = deque(maxlen=MEMORY_SIZE)

    def updateStats(self, reward, game):

        # self.history.append(reward)
        
        # total = 0
        # for game in self.history:
        #     if game == 1:
        #         total += 1 
        
        # self.winRate = total / MEMORY_SIZE

        # set window length
        length = 10
        self.numGames = game

        # if we win increase total wins
        if reward == 1:
            self.numWins += 1
            self.windowSum += 1

        # update window
        if self.numGames > length:
            # hold the first reward value in the window for
            # sum of reward update
            temp = self.window.pop(0)
            self.window.append(reward)

            # update window sum if we are moving past a win
            if temp == 1:
                self.windowSum -= 1
        
        else:
            self.window.append(reward)
        
        self.absWinRate = np.round((self.numWins / self.numGames) * 100, 2)
        self.relWinRate = np.round(self.windowSum / length * 100, 0)
            

    def showAbsWinRate(self):
        print(f'Absolute Win Rate: {self.absWinRate}%')

    def showRelWinRate(self):
        print(f'Relative Win Rate: {self.relWinRate}%')

    def showNumGames(self):
        print(f'Number of Games: {self.numGames}')

    

    def showWinRate(self):
        print(f'Win Rate: {self.winRate}')

    def getWinRate(self):
        return self.winRate
    # def plotWinRate():
    #     #TODO plot win rate


