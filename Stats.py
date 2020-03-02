# Helps to view the battle statistics
import numpy as np

class Stats:
    def __init__(self):
        self.numGames = 0
        self.numWins = 0
        self.winRate = 0
        self.winHistory = []

    def updateStats(self, reward, game):
        self.numGames = game
        self.winHistory.append(reward)

        if reward == 1:
            self.numWins = self.numWins + 1
        
        self.winRate = np.round((self.numWins / self.numGames)*100, 2)
    

    def showWinRate(self):
        print(f'Win Rate: {self.winRate}%')


    # def plotWinRate():
    #     #TODO plot win rate
