# Helps to view the battle statistics
import numpy as np
from collections import deque

MEMORY_SIZE = 50

class Stats:
    def __init__(self):
        self.numGames = 0
        self.numWins = 0
        self.winRate = 0
        self.history = deque(maxlen=MEMORY_SIZE)

    def updateStats(self, reward, game):
        self.numGames = game
        self.history.append(reward)
        
        total = 0
        for game in self.history:
            if game == 1:
                total += 1 
        
        self.winRate = total / MEMORY_SIZE
    

    def showWinRate(self):
        print(f'Win Rate: {self.winRate}')


    # def plotWinRate():
    #     #TODO plot win rate
