# Helps to view the battle statistics
import numpy as np

class Stats:
    def __init__(self):
        self.numGames = 0
        self.numWins = 0
        self.absWinRate = 0
        self.relWinRate = 0
        self.window = []
        self.windowSum = 0

    def updateStats(self, reward, game):
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
        
        self.absWinRate = np.round((self.numWins / self.numGames)*100, 2)
        self.relWinRate = np.round(self.windowSum / length * 100, 0)
            

    def showAbsWinRate(self):
        print(f'Absolute Win Rate: {self.absWinRate}%')

    def showRelWinRate(self):
        print(f'Relative Win Rate: {self.relWinRate}%')

    def showNumGames(self):
        print(f'Number of Games: {self.numGames}')


    # def plotWinRate():
    #     #TODO plot win rate


