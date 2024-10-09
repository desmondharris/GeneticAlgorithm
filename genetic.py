import numpy as np


"""
Chromosome: a 100-element numpy array containing the path
Initial Population: 150
Fitness Function: total distance traveled over path
Selection A: 2 fixed point roulette wheel
Selection B: Tournament
Mutation A:
The two longest 2 10-gene subsets are located heuristically and scrambled
Mutation B:
8 pairs of genes are randomly chosen and swapped
Crossover: 3 point crossover
Survivor Selection: Tournament
Pm: 0.05
Pc: 0.36
"""

class GeneticAlgorithm:
    def __init__(self, initialPopulation, selection, mutation, pm, pc):
        # set parameters
        self.populationSize = initialPopulation

        if selection not in ["roulette", "tournament"]:
            raise TypeError("Selection must be either roulette or tournament.")
        self.selection = self.roulette if selection == "roulette" else self.tournament

        if mutation not in ["scramble", "swap"]:
            raise TypeError("Mutation must be either scramble or swap.")
        self.mutation = self.scramble if mutation == "scramble" else self.swap

        # mutation and crossover probabilities
        self.pm = pm
        self.pc = pc

        # this is used to
        self.pSet = np.arange(100)



    def roulette(self):
        pass

    def tournament(self):
        pass

    def scramble(self):
        pass

    def swap(self):
        pass



GeneticAlgorithm(5, "roulette", "scramble", .05, 0.2)