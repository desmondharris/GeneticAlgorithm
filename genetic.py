import threading
import time

import numpy as np
import random
import matplotlib.pyplot as plt
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

def cost(ch, dist):
    total = 0.0
    for i in range(1, len(ch)):
        total += dist[ch[i-1]][ch[i]]
    return total


def crossover(parent1, parent2):
    a, b = np.random.choice(np.arange(1, 99), size=2, replace=False)
    larger = a if a>b else b
    smaller = b if b<a else a
    #TODO: Make sure all paths are strictly 1-100, i.e check to make sure mo repeated values
    child1 = np.append(np.append(parent1[:smaller], parent2[smaller:larger]), parent1[larger:])
    child2 = np.append(np.append(parent2[:smaller], parent1[smaller:larger]), parent2[larger:])
    return child1, child2






class GeneticAlgorithm:
    def __init__(self, cities, initialPopulation, selection, mutation, pm, pc):
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

        # dummy values
        self.population = np.empty(self.populationSize, dtype=object) # all chromosomes
        self.generationTask = None
        self.spinner_index = 0

        # create distance matrix to avoid repeated cost calculations
        self.dist = np.zeros((100, 100)) # distance matrix
        for i in range(100):
            for j in range(100):
                self.dist[i][j] = np.sqrt((cities[i][0] - cities[j][0])**2 + (cities[i][1] - cities[j][1])**2)

        self.history = {
            "generations": 0,
            "mins": [],
            "maxes": [],
            "averages": [],
        }

        self.populate()

    def evaluate(self):
        costs = []
        for i in range(self.populationSize):
            costs.append(cost(self.population[i], self.dist))
        costs = np.array(costs)
        min = costs[np.argmin(costs)]
        max = costs[np.argmax(costs)]
        average = np.mean(costs)
        return min, max, average


    def generation(self):
        parents = self.selection()
        pairs = []
        # assign parent pairs
        for i in range(0, int(len(parents) / 2) + 1, 2):
            pairs.append((self.population[parents[i]], self.population[parents[i + 1]]))

        # create offspring
        offspring = []
        for pair in pairs:
            off1, off2 = crossover(pair[0], pair[1])
            offspring.extend([off1, off2])
        _, losers = self.tournament()
        for w, l in zip(offspring, losers):
            self.population[l] = w
        # mutate for each indv
        for i in range(self.populationSize):
            self.swap(i)
        self.history["generations"] += 1
        min, max, average = self.evaluate()
        print(f"Generation: {self.history['generations']} \n Min: {min} \n Max: {max} \n Avg: {average}")
        self.history["mins"].append(min)
        self.history["maxes"].append(max)
        self.history["averages"].append(average)

    def populate(self):
        # set including all cities
        pSet = np.arange(100)
        for i in range(self.populationSize):
            ch = pSet.copy()
            np.random.shuffle(ch)
            self.population[i] = ch

    def run(self, n):
        for _ in range(n):
            self.generation()


    def roulette(self):
        totalFitness = sum([1 / cost(ch, self.dist) for ch in self.population])
        # probability distribution for fitness proportionate selection
        probs = [(1 / cost(ch, self.dist))/totalFitness for ch in self.population]

        numOffSpring = int(self.pc * self.populationSize)
        if numOffSpring%2 != 0:
            raise ValueError("Crossover probability must be divisible by 2.")

        # pick appropriate number of parents according to fps
        parents = np.random.choice(len(self.population), p=probs, size=numOffSpring, replace=False)
        return parents

    def tournament(self):
        inds = range(self.populationSize)
        inds = np.array(inds)
        np.random.shuffle(inds)
        numOffSpring = int(self.pc * self.populationSize)
        if numOffSpring%2 != 0:
            raise ValueError("Crossover probability must be divisible by 2.")
        parents = []
        leastFit = []
        # generate population * crossover prob winners and losers(children and individuals(indvs) to eliminate)
        for i in range(numOffSpring):
            # get 5 random indvs to play this round
            players = []
            if numOffSpring * 5 > self.populationSize:
                raise ValueError("Crossover probability too large")
            for j in range(5):
                last, newArray = inds[-1], inds[:-1]
                players.append(last)
                inds = newArray

            # get each indvs score, best will be a parent worst will be killed off
            scores = []
            for p in players:
                scores.append(cost(self.population[p], self.dist))
            scores = np.array(scores)
            parents.append(players[np.argmin(scores)])
            leastFit.append(players[np.argmax(scores)])

        return np.array(parents), np.array(leastFit)



    def scramble(self):
        pass

    def swap(self, ch):
        if random.random() < self.pm:
            idxs = np.arange(100)
            np.random.shuffle(idxs)
            for i in range(8):
                pair = []
                for j in range(2):
                    last, newArray = idxs[-1], idxs[:-1]
                    pair.append(last)
                    idxs = newArray
                tmp = self.population[ch][pair[1]]
                self.population[ch][pair[1]] = self.population[ch][pair[0]]
                self.population[ch][pair[0]] = tmp


def main():
    with open("Random100.tsp", "r") as f:
        for _ in range(7):
            f.readline()
        cities = []
        for i in range(100):
            _, x, y = f.readline().split()
            cities.append((float(x), float(y)))
    g = GeneticAlgorithm(cities, 200, "roulette", "scramble", .05, 0.2)
    # Run the genetic algorithm for a specified number of generations
    for _ in range(50):  # Increase the number of generations if needed
        g.generation()


    # Extract data from history
    generations = list(range(1, g.history["generations"] + 1))
    mins = g.history["mins"]
    maxes = g.history["maxes"]
    averages = g.history["averages"]

    # Plot the history of min, max, and average costs
    plt.figure(figsize=(10, 6))
    plt.plot(generations, mins, label="Min Cost", marker='o')
    plt.plot(generations, maxes, label="Max Cost", marker='x')
    plt.plot(generations, averages, label="Average Cost", marker='s')

    plt.xlabel('Generations')
    plt.ylabel('Cost')
    plt.title('Genetic Algorithm Cost over Generations')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()