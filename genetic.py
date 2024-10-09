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

def cost(ch, dist):
    total = 0.0
    for i in range(1, len(ch)):
        total += dist[ch[i-1]][ch[i]]
    return total


def crossover(parent1, parent2):
    a, b = np.random.choice(np.arange(1, 99), size=2, replace=False)
    larger = a if a>b else b
    smaller = b if b<a else a
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

        # create distance matrix to avoid repeated cost calculations
        self.dist = np.zeros((100, 100)) # distance matrix
        for i in range(100):
            for j in range(100):
                self.dist[i][j] = np.sqrt((cities[i][0] - cities[j][0])**2 + (cities[i][1] - cities[j][1])**2)

        self.populate()



    def populate(self):
        # set including all cities
        pSet = np.arange(100)

        for i in range(self.populationSize):
            ch = pSet.copy()
            np.random.shuffle(ch)
            self.population[i] = ch
        self.roulette()


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

    def generation(self):
        parents = self.roulette()
        pairs = []
        # assign parent pairs
        for i in range(0, int(len(parents)/2) + 1, 2):
            pairs.append((self.population[parents[i]], self.population[parents[i+1]]))
        # create offspring
        offspring = []
        for pair in pairs:
            off1, off2 = crossover(pair[0], pair[1])
            offspring.extend([off1, off2])
        pass

    def tournament(self):
        pass

    def scramble(self):
        pass

    def swap(self):
        pass



def main():
    with open("Random100.tsp", "r") as f:
        for _ in range(7):
            f.readline()
        cities = []
        for i in range(100):
            _, x, y = f.readline().split()
            cities.append((float(x), float(y)))
    g = GeneticAlgorithm(cities, 20, "roulette", "scramble", .05, 0.2)
    g.generation()


if __name__ == "__main__":
    main()