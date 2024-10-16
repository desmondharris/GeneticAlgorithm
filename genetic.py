import queue
import numpy as np
import random
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

def cost(ch: np.array, dist: np.array):
    total = 0.0
    for i in range(1, len(ch)):
        total += dist[int(ch[i-1])][int(ch[i])]
    total += dist[int(ch[98])][99]
    return total


def verifyIndividual(ind: np.array):
    # make sure every city is visited
    if False in [n in ind for n in range(99)]:
        raise ValueError("Missing city(ies)")

    if len(ind) != 99:
        raise ValueError("Invalid chromosone length")




def eventQueueMsg(q, type, payload):
    if type not in ["guiout", "generationinfo", "eos"]:
        raise TypeError("Unknown event type")
    #q.put((type, payload+"\n"))



class GeneticAlgorithm:
    def __init__(self, cities, initialPopulation, selection, mutation, pm, pc, q=queue.Queue()):
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
        self.q = q

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
        eventQueueMsg(self.q, "guiout", f"Beginning Generation {self.history['generations']}")
        if self.selection == "roulette":
            parents = self.roulette()
        elif self.selection == "tournament":
            parents, _ = self.tournament()
        pairs = []
        # assign parent pairs
        for i in range(0, int(len(parents) / 2) + 1, 2):
            pairs.append((self.population[parents[i]], self.population[parents[i + 1]]))

        # create offspring
        offspring = []
        for pair in pairs:
            off1, off2 = self.csox(pair[0], pair[1])
            offspring.extend([off1, off2])
        _, losers = self.tournament()
        for w, l in zip(offspring, losers):
            self.population[l] = w

        eventQueueMsg(self.q, "guiout", f"Offspring generated, losing individuals killed off. Checking for mutations...")
        # mutate for each indv
        for i in range(self.populationSize):
            self.swap(i)
        self.history["generations"] += 1
        min, max, average = self.evaluate()
        # print(f"Generation: {self.history['generations']} \n Min: {min} \n Max: {max} \n Avg: {average}")
        self.history["mins"].append(min)
        self.history["maxes"].append(max)
        self.history["averages"].append(average)
        eventQueueMsg(self.q, "guiout", "Finished!")

    def populate(self):
        # set including all cities
        pSet = np.arange(99, dtype="int64")
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


    # python implementation of psuedocode in Toathom and Camprasert(2022)
    def csox(self, parent1, parent2):
        # get two distinct integers for breakpoints
        r1 =  random.randint(1, 97)
        r2 =  random.randint(1, 97)

        if r1 == r2:
            while r1==r2:
                r2 = random.randint(1, 97)
        if r1 > r2:
            temp = r1
            r1 = r2
            r2 = temp


        offspring = np.full((6, len(parent1)), -556456)
        # for i = 0 to 2 do
        for i in range(3):
            if i==0:
                pos1 = r1
                pos2 = r2
            elif i==1:
                pos1 = 0
                pos2 = r1
            elif i==2:
                pos1 = r2
                pos2 = len(parent1)

            # init empty offspring
            offspring[2*i] = np.full(len(parent1), -345345345, dtype="int64")
            offspring[2*i+1] = np.full(len(parent1), -345345, dtype="int64")

            # O[2i + 1](pos1 : pos2) ← P1(pos1 : pos2)
            # O[2i + 2](pos1 : pos2) ← P2(pos1 : pos2)
            offspring[2*i][pos1:pos2] = parent1[pos1:pos2]
            middleChunk = parent1[pos1:pos2]
            offspring[2*i+1][pos1:pos2] = parent2[pos1:pos2]

            p1 = np.full(len(parent1)-(pos2-pos1), -34545, dtype="int64")
            p2 = np.full(len(parent1)-(pos2-pos1), -234234, dtype="int64")

            pIdx = pos2
            if pIdx > len(parent1) - 1:
                pIdx = 0
            for j in range(len(p1)):
                while parent1[pIdx] in offspring[2*i+1]:
                    pIdx += 1
                    if pIdx > len(parent1) - 1:
                        pIdx = 0
                p1[j] = parent1[pIdx]
                pIdx += 1
                if pIdx > len(parent1) - 1:
                    pIdx = 0

            pIdx = pos2
            if pIdx > len(parent1) - 1:
                pIdx = 0
            for j in range(len(p2)):
                while parent2[pIdx] in offspring[2*i]:
                    pIdx += 1
                    if pIdx > len(parent2) - 1:
                        pIdx = 0
                p2[j] = parent2[pIdx]
                pIdx += 1
                if pIdx > len(parent2) - 1:
                    pIdx = 0

            offspring[i*2][pos2:] = p2[:(len(parent1) - pos2)]
            rightChunk = p2[:(len(parent1) - pos2)]
            offspring[i*2][:pos1] = p2[(len(parent1) - pos2):]
            leftChunk = p2[(len(parent1) - pos2):]

            offspring[i*2+1][pos2:] = p1[:(len(parent1) - pos2)]
            offspring[i*2+1][:pos1] = p1[(len(parent1) - pos2):]
            verifyIndividual(offspring[i*2])
            verifyIndividual(offspring[i*2+1])
        for o in offspring:
            verifyIndividual(o)
        costs = [cost(ind, self.dist) for ind in offspring]
        winnerIndices = np.argpartition(costs, len(costs) -2)[:2]
        return (offspring[winnerIndices[0]], offspring[winnerIndices[0]])






    def scramble(self):
        pass

    def swap(self, ch):
        if random.random() < self.pm:
            idxs = np.arange(99)
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
    # Run the genetic algorithm for a specified number of generations
    for _ in range(50):  # Increase the number of generations if needed
        g.generation()


    # Extract data from history
    generations = list(range(1, g.history["generations"] + 1))
    mins = g.history["mins"]
    maxes = g.history["maxes"]
    averages = g.history["averages"]

    # Plot the history of min,/,.., max, and average costs
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
    print("he")
    main()