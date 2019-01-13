# -*- coding: utf-8 -*-
# TODO(alexis): add **tests** (and try unittest.mock.MagicMock at the same
# time).

import collections
import random


__all__ = ['DE']
Individual = collections.namedtuple('Individual', 'ind fit')


class DE(object):
    """This class implements differential evolution.

    This method is described by Storne and Price in a paper titled
    "Differential evolution-a simple and efficient heuristic for global
    optimization over continuous spaces".

    Variables names in the implementation (x, y, z, u, v, F, CR, NP) try to
    match their equivalents in the paper.
    """

    def __init__(self, x='rand', y=1, z='bin', *, F=.5, CR=.1):
        self.x = x #Specifies the vector to be mutated 
        self.y = y #Specifies the the number of vectors to be used in finding the difference
        self.z = z #Specifies the crossover scheme bin ==> binomial and exp ==>exponential
        self.F = F #Denotes the scaling factor
        self.CR = CR #Denoted the crossover ratio

    # TODO(alexis): add a fitness_aim param?
    # TODO(alexis): add a generic way to generate initial pop?
    def solve(self, fitness, initial_population, iterations=1000):
        current_generation = [Individual(ind, fitness(*ind)) for ind in
                              initial_population]

        for _ in range(iterations):
            trial_generation = []

            for ind in current_generation:
                v = self._mutate(current_generation)
                u = self._crossover(ind.ind, v)

                trial_generation.append(Individual(u, fitness(*u)))

            current_generation = self._selection(current_generation,
                                                 trial_generation)

        best_index = self._get_best_index(current_generation)
        return current_generation[best_index].ind

    # def init_RewardTable:

    # def init_QTable:

    # def Roulette_choice_selection():


    def _mutate(self, population):
        if self.x == 'rand':
            r1, *r = self._get_indices(self.y * 2 + 1, len(population))

        elif self.x == 'best':
            r1 = self._get_best_index(population)
            r = self._get_indices(self.y * 2, len(population), but=xr1)

        mutated = population[r1].ind[:]  # copy base vector
        dimension = len(mutated)
        difference = [0] * dimension

        for plus in r[:self.y]:
            for i in range(dimension):
                difference[i] += population[plus].ind[i]

        for minus in r[self.y:]:
            for i in range(dimension):
                difference[i] -= population[minus].ind[i]

        for i in range(dimension):
            mutated[i] += self.F * difference[i]

        return mutated

    def _crossover(self, x, v):
        # assume self.z == 'bin'

        u = x[:]
        i = random.randrange(len(x))  # NP

        for j, (a, b) in enumerate(zip(x, v)):
            if i == j or random.random() <= self.CR:
                u[j] = v[j]

        return u

    def _selection(self, current_generation, trial_generation):
        generation = []

        for a, b in zip(current_generation, trial_generation):
            if a.fit < b.fit:
                generation.append(a)
            else:
                generation.append(b)

        return generation

    def _get_indices(self, n, upto, but=None):
        candidates = list(range(upto))

        if but is not None:
            # yeah O(n) but random.sample cannot use a set
            candidates.remove(but)

        return random.sample(candidates, n)

    def _get_best_index(self, population):
        min_fitness = population[0].fit
        best = 0

        for i, x in enumerate(population):
            if x.fit < min_fitness:
                best = i

        return best

    def _set_x(self, x):
        if x not in ['rand', 'best']:
            raise ValueError("x should be either 'rand' or 'best'.")

        self._x = x

    def _set_y(self, y):
        if y < 1:
            raise ValueError('y should be > 0.')

        self._y = y

    def _set_z(self, z):
        if z != 'bin':
            raise ValueError("z should be 'bin'.")

        self._z = z

    def _set_F(self, F):
        if not 0 <= F <= 2:
            raise ValueError('F should belong to [0, 2].')

        self._F = F

    def _set_CR(self, CR):
        if not 0 <= CR <= 1:
            raise ValueError('CR should belong to [0, 1].')

        self._CR = CR

    x = property(lambda self: self._x, _set_x, doc='How to choose the vector '
                 'to be mutated.')
    y = property(lambda self: self._y, _set_y, doc='The number of difference '
                 'vectors used.')
    z = property(lambda self: self._z, _set_z, doc='Crossover scheme.')
    F = property(lambda self: self._F, _set_F, doc='Weight used during '
                                                   'mutation.')
    CR = property(lambda self: self._CR, _set_CR, doc='Weight used during '
                                                      'bin crossover.')


if __name__ == '__main__':
    import math

    # http://tracer.lcc.uma.es/problems/ackley/ackley.html
    # Conversion to 10 dimensional optimization problem
    # Source : https://arxiv.org/pdf/1308.4008.pdf
    def ackley_10d(a,b,c,d,e,f,g,h,i,j):
        return (20 + math.e
                - 20 * math.exp(-.2 * (.1 * (a ** 2 + b ** 2 + c ** 2 + d ** 2 + e ** 2 + f ** 2 + g ** 2 + h ** 2 + i ** 2 + j ** 2)) ** .5)
                - math.exp(.1 *
                           (math.cos(2 * math.pi * a)
                            + math.cos(2 * math.pi * b) + math.cos(2 * math.pi * c)
                            + math.cos(2 * math.pi * d) + math.cos(2 * math.pi * e)
                            + math.cos(2 * math.pi * f) + math.cos(2 * math.pi * g)
                            + math.cos(2 * math.pi * h) + math.cos(2 * math.pi * i)
                            + math.cos(2 * math.pi * j))))

    de = DE()
    bound = 35
    # Following represents the initial population
    pop = [[random.uniform(-bound, bound), random.uniform(-bound, bound),random.uniform(-bound, bound), random.uniform(-bound, bound),
                            random.uniform(-bound, bound), random.uniform(-bound, bound),random.uniform(-bound, bound),
                            random.uniform(-bound, bound), random.uniform(-bound, bound), random.uniform(-bound, bound)]
           for _ in range(20 * 10)]  # 20 * dimension of the problem

    #print(pop)

    for i in [x ** 2 for x in range(15)]:
        random.seed(0)
        v = de.solve(ackley_10d, pop, iterations=i)
        print(v, '->', ackley_10d(*v))



















