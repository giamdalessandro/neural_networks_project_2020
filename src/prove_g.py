'''
import scipy.optimize as op
import tensorflow as tf


g1 = [1,2,3,4]
g2 = [1,2,3,4]


g_sum = tf.math.add(g1, g2)

#g = - op.linprog(-g_sum)

print(op.linprog(-g_sum, bounds=(-1,1)))


from pulp import *
import tensorflow as tf
import numpy

class MultiDimensionalLpVariable:
    def __init__(self, name, dimensions, low_bound, up_bound):
        self.name = name
        try:
            self.dimensions = (*dimensions,)
        except:
            self.dimensions = (dimensions,)
        self.low_bound = low_bound
        self.up_bound = up_bound
        self.variables = self._build_variables_array()
        self.values = None

    def __getitem__(self, index):
        return self.variables[index]

    def _build_variables_array(self):
        f = numpy.vectorize(self._define_variable)
        return numpy.fromfunction(f, self.dimensions, dtype="int")

    def _define_variable(self, *index):
        name = "_".join(map(str, (self.name, *index)))
        return pulp.LpVariable(name, self.low_bound, self.up_bound)

    def evaluate(self):
        f = numpy.vectorize(lambda i: pulp.value(i))
        self.values = f(self.variables)



g1 = [0.21, 0.17, -0.7, -0.2]
g2 = [-0.21, 0.17, -0.7, -0.2]


prob = LpProblem("test09", LpMaximize)

g_sum = tf.math.add(g1, g2)

g = LpVariable.dicts("g", list(range(4)), -1, 1)
for i in range(4):
    g[i].lowBound = -1
    g[i].upBound = 1

prob += lpSum(g_sum[i]*g[i] for i in range(4)), "obj"
prob += lpSum(g[i]*g[i] for i in range(4)) == 1, "c1"

print(g)

'''

import tensorflow as tf
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import Bounds



g1 = [0.21, 0.17, -0.7, -0.2]
g2 = [-0.21, 0.17, -0.7, -0.2]



def objective(x):
    g_sum = tf.math.add(g1, g2)
    #for i in range(4):
        #obj += -g_sum[i]*x[i]
    return -g_sum[0]*x[0] - g_sum[1]*x[1]

def constraint1(x):
    sum_eq = 1.0
    for i in range(4):
        sum_eq = sum_eq - x[i]**2
    return sum_eq

n = 4
x0 = np.zeros(n)

b = (-1.0,1.0)
bnds = ([b, b, b, b])
#bounds = Bounds(-1, 1)
con1 = {'type': 'eq', 'fun': constraint1}
cons = ([con1])
solution = minimize(objective, x0, bounds=bnds,constraints=cons)
x = solution.x

print(x)
