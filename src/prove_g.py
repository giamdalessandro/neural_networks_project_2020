'''
import scipy.optimize as op
import tensorflow as tf


g1 = [1,2,3,4]
g2 = [1,2,3,4]


g_sum = tf.math.add(g1, g2)

#g = - op.linprog(-g_sum)

print(op.linprog(-g_sum, bounds=(-1,1)))
'''
class MultiDimensionalLpVariable:
    def __init__(self, name, dimensions, low_bound, up_bound, cat):
        self.name = name
        try:
            self.dimensions = (*dimensions,)
        except:
            self.dimensions = (dimensions,)
        self.low_bound = low_bound
        self.up_bound = up_bound
        assert cat in pulp.LpCategories, 'cat must be one of ("{}").'.format(
            '", "'.join(pulp.LpCategories)
        )
        self.cat = cat
        self.variables = self._build_variables_array()
        self.values = None

    def __getitem__(self, index):
        return self.variables[index]

    def _build_variables_array(self):
        f = numpy.vectorize(self._define_variable)
        return numpy.fromfunction(f, self.dimensions, dtype="int")

    def _define_variable(self, *index):
        name = "_".join(map(str, (self.name, *index)))
        return pulp.LpVariable(name, self.low_bound, self.up_bound, self.cat)

    def evaluate(self):
        f = numpy.vectorize(lambda i: pulp.value(i))
        self.values = f(self.variables)


from pulp import *
import tensorflow as tf

g1 = [1,2,3,4]
g2 = [1,2,3,4]

g_sum = tf.math.add(g1, g2)


prob = LpProblem("test09", LpMaximize)
g = pulp.LpVariable.dicts("g", RANGE,  cat="Real")
for i in g.viewkeys():
     g[i].lowBound = -1
     g[i].upBound = 1

prob += g_sum*g, "obj"
prob += g*g == 1, "c1"

print(value(g))