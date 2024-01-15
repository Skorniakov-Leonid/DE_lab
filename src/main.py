import math

from graph import *
from solution import *
from ode import *
from solve import *
from checker import *

if __name__ == '__main__':
    a, b, c = 3.0, 1.0, -2.0
    Checker(SolutionLC1([a, b, c])).check()
