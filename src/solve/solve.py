import math
import typing as tp

from abc import abstractmethod, ABC
from scipy.integrate import odeint

from graph import Graph
from ode import ODE, LC1
import numpy as np
from scipy.optimize import curve_fit


class Solver(ABC):
    """Interface for solvers ODE"""

    @abstractmethod
    def solve(self, graph: Graph, condition: tuple[float, float]) -> ODE:
        """
        Find approximation of graph
        :param graph:       Graph with points of approximating ODE
        :param condition    Point (t, y)
        :return:            Coefficients of approximation of the required ODE
        """
        pass

    @abstractmethod
    def label(self) -> str:
        """
        Get name of solver
        :return:            Name of solver
        """

    @abstractmethod
    def color(self) -> str:
        """
        Get color of solver
        :return:            Name of color
        """


class LSSolver(Solver):
    """Implementation of the Solver using the least squares method"""

    def solve(self, graph: Graph, condition: tuple[float, float]) -> ODE:
        t_data, dydt_data = graph.data()

        def function(t: np.ndarray, a: float, b: float) -> np.ndarray:
            def de(y: np.ndarray, t: np.ndarray, a: float, b: float) -> np.ndarray:
                dydt = -a * y - b
                return dydt

            y = odeint(de, condition[1], t, (a, b)).flatten()
            return de(y, t, a, b)

        a, b = curve_fit(function, t_data, dydt_data, maxfev=1000)[0]

        return LC1([a, b])

    def label(self) -> str:
        return 'Least Square'

    def color(self) -> str:
        return 'blue'


class LSSolverExp(Solver):
    """Implementation of the Solver using the least squares method for exponent version"""

    def solve(self, graph: Graph, condition: tuple[float, float]) -> ODE:
        t_data, dydt_data = graph.data()

        def function(t: np.ndarray, a: float, c: float) -> np.ndarray:
            return -a * c * np.exp(-a * t)

        a, c = curve_fit(function, t_data, dydt_data, maxfev=5000)[0]

        def get_b(a: float, c: float, condition: tuple[float, float]) -> float:
            t, y = condition
            return a * c * math.exp(-a * t) - a * y

        b = get_b(a, c, condition)

        return LC1([a, b])

    def label(self) -> str:
        return 'Least Square Exp'

    def color(self) -> str:
        return 'gray'


class Calculator:
    """Calculation coefficients by all pair of points"""

    def calculate(self, graph: Graph, condition: tuple[float, float]) -> tuple[list[float], list[float]]:
        t_data, dydt_data = graph.data()

        a_lst: list[float] = []
        b_lst: list[float] = []

        def get_b(a: float, c: float, condition: tuple[float, float]) -> float:
            t, y = condition
            try:
                return a * c * math.exp(-a * t) - a * y
            except OverflowError:
                return 1000000000000

        for i in range(len(t_data)):
            x0, dydt0 = t_data[i], dydt_data[i]
            for j in range(i + 1, len(t_data)):
                x1, dydt1 = t_data[j], dydt_data[j]

                try:
                    a = -math.log(dydt0 / dydt1) / (x0 - x1)
                    c = - dydt0 / (a * math.exp(-a * x0))
                except Exception:
                    continue

                b = get_b(a, c, condition)

                a_lst.append(a)
                b_lst.append(b)

        return (a_lst, b_lst)


class AveragePowerSolver(Solver, Calculator):
    """Implementation of solver using average power of calculated coefficients"""

    def __init__(self, parts: int = 10, pow: int = 3, label: tp.Optional[str] = None, color: tp.Optional[str] = None) -> None:
        self.parts = parts
        self.pow = pow
        self.label_ = label
        self.color_ = color

    def solve(self, graph: Graph, condition: tuple[float, float]) -> ODE:
        a_lst, b_lst = self.calculate(graph, condition)
        cnt = len(a_lst)
        half_median = max(cnt // (2 * self.parts), 1)
        a_lst = sorted(a_lst)[cnt // 2 - half_median: cnt // 2 + half_median]
        b_lst = sorted(b_lst)[cnt // 2 - half_median: cnt // 2 + half_median]

        def pow(v: float, p: float) -> float:
            if v < 0:
                return -math.pow(-v, p)
            return math.pow(v, p)

        a = pow(np.mean(np.power(a_lst, self.pow)), 1 / self.pow)
        b = pow(np.mean(np.power(b_lst, self.pow)), 1 / self.pow)
        return LC1([a, b])

    def label(self) -> str:
        if self.label_ is None:
            return 'Average Power 1/{parts} {power}'.format(parts=self.parts, power=self.pow)
        else:
            return self.label_

    def color(self) -> str:
        if self.color_ is None:
            return 'grey'
        else:
            return self.color_

median = AveragePowerSolver(10000, 1, 'Median', 'brown')
mean_power = AveragePowerSolver(2, 3, 'Mean Power', 'orange')
solvers = [median, LSSolver()]
