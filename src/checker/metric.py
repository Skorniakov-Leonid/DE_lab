from abc import abstractmethod, ABC

from matplotlib import pyplot as plt
from sklearn.metrics import r2_score

from .result import Result
from solution import Solution
from graph import Graph
import numpy as np


class Metric(ABC):
    """Interface for metric"""

    @abstractmethod
    def eval(self, graph: Graph, solution: Solution) -> float:
        """
        Evaluating metric for graph and his solution
        :param graph:       Graph with data
        :param solution:    Solution for graph
        :return:            Evaluated metric
        """
        pass

    @abstractmethod
    def name(self) -> str:
        """
        Name of metric
        :return:            Name of metric
        """
        pass

    @staticmethod
    def draw_metrics(results: list[Result]) -> None:
        """
        Draw metric
        :param metric_results: Metric results
        :param label:       Label of metric
        :param color:       Color of metric
        :return:
        """
        cnt = len(results)
        for i, metric in enumerate(metrics):
            pltm[i].set_title(metric.name(), y=-0.65)
            for j, result in enumerate(results):
                pltm[i].bar(j / cnt, result.metric_results[i], width=1/cnt, color=result.graph.color)


class MseMetric(Metric):
    """Mean Squared Error"""

    def eval(self, graph: Graph, solution: Solution) -> float:
        t_data, dydt_data = graph.data()
        dydt_solution = np.array([solution.eval_n_dydt(1, t) for t in t_data])
        return np.square(dydt_data - dydt_solution).mean()

    def name(self) -> str:
        return "MSE"


class MaeMetric(Metric):
    """Mean Absolute Error"""

    def eval(self, graph: Graph, solution: Solution) -> float:
        t_data, dydt_data = graph.data()
        dydt_solution = np.array([solution.eval_n_dydt(1, t) for t in t_data])
        return np.abs(dydt_data - dydt_solution).mean()

    def name(self) -> str:
        return "MAE"


class HuberCriterionMetric(Metric):
    """Huber Criterion"""

    def eval(self, graph: Graph, solution: Solution) -> float:
        t_data, dydt_data = graph.data()
        dydt_solution = np.array([solution.eval_n_dydt(1, t) for t in t_data])
        deviation = np.abs(dydt_data - dydt_solution)
        delta = 1.0
        return np.where(deviation <= delta, 0.5 * deviation ** 2, delta * (deviation - 0.5 * delta)).mean()

    def name(self) -> str:
        return "HC"

class RSquareMetric(Metric):
    """R Square"""
    def eval(self, graph: Graph, solution: Solution) -> float:
        t_data, dydt_data = graph.data()
        dydt_solution = np.array([solution.eval_n_dydt(1, t) for t in t_data])
        return r2_score(dydt_solution, dydt_data)

    def name(self) -> str:
        return 'RS'

metrics: list[Metric] = [MseMetric(), MaeMetric(), HuberCriterionMetric(), RSquareMetric()]
mcnt = len(metrics)
scale = mcnt + 1
pltm = [plt.subplot2grid((9, mcnt * scale - 1), (7, scale * i), rowspan=2, colspan=scale - 1)
        for i in range(mcnt)]
