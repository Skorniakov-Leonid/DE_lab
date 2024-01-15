import random

from solution import Solution
from solve import Solver, solvers
from graph import Graph
from .metric import Metric, metrics
from .result import Result
import typing as tp

default_start = -1.0
default_end = 1.0
eps = 0.15
default_noise = (0.00, 0.05)
default_data_steps = 16
default_condition_t = 0.0
coefficient_limit = 20.0


class Checker:
    @staticmethod
    def generate_coefficients(size: int) -> list[float]:
        return [random.uniform(-coefficient_limit, coefficient_limit) for _ in range(size)]

    def __init__(self, solution: Solution) -> None:
        self.solution = solution
        self.results: list[Result] = []

    def check(self, solvers: list[Solver] = solvers, start: float = default_start, end: float = default_end,
              noise: tuple[float, float] = default_noise, steps: int = default_data_steps) -> None:
        synthetic_data = Graph.generate(self.solution, start, end,
                                        steps=steps, noise=noise, color='red', label='Synthetic data')
        t0 = float(synthetic_data.data()[0][0])
        condition = (t0, self.solution.eval_n_dydt(0, t0))

        original_metric_result = [metric.eval(synthetic_data, self.solution) for metric in metrics]
        self.add_result(self.solution, original_metric_result, start=start, end=end, label='Function', color='green')
        for solver in solvers:
            found_ode = solver.solve(synthetic_data, condition)
            found_solution = found_ode.get_solution(condition)
            found_metric_result = [metric.eval(synthetic_data, found_solution) for metric in metrics]
            self.add_result(found_solution, found_metric_result, start=start, end=end, label=solver.label(),
                            color=solver.color(), linestyle='--')
        self.add_data(synthetic_data)
        coefficients = self.solution.get_coefficients()
        self.print_results('a = {a}, b = {b}, c = {c} | noise = {noise}'.format(a=coefficients[0], b=coefficients[1],
                                                                           c=coefficients[2], noise=noise))

    def add_result(self, solution: Solution, metric_results: list[float],
                   start: float = default_start, end: float = default_end, label: tp.Optional[str] = None,
                   color: tp.Optional[str] = None, linestyle: tp.Optional[str] = None) -> None:
        graph = Graph.generate(solution, start - eps, end + eps,
                               color=color, label=label, linestyle=linestyle)
        coefficients = solution.get_coefficients()
        self.results.append(Result(graph, metric_results, [metric.name() for metric in metrics], coefficients, False))

    def add_data(self, graph) -> None:
        self.results.append(Result(graph, points=True))

    def print_results(self, title: str) -> None:
        for result in self.results:
            result.print_info()
            result.draw()
            Metric.draw_metrics(self.results[:-1])
        Graph.show(title=title)
