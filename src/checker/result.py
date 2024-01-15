from dataclasses import dataclass
from graph import Graph
import typing as tp


@dataclass
class Result:
    graph: Graph
    metric_results: tp.Optional[list[float]] = None
    metrics: list[str] = None
    coefficients: tp.Optional[list[float]] = None
    points: bool = False

    def print_info(self):
        print('-' * 50)
        print('\tLabel: {label}'.format(label=self.graph.label))
        print('\tColor: {color}'.format(color=self.graph.color))
        if not self.points:
            print('Coefficients: {coefficients}'.format(coefficients=self.coefficients))
            for ind, metric in enumerate(self.metrics):
                print('{metric_name}: {metric_result}'.format(metric_name=metric,
                                                              metric_result=self.metric_results[ind]))
        print('=' * 50)

    def draw(self):
        if self.points:
            self.graph.draw_points()
        else:
            self.graph.draw_graph()
