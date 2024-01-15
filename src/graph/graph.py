from dataclasses import dataclass

from solution import Solution
import numpy as np
import typing as tp
import matplotlib.pyplot as plt

default_start = -1.0
default_end = 1.0
default_steps = 100
default_color = 'royalblue'
default_mu = 0.0
default_sigma = 0.01
default_point_size = 10

pltg = plt.subplot2grid((3, 3), (0, 0), rowspan=2, colspan=3)


@dataclass
class Graph:
    """Class for graph"""
    points: np.ndarray
    color: str = default_color
    label: tp.Optional[str] = None
    linestyle: tp.Optional[str] = None

    @classmethod
    def generate(cls, solution: Solution, start: float = default_start, end: float = default_end,
                 steps: int = default_steps, noise: tuple[float, float] = (0.0, 0.0), color: str = default_color,
                 label: tp.Optional[str] = None, linestyle: tp.Optional[str] = None) -> 'Graph':
        """
        Generate graph of solution by parameters
        :param solution:    The solution for which the graph is generating
        :param start:       Starting value of t
        :param end:         Final value of t
        :param steps:       Count of points
        :param noise:       Noise in format (mu, sigma)
        :param color:       Color of graph [HEX | NAME]
        :param label:       Label for graph
        :param linestyle:   Linestyle for graph
        :return:            Generated graph
        """
        t_data: np.ndarray = np.arange(start, end, (end - start) / steps)
        dydt_data: list[float] = []
        for t in t_data:
            dydt_data.append(solution.eval_n_dydt(1, t))
        points = np.column_stack((t_data, dydt_data))
        graph = cls(points, color, label, linestyle)
        graph.add_noise(*noise)
        return graph

    def add_noise(self, mu: float = default_mu, sigma: float = default_sigma) -> None:
        """
        Add 'noise' to graph by parameters
        :param mu:          The average value of the distribution
        :param sigma:       Standard deviation
        :return:            Add noise by Normal Distribution [None]
        """
        noise = np.random.normal(mu, sigma, self.points.shape)
        self.points += noise
        self.points = self.points[np.argsort(self.points[:, 0])]

    def data(self) -> np.ndarray:
        """
        Transform array of points in two arrays [t_data, y_data]
        :return:            Two arrays [t_data, dydt_data]
        """
        return np.transpose(self.points)

    def draw_points(self, color: tp.Optional[None] = None) -> None:
        """
        Draw all points of graph
        :param color:       Color of points [HEX | NAME]
        :return:            Draw points [None]
        """
        color = self.color if color is None else color
        t_data, dydt_data = self.data()
        pltg.scatter(t_data, dydt_data, s=default_point_size, color=color, label=self.label)

    def draw_graph(self, color: tp.Optional[None] = None, label: tp.Optional[None] = None):
        """
        Draw graph
        :param color:       Color of graph [HEX | NAME]
        :param label:       Label of graph
        :return:            Draw graph [None]
        """
        color = self.color if color is None else color
        label = self.label if label is None else label
        t_data, dydt_data = self.data()
        pltg.plot(t_data, dydt_data, color=color, label=label, linestyle=self.linestyle)

    @staticmethod
    def show(legend: bool = True, grid: bool = True, title: str = '') -> None:
        """
        Show current plot
        :param legend:      Show legend
        :param grid:        Show grid
        :return:            Show plot [None]
        """
        if legend:
            pltg.legend()
        if grid:
            pltg.grid(True)
        pltg.set_title(title, loc='left')
        plt.show()
