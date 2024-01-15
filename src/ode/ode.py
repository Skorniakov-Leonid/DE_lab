import math
from abc import ABC, abstractmethod
from solution import Solution, SolutionLC1


class ODE(ABC):
    """Interface for ODE"""

    @abstractmethod
    def get_solution(self, condition: tuple[float, float]) -> Solution:
        """
        Get solution by condition
        :param condition:   Point (t, y)
        :return:            Solution to this ODE satisfying the condition
        """
        pass

    @abstractmethod
    def get_coefficients(self) -> list[float]:
        """
        Get list of coefficients this ODE
        :return:            List of coefficients
        """
        pass


class LC1(ODE):

    def __init__(self, coefficients: list[float]) -> None:
        """
        Coefficients of ode: 'y` + ay + b = 0' in order [a, b]
        :param coefficients: List of coefficients ODE
        """
        if len(coefficients) != 2:
            raise ValueError('This ODE require 2 coefficients')
        self.coefficients = coefficients

    def get_solution(self, condition: tuple[float, float]) -> Solution:
        a, b = self.coefficients

        def get_c(a: float, b: float, condition: tuple[float, float]) -> float:
            t, y = condition
            return (y + b / a) * math.exp(a * t)

        c = get_c(a, b, condition)
        return SolutionLC1([a, b, c])

    def get_coefficients(self) -> list[float]:
        return self.coefficients
