import math
from abc import ABC, abstractmethod


class Solution(ABC):
    """Interface for solution of ODE"""

    @abstractmethod
    def eval_n_dydt(self, n: int, t: float) -> float:
        """
        Function for evaluate n-th dy/dt by t
        :param n:           Value of n
        :param t:           Value of t
        :return:            Evaluated n-th dydt for t
        """
        pass

    @abstractmethod
    def get_coefficients(self) -> list[float]:
        """
        Get list of coefficients of this solution
        :return:            List of coefficients
        """
        pass


class SolutionLC1(Solution):
    """Solution of linear differential equation with constant coefficients of the first order"""

    def __init__(self, coefficients: list[float]) -> None:
        """
        Coefficients of solution: 'y = c * e ^ -at - b / a' in order [a, b, c]
        :param coefficients: List of coefficients of solution this type ODE
        """
        if len(coefficients) != 3:
            raise ValueError('This type of solution require 3 coefficients')
        self.coefficients = coefficients

    def eval_n_dydt(self, n: int, t: float) -> float:
        if n < 0:
            raise ValueError('n must be higher or equal to 0')
        a, b, c = self.coefficients
        match n:
            case 0:
                return c * math.exp(-a * t) - b / a
            case 1:
                return -a * c * math.exp(-a * t)
            case _:
                raise ValueError("This solution doesn't support dy/dt higher than 1")

    def get_coefficients(self) -> list[float]:
        return self.coefficients
