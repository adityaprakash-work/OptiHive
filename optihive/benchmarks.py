# ---INFO-----------------------------------------------------------------------
# Author: Aditya Prakash
# Last edited: 2023-07-17

# --Needed functionalities

# ---DEPENDENCIES---------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ---BENCHMARKS-----------------------------------------------------------------
class BenchmarkObjective(object):
    bounds_2d = (None, None) * 2
    global_minima_2d = ((None, None, None),)

    def __call__(self, kwargs):
        raise NotImplementedError

    def plot2d(self, bounds_2d=None, resolution=100):
        if bounds_2d is None:
            bounds_2d = self.bounds_2d
        x = np.linspace(bounds_2d[0], bounds_2d[1], resolution)
        y = np.linspace(bounds_2d[2], bounds_2d[3], resolution)
        X, Y = np.meshgrid(x, y)
        Z = np.empty_like(X)
        for i in range(resolution):
            for j in range(resolution):
                Z[i, j] = self({"x": X[i, j], "y": Y[i, j]})
        sns.set_style("darkgrid")
        sns.set_context("paper")
        plt.figure(figsize=(6, 5))
        cf = plt.contourf(X, Y, Z, cmap="RdYlBu", levels=50)
        for gm in self.global_minima_2d:
            plt.scatter(
                gm[0],
                gm[1],
                marker="*",
                s=100,
                zorder=10,
                color="silver",
            )
        plt.colorbar(cf)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(self.__class__.__name__)
        plt.show()


class RastriginObjective(BenchmarkObjective):
    bounds_2d = (-5.12, 5.12) * 2
    global_minima_2d = ((0, 0, 0),)

    def __call__(self, kwargs):
        x = np.array(list(kwargs.values()))
        v = 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))
        return v


class AckleyObjective(BenchmarkObjective):
    bounds_2d = (-5, 5) * 2
    global_minima_2d = ((0, 0, 0),)

    def __call__(self, kwargs):
        x = np.array(list(kwargs.values()))
        v = (
            -20 * np.exp(-0.2 * np.sqrt(np.sum(x**2) / len(x)))
            - np.exp(np.sum(np.cos(2 * np.pi * x)) / len(x))
            + 20
            + np.exp(1)
        )
        return v


class SphereObjective(BenchmarkObjective):
    bounds_2d = (-10000, 10000) * 2  # actual bounds_2d are (-inf, inf)
    global_minima_2d = ((0, 0, 0),)

    def __call__(self, kwargs):
        x = np.array(list(kwargs.values()))
        v = np.sum(x**2)
        return v


class RosenbrockObjective(BenchmarkObjective):
    bounds_2d = (-10000, 10000) * 2  # actual bounds_2d are (-inf, inf)
    global_minima_2d = ((1, 1, 0),)

    def __call__(self, kwargs):
        x = np.array(list(kwargs.values()))
        v = np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (x[:-1] - 1) ** 2)
        return v


class BealeObjective(BenchmarkObjective):
    bounds_2d = (-4.5, 4.5) * 2
    global_minima_2d = ((3, 0.5, 0),)

    def __call__(self, kwargs):
        x = np.array(list(kwargs.values()))
        if len(x) > 2:
            raise ValueError("Beale objective function is only defined for 2D.")
        v = (
            (1.5 - x[0] + x[0] * x[1]) ** 2
            + (2.25 - x[0] + x[0] * x[1] ** 2) ** 2
            + (2.625 - x[0] + x[0] * x[1] ** 3) ** 2
        )
        return v


class GoldsteinPriceObjective(BenchmarkObjective):
    bounds_2d = (-2, 2) * 2
    global_minima_2d = ((0, -1, 3),)

    def __call__(self, kwargs):
        x = np.array(list(kwargs.values()))
        if len(x) > 2:
            raise ValueError(
                "Goldstein-Price objective function is only defined for 2D."
            )
        v = (
            1
            + (x[0] + x[1] + 1) ** 2
            * (
                19
                - 14 * x[0]
                + 3 * x[0] ** 2
                - 14 * x[1]
                + 6 * x[0] * x[1]
                + 3 * x[1] ** 2
            )
        ) * (
            30
            + (2 * x[0] - 3 * x[1]) ** 2
            * (
                18
                - 32 * x[0]
                + 12 * x[0] ** 2
                + 48 * x[1]
                - 36 * x[0] * x[1]
                + 27 * x[1] ** 2
            )
        )
        return v


class BoothObjective(BenchmarkObjective):
    bounds_2d = (-10, 10) * 2
    global_minima_2d = ((1, 3, 0),)

    def __call__(self, kwargs):
        x = np.array(list(kwargs.values()))
        if len(x) > 2:
            raise ValueError("Booth objective function is only defined for 2D.")
        v = (x[0] + 2 * x[1] - 7) ** 2 + (2 * x[0] + x[1] - 5) ** 2
        return v


class BukinN6Objective(BenchmarkObjective):
    bounds_2d = (-15, -5, -3, 3)
    global_minima_2d = ((-10, 1, 0),)

    def __call__(self, kwargs):
        x = np.array(list(kwargs.values()))
        if len(x) > 2:
            raise ValueError(
                "Bukin function N.6 objective function is only defined for 2D."
            )
        v = 100 * np.sqrt(np.abs(x[1] - 0.01 * x[0] ** 2)) + 0.01 * np.abs(x[0] + 10)
        return v


class MatyasObjective(BenchmarkObjective):
    bounds_2d = (-10, 10) * 2
    global_minima_2d = ((0, 0, 0),)

    def __call__(self, kwargs):
        x = np.array(list(kwargs.values()))
        if len(x) > 2:
            raise ValueError("Matyas objective function is only defined for 2D.")
        v = 0.26 * (x[0] ** 2 + x[1] ** 2) - 0.48 * x[0] * x[1]
        return v


class LeviN13Objective(BenchmarkObjective):
    bounds_2d = (-10, 10) * 2
    global_minima_2d = ((1, 1, 0),)

    def __call__(self, kwargs):
        x = np.array(list(kwargs.values()))
        if len(x) > 2:
            raise ValueError("Levi N.13 objective function is only defined for 2D.")
        v = (
            np.sin(3 * np.pi * x[0]) ** 2
            + (x[0] - 1) ** 2 * (1 + np.sin(3 * np.pi * x[1]) ** 2)
            + (x[1] - 1) ** 2 * (1 + np.sin(2 * np.pi * x[1]) ** 2)
        )
        return v


class HimmelblauObjective(BenchmarkObjective):
    bounds_2d = (-5, 5) * 2
    global_minima_2d = (
        (3, 2, 0),
        (-2.805118, 3.131312, 0),
        (-3.779310, -3.283186, 0),
        (3.584428, -1.848126, 0),
    )

    def __call__(self, kwargs):
        x = np.array(list(kwargs.values()))
        if len(x) > 2:
            raise ValueError("Himmelblau objective function is only defined for 2D.")
        v = (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2
        return v


class ThreeHumpCamelObjective(BenchmarkObjective):
    bounds_2d = (-5, 5) * 2
    global_minima_2d = ((0, 0, 0),)

    def __call__(self, kwargs):
        x = np.array(list(kwargs.values()))
        if len(x) > 2:
            raise ValueError(
                "Three-hump camel objective function is only defined for 2D."
            )
        v = 2 * x[0] ** 2 - 1.05 * x[0] ** 4 + x[0] ** 6 / 6 + x[0] * x[1] + x[1] ** 2
        return v


class EasomObjective(BenchmarkObjective):
    bounds_2d = (-100, 100) * 2
    global_minima_2d = ((np.pi, np.pi, -1),)

    def __call__(self, kwargs):
        x = np.array(list(kwargs.values()))
        if len(x) > 2:
            raise ValueError("Easom objective function is only defined for 2D.")
        v = (
            -np.cos(x[0])
            * np.cos(x[1])
            * np.exp(-((x[0] - np.pi) ** 2 + (x[1] - np.pi) ** 2))
        )
        return v


class CrossInTrayObjective(BenchmarkObjective):
    bounds_2d = (-10, 10) * 2
    global_minima_2d = (
        (1.34941, -1.34941, -2.06261),
        (1.34941, 1.34941, -2.06261),
        (-1.34941, 1.34941, -2.06261),
        (-1.34941, -1.34941, -2.06261),
    )

    def __call__(self, kwargs):
        x = np.array(list(kwargs.values()))
        if len(x) > 2:
            raise ValueError("Cross-in-tray objective function is only defined for 2D.")
        v = (
            -0.0001
            * (
                np.abs(
                    np.sin(x[0])
                    * np.sin(x[1])
                    * np.exp(np.abs(100 - np.sqrt(x[0] ** 2 + x[1] ** 2) / np.pi))
                )
                + 1
            )
            ** 0.1
        )
        return v


class EggHolderObjective(BenchmarkObjective):
    bounds_2d = (-512, 512) * 2
    global_minima_2d = ((512, 404.2319, -959.6407),)

    def __call__(self, kwargs):
        x = np.array(list(kwargs.values()))
        if len(x) > 2:
            raise ValueError("Egg-holder objective function is only defined for 2D.")
        v = -(x[1] + 47) * np.sin(np.sqrt(np.abs(x[0] / 2 + (x[1] + 47)))) - x[
            0
        ] * np.sin(np.sqrt(np.abs(x[0] - (x[1] + 47))))
        return v


class HolderTableObjective(BenchmarkObjective):
    bounds_2d = (-10, 10) * 2
    global_minima_2d = (
        (8.05502, 9.66459, -19.2085),
        (-8.05502, 9.66459, -19.2085),
        (8.05502, -9.66459, -19.2085),
        (-8.05502, -9.66459, -19.2085),
    )

    def __call__(self, kwargs):
        x = np.array(list(kwargs.values()))
        if len(x) > 2:
            raise ValueError("Holder table objective function is only defined for 2D.")
        v = -np.abs(
            np.sin(x[0])
            * np.cos(x[1])
            * np.exp(np.abs(1 - np.sqrt(x[0] ** 2 + x[1] ** 2) / np.pi))
        )
        return v


class McCormickObjective(BenchmarkObjective):
    bounds_2d = (-1.5, 4, -3, 4)
    global_minima_2d = ((-0.54719, -1.54719, -1.9133),)

    def __call__(self, kwargs):
        x = np.array(list(kwargs.values()))
        if len(x) > 2:
            raise ValueError("McCormick objective function is only defined for 2D.")
        v = np.sin(x[0] + x[1]) + (x[0] - x[1]) ** 2 - 1.5 * x[0] + 2.5 * x[1] + 1
        return v


class SchafferN2Objective(BenchmarkObjective):
    bounds_2d = (-100, 100) * 2
    global_minima_2d = ((0, 0, 0),)

    def __call__(self, kwargs):
        x = np.array(list(kwargs.values()))
        if len(x) > 2:
            raise ValueError("Schaffer N.2 objective function is only defined for 2D.")
        v = (
            0.5
            + (np.sin(x[0] ** 2 - x[1] ** 2) ** 2 - 0.5)
            / (1 + 0.001 * (x[0] ** 2 + x[1] ** 2)) ** 2
        )
        return v


class SchafferN4Objective(BenchmarkObjective):
    bounds_2d = (-100, 100) * 2
    global_minima_2d = (
        (0, 1.25313, 0.292579),
        (0, -1.25313, 0.292579),
        (1.25313, 0, 0.292579),
        (-1.25313, 0, 0.292579),
    )

    def __call__(self, kwargs):
        x = np.array(list(kwargs.values()))
        if len(x) > 2:
            raise ValueError("Schaffer N.4 objective function is only defined for 2D.")
        v = (
            0.5
            + (np.cos(np.sin(np.abs(x[0] ** 2 - x[1] ** 2))) ** 2 - 0.5)
            / (1 + 0.001 * (x[0] ** 2 + x[1] ** 2)) ** 2
        )
        return v


class StyblinskiTangObjective(BenchmarkObjective):
    bounds_2d = (-5, 5) * 2
    global_minima_2d = ((-2.903534, -2.903534, -39.166165 * 2),)
    # global minima = -39.16617 * n_dim < f(-2.903534, ..., -2.903534) < -39.16616 * n_dim
    # source - https://en.wikipedia.org/wiki/Test_functions_for_optimization

    def __call__(self, kwargs):
        x = np.array(list(kwargs.values()))
        v = np.sum(x**4 - 16 * x**2 + 5 * x) / 2
        return v
