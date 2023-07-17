# ---INFO-----------------------------------------------------------------------
# Author: Aditya Prakash
# Last edited: 2023-07-16

# --Needed functionalities
# - 1. Implentation of dummy particles to sample the objective contour.
# - 2. Efficient loggers in SwarmObjectiveTracker
# - 3. Add support for bool parameter types in VanillaSwarm
# - 4. Eager support in SwarmObjectiveTracker

# ---DEPENDENCIES---------------------------------------------------------------
import numpy as np

from . import utils

import matplotlib.pyplot as plt
import seaborn as sns


# ---SWARM----------------------------------------------------------------------
class VanillaSwarm(object):
    """
    Vanilla PSO

    xi: position of the ith particle
    vi: velocity of the ith particle
    iw: inertia weight
    cc: cognitive constant
    sc: social constant
    r1: random number bound
    r2: random number bound
    gw: gradient weight
    r3: random number bound

    vi(t + 1) = iw * vi(t)
                + cc * U(0, r1) * (ibest(t) - xi(t))
                + sc * U(0, r2) * (gbest(t) - xi(t))
                + gw * argmin(objective_function(J + xi(t + 1))| J = U(-r3, r3))

    xi(t + 1) = xi(t) + v(t)

    An extra term is added to the velocity update to account for the gradient
    of the objective function. The gradient is approximated by evaluating the
    objective function at J + xi(t + 1) for J = U(-r3, r3) and taking the
    minimum. The gradient weight (gw) is a hyperparameter that controls the
    strength of the gradient update. The gradients are checked n_perturb times
    with random perturbations to X. The gradient update is only applied if the
    objective function is improved. [credit: Author@INFO]
    """

    def __init__(
        self,
        search_space,
        n_particles,
        objective_function,
        iw=0.4,
        cc=0.3,
        sc=0.3,
        r1=0.3,
        r2=0.3,
        use_gradient=False,
        gw=0.1,
        n_perturb=3,
        r3=0.1,
        trackers=None,
    ):
        self.search_space = search_space
        self.n_particles = n_particles
        self.objective_function = np.vectorize(
            objective_function,
            signature="()->()",
        )
        self.iw = iw
        self.cc = cc
        self.sc = sc
        self.r1 = r1
        self.r2 = r2
        self.use_gradient = use_gradient
        self.gw = gw
        self.n_perturb = n_perturb
        self.r3 = r3
        self.trackers = trackers
        if self.trackers is not None:
            for tracker in self.trackers:
                tracker.set_trackable(self)

        self.X = self.init_X()
        self.V = self.init_V()
        self.P = self.init_P()
        self.G = self.init_G()

        self.Xs = self.evaluate(self.X)
        self.Ps = self.evaluate(self.P)
        self.Gs = self.evaluate(self.G)[0]

        # for tracking
        self.run_last_n_iterations = None
        self.total_runs = 0

    @property
    def R1(self):
        rn = np.random.uniform(0, self.r1, (self.n_particles, 1))
        return rn

    @property
    def R2(self):
        rn = np.random.uniform(0, self.r2, (self.n_particles, 1))
        return rn

    @property
    def R3(self):
        rn = np.random.uniform(
            -self.r3, self.r3, (self.n_particles, len(self.search_space))
        )
        return rn

    def init_X(self):
        X = np.empty(
            (self.n_particles, len(self.search_space)),
            dtype=np.float32,
        )
        for i, (par, (par_type, ref)) in enumerate(self.search_space.items()):
            if par_type == "cat":
                ref = list(range(len(ref)))
                X[:, i] = np.random.choice(ref, self.n_particles)
            elif par_type == "dis_int":
                X[:, i] = np.random.choice(ref, self.n_particles)
            elif par_type == "dis":
                X[:, i] = np.random.choice(ref, self.n_particles)
            elif par_type == "con":
                X[:, i] = np.random.uniform(ref[0], ref[1], self.n_particles)
        return X

    def init_V(self):
        return self.init_X() / 10

    def init_P(self):
        return self.X.copy()

    def init_G(self):
        P_eval = self.evaluate(self.P)
        return self.P[np.argmin(P_eval)].reshape(1, -1)

    @staticmethod
    def closest_to(arr, target):
        """
        Returns the closest value in arr to target

        arr: array of values
        target: target value

        returns: closest value in arr to target
        """
        arr = np.array(arr)
        target = np.array(target)

        indices = np.searchsorted(arr, target)
        indices = np.clip(indices, 1, len(arr) - 1)
        left = arr[indices - 1]
        right = arr[indices]
        diff_left = np.abs(target - left)
        diff_right = np.abs(target - right)
        mask = diff_left <= diff_right
        result = np.where(mask, left, right)

        return result

    def x_dis_parse(self, x_dis, reference, enforce_int):
        """
        Parses a discrete parameter

        x_dis: discrete parameter
        reference: reference values
        enforce_int: whether to enforce integer values

        returns: parsed discrete parameter
        """
        x_dis_p = self.closest_to(reference, x_dis)
        if enforce_int:
            x_dis_p = x_dis_p.astype(np.int32)
        return x_dis_p

    def x_cat_parse(self, x_cat, reference):
        """
        Parses a categorical parameter

        x_cat: categorical parameter
        reference: reference values

        returns: parsed categorical parameter
        """
        reference = np.array(reference)
        reference_int = list(range(len(reference)))
        x_cat_p = self.closest_to(reference_int, x_cat).astype(np.int32)
        x_cat_p = np.apply_along_axis(lambda x: reference[x], 0, x_cat_p)
        return x_cat_p

    def x_con_parse(self, x_con, reference):
        """
        Parses a continuous parameter

        x_con: continuous parameter
        reference: reference values

        returns: parsed continuous parameter
        """
        x_con_p = np.clip(x_con, reference[0], reference[1])
        return x_con_p

    def X_parse(self, X):
        if X.shape[1] != len(self.search_space):
            raise ValueError("Invalid shape for X")
        X_p = np.empty((X.shape[0], len(self.search_space)), dtype=object)
        for i, (par, (par_type, ref)) in enumerate(self.search_space.items()):
            x = X[:, i]
            if par_type == "cat":
                X_p[:, i] = self.x_cat_parse(x, ref)
            elif par_type == "dis_int":
                X_p[:, i] = self.x_dis_parse(x, ref, True)
            elif par_type == "dis":
                X_p[:, i] = self.x_dis_parse(x, ref, False)
            elif par_type == "con":
                X_p[:, i] = self.x_con_parse(x, ref)
        return X_p

    def x_to_dict(self, x):
        return dict(zip(self.search_space.keys(), x))

    def evaluate(self, X):
        X_p = self.X_parse(X)
        X_pd = np.apply_along_axis(self.x_to_dict, 1, X_p)
        return self.objective_function(X_pd)

    def update_X(self):
        self.X += self.V

    def update_V(self):
        self.V *= self.iw
        self.V += self.cc * self.R1 * (self.P - self.X)
        self.V += self.sc * self.R2 * (self.G - self.X)
        if self.use_gradient:
            V_grad = np.zeros_like(self.V)
            current_eval = self.Xs.copy()
            for _ in range(self.n_perturb):
                X_plus_dX = self.X + self.R3
                X_plus_dX_eval = self.evaluate(X_plus_dX)
                update = X_plus_dX_eval < current_eval
                V_grad[update] = self.R3[update]
                current_eval[update] = X_plus_dX_eval[update]
            self.V += self.gw * V_grad

    def update_P(self):
        update = self.Xs < self.Ps
        self.P[update] = self.X[update]

    def update_G(self):
        if np.min(self.Xs) < self.Gs:
            self.G = self.X[np.argmin(self.Xs)].reshape(1, -1)

    def update_Xs(self):
        self.Xs = self.evaluate(self.X)

    def update_Ps(self):
        update = self.Xs < self.Ps
        self.Ps[update] = self.Xs[update]

    def update_Gs(self):
        if np.min(self.Xs) < self.Gs:
            self.Gs = np.min(self.Xs)

    def run(self, n_iterations):
        self.run_last_n_iterations = n_iterations
        self.total_runs += n_iterations

        for i in range(n_iterations):
            self.update_X()
            self.update_Xs()
            self.update_V()
            self.update_P()
            self.update_Ps()
            self.update_G()
            self.update_Gs()
            if self.trackers is not None:
                for tracker in self.trackers:
                    tracker.track(i)
        for tracker in self.trackers:
            tracker.cease_tracking()


# ---SWARM TRACKERS-------------------------------------------------------------
class SwarmObjectiveTracker(utils.Tracker):
    """
    Tracks the objective function of the swarm

    eager_step_particles: iterations that are multiples of 'this' number are
                        tracked for particles
    eager_step_objective: iterations that are multiples of 'this' number are
                        tracked for objective
    eager_cap_objective: maximum iteration where data for approx. contour is
                        logged
    eager: whether to draw live plots or not
    lazy_step: iterations that are multiples of 'this' number are tracked
    lazy_cap_objective: maximum iteration where data for approx. contour is
                        logged
    n_dummy_particles: number of dummy particles to sample the objective
                        contour

    The objective function is tracked in two ways:
    1. Eager: The objective function is tracked for all particles at every
            iteration that is a multiple of 'eager_step_particles'. The
            objective function is also tracked for the global best particle
            at every iteration that is a multiple of 'eager_step_objective'.
            The objective function is plotted live at every iteration that is
            a multiple of 'eager_step_objective'. The particles are  plotted
            live at every iteration that is a multiple of 'eager_step_particles'
            and the global best particle is plotted live at every iteration that
            is a multiple of 'eager_step_particle'.
    2. Lazy: The objective function is tracked for all particles at every
            iteration that is a multiple of 'lazy_step'. The objective function
            is also tracked for the global best particle at every iteration
            that is a multiple of 'lazy_step'. The objective function is plotted
            at every iteration that is a multiple of 'lazy_step'. The particles
            are plotted at every iteration that is a multiple of 'lazy_step'
            and the global best particle is plotted at every iteration that is
            a multiple of 'lazy_step'.

    The objective function is approximated by sampling the search space with
    'n_dummy_particles' dummy particles and real particles. The dummy particles
    move randomly.
    """

    def __init__(
        self,
        track_params,
        eager_step_particles=3,
        eager_step_objective=9,
        eager_cap_objective=100,
        eager=True,
        lazy_step=3,
        lazy_cap_objective=100,
        n_dummy_particles=None,
    ):
        if len(track_params) != 2:
            raise ValueError("Only 2D plots are supported")
        self.track_params = track_params
        self.esp = eager_step_particles
        self.eso = eager_step_objective
        self.eco = eager_cap_objective
        self.eager = eager
        self.ls = lazy_step
        self.lco = lazy_cap_objective
        self.ndp = n_dummy_particles
        self.XP_log = None
        self.XO_log = None
        self.XOs_log = None
        self.XL_log = None
        self.XLs_log = None
        self.swarm_consts = None
        self.set_trackable()

    def track(self, iteration):
        if self.swarm_consts is None:
            self.swarm_consts = {}
            self.swarm_consts["tpi"] = [
                list(self.trackable.search_space.keys()).index(param)
                for param in self.track_params
            ]
            self.swarm_consts["npr"] = self.trackable.n_particles
            self.swarm_consts["ss"] = self.trackable.search_space
        if self.eager:
            pass
        else:
            if iteration % self.ls == 0:
                if self.XL_log is None:
                    self.XL_log = self.trackable.X[:, self.swarm_consts["tpi"]]
                    self.XLs_log = self.trackable.Xs
                else:
                    self.XL_log = np.append(
                        self.XL_log,
                        self.trackable.X[:, self.swarm_consts["tpi"]],
                        axis=0,
                    )
                    self.XLs_log = np.append(
                        self.XLs_log,
                        self.trackable.Xs,
                        axis=0,
                    )

    def draw_lazy(
        self, particle_indices, cmap="RdYlBu", xlim=None, ylim=None, levels=50
    ):
        if self.XL_log is None:
            raise ValueError("No data to draw")
        else:
            sns.set_style("darkgrid")
            sns.set_context("paper")
            plt.figure(figsize=(6, 5))
            tcf = plt.tricontourf(
                self.XL_log[:, 0],
                self.XL_log[:, 1],
                self.XLs_log,
                cmap=cmap,
                levels=levels,
            )
            for pi in particle_indices:
                pix = self.XL_log[pi :: self.swarm_consts["npr"], 0]
                piy = self.XL_log[pi :: self.swarm_consts["npr"], 1]
                ix, iy, fx, fy = pix[0], piy[0], pix[-1], piy[-1]
                plt.plot(
                    pix,
                    piy,
                    color="black",
                    marker="o",
                    markersize=4,
                )
                plt.scatter(ix, iy, color="green", marker="o", s=30, zorder=10)
                plt.scatter(fx, fy, color="red", marker="o", s=30, zorder=10)
            if xlim is not None:
                plt.xlim(xlim[0], xlim[1])
            else:
                plt.xlim(
                    min(self.swarm_consts["ss"][self.track_params[0]][1]),
                    max(self.swarm_consts["ss"][self.track_params[0]][1]),
                )
            if ylim is not None:
                plt.ylim(ylim[0], ylim[1])
            else:
                plt.ylim(
                    min(self.swarm_consts["ss"][self.track_params[1]][1]),
                    max(self.swarm_consts["ss"][self.track_params[1]][1]),
                )
            plt.xlabel(self.track_params[0])
            plt.ylabel(self.track_params[1])
            plt.title("Objective Function Contour")
            plt.colorbar(tcf)
            plt.show()
