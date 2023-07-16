# ---INFO-----------------------------------------------------------------------
# Author: Aditya Prakash
# Last edited: 2023-07-16

# ---DEPENDENCIES---------------------------------------------------------------
import numpy as np
from tqdm import tqdm


# ---TRACKER--------------------------------------------------------------------
class Tracker(object):
    def __call__(self, swarm, iteration):
        self.task(swarm, iteration)

    def task(self, swarm, iteration):
        raise NotImplementedError


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
        tracker=None,
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
        self.tracker = tracker

        self.X = self.init_X()
        self.V = self.init_V()
        self.P = self.init_P()
        self.G = self.init_G()

        self.Ps = self.evaluate(self.P)
        self.Gs = self.evaluate(self.G)[0]

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
        x_dis_p = self.closest_to(reference, x_dis)
        if enforce_int:
            x_dis_p = x_dis_p.astype(np.int32)
        return x_dis_p

    def x_cat_parse(self, x_cat, reference):
        reference = np.array(reference)
        reference_int = list(range(len(reference)))
        x_cat_p = self.closest_to(reference_int, x_cat).astype(np.int32)
        x_cat_p = np.apply_along_axis(lambda x: reference[x], 0, x_cat_p)
        return x_cat_p

    def x_con_parse(self, x_con, reference):
        x_con_p = np.clip(x_con, reference[0], reference[1])
        return x_con_p

    def X_parse(self, X):
        if X.shape[1] != len(self.search_space):
            raise ValueError("Invalid shape for X")
        X_p = np.empty((self.n_particles, len(self.search_space)), dtype=object)
        for i, (par, (par_type, ref)) in enumerate(self.search_space.items()):
            x = X[:, i]
            if par_type == "cat":
                X_p[:, i] = self.x_cat_parse(x, ref)
            if par_type == "dis_int":
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
        if self.gw is not None:
            X_eval = self.evaluate(self.X)
            V_grad = np.empty_like(self.V)
            for _ in range(self.n_perturb):
                X_plus_dX = self.X + self.R3
                X_plus_dX_eval = self.evaluate(X_plus_dX)
                update = X_plus_dX_eval < X_eval
                V_grad[update] = self.R3[update]
            self.V += self.gw * V_grad

    def update_P_G(self):
        X_eval = self.evaluate(self.X)
        update = X_eval < self.Ps
        self.P[update] = self.X[update]
        self.Ps[update] = X_eval[update]
        if np.min(X_eval) < self.Gs:
            self.G = self.X[np.argmin(X_eval)].reshape(1, -1)
            self.Gs = np.min(X_eval)

    def run(self, n_iterations):
        for i in tqdm(range(n_iterations)):
            self.update_X()
            self.update_V()
            self.update_P_G()
            if self.tracker is not None:
                self.tracker(self, i)
