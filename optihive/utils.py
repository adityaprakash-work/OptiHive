# ---DEPENDENCIES---------------------------------------------------------------
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# ---TRACKER--------------------------------------------------------------------
class Tracker(object):
    def __call__(self, swarm, iteration):
        self.task(swarm, iteration)

    def task(self, swarm, iteration):
        raise NotImplementedError


# ---BENCHMARK FUNCTIONS--------------------------------------------------------
class RandomForestClassifierObjective(object):
    def __init__(self, X, Y, random_state=42):
        self.random_state = random_state
        self.Xtr, self.Xte, self.Ytr, self.Yte = train_test_split(
            X, Y, test_size=0.2, random_state=self.random_state
        )

    def __call__(self, kwargs):
        clf = RandomForestClassifier(**kwargs)
        clf.fit(self.Xtr, self.Ytr)
        return 1 - clf.score(self.Xte, self.Yte)