# ---DEPENDENCIES---------------------------------------------------------------
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from tqdm import tqdm


# ---TRACKERS-------------------------------------------------------------------
class Tracker(object):
    def __init__(self):
        self.set_trackable()

    def set_trackable(self, trackable=None):
        self.trackable = trackable

    def track(self, iteration):
        raise NotImplementedError

    def cease_tracking(self):
        raise NotImplementedError


class ProgressBarTracker(Tracker):
    def __init__(self, n_iterations):
        self.pb = tqdm(total=n_iterations)
        self.set_trackable()

    def track(self, iteration):
        self.pb.update(1)

    def cease_tracking(self):
        self.pb.close()


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
