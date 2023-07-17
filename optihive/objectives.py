# ---INFO-----------------------------------------------------------------------
# Author: Aditya Prakash
# Last edited: 2023-07-17

# --Needed functionalities

# ---DEPENDENCIES---------------------------------------------------------------
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import lightgbm as lgb


# ---COMMON OBJECTIVES----------------------------------------------------------
class RandomForestClassifierObjective(object):
    """
    VanillaSwarm compatible objective function for a random forest classifier.
    """

    def __init__(self, X, Y, random_state=42):
        self.random_state = random_state
        self.Xtr, self.Xte, self.Ytr, self.Yte = train_test_split(
            X, Y, test_size=0.2, random_state=self.random_state
        )

    def __call__(self, kwargs):
        clf = RandomForestClassifier(**kwargs)
        clf.fit(self.Xtr, self.Ytr)
        return 1 - clf.score(self.Xte, self.Yte)


class LightGbmClassifierObjective(object):
    """
    VanillaSwarm compatible objective function for a lightgbm classifier.
    """

    def __init__(self, X, Y, random_state=42):
        self.random_state = random_state
        self.Xtr, self.Xte, self.Ytr, self.Yte = train_test_split(
            X, Y, test_size=0.2, random_state=self.random_state
        )

    def __call__(self, kwargs):
        clf = lgb.LGBMClassifier(**kwargs)
        clf.fit(self.Xtr, self.Ytr)
        return 1 - clf.score(self.Xte, self.Yte)
