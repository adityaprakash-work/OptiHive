# ---INFO-----------------------------------------------------------------------
# Author: Aditya Prakash
# Last edited: 2023-07-17

# --Needed functionalities

# ---DEPENDENCIES---------------------------------------------------------------
from tqdm import tqdm


# ---TRACKERS-------------------------------------------------------------------
class Tracker(object):
    """
    A tracker is an object that tracks the progress of an optimization algorithm
    during its run.
    """

    def __init__(self):
        self.set_trackable()

    def set_trackable(self, trackable=None):
        self.trackable = trackable

    def track(self, iteration):
        pass

    def cease_tracking(self):
        pass


class ProgressBarTracker(Tracker):
    """
    A progress bar tracker is a tracker that tracks the progress of an
    optimization algorithm using a progress bar.
    """

    def __init__(self):
        self.pb = None
        self.set_trackable()

    def track(self, iteration):
        if self.pb is None:
            self.pb = tqdm(total=self.trackable.run_last_n_iterations)
        self.pb.update(1)

    def cease_tracking(self):
        self.pb.close()
