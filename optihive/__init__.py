# ---INFO-----------------------------------------------------------------------
# Author: Aditya Prakash
# Last edited: 2023-07-17

# ---MODULES--------------------------------------------------------------------
from . import pso
from . import utils
from . import benchmarks

# ---CONSTANTS------------------------------------------------------------------
__version__ = "0.5.0"

SET_GLOBAL_VERBOSE = 0

# ---STATUS MESSAGES------------------------------------------------------------
if SET_GLOBAL_VERBOSE != 0:
    print(f"OptiHive version {__version__}")
