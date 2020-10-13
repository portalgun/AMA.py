#/usr/bin/env python3

import cProfile
from ..src.wrapper import wrapper

stimFname="/home/dambam/Code/mat/projects/ama/AMA_GITHUB/AMAdataDisparity.mat"

W=wrapper(stimFname)
#cProfile.run(W.optimize())
W.optimize()

W.plot_f()

# target cost: 59

# TODO
# P
# test cost
# constraints?
# Ac?
