#/usr/bin/env python3

import cProfile
from ..src.wrapper import wrapper

stimFname="/home/dambam/Code/mat/projects/ama/AMA_GITHUB/AMAdataDisparity.mat"
alpha=1.3600;
s0=0.230;
rmax=5.70

nF=4
nFix=2
nFset=2
btchSz=0 #XXX
maxiter=10000
#stpSize=1.4901161193847656e-08
errType=2
normType=0
bRectify=0
bMean=1
bPrint=1
tol=1e-8
tol=None
#optMethod='BFGS'
optMethod='SLSQP'
#optMethod='COBYLA'
#optMethod='BFGS'

AMA=wrapper(stimFname,alpha,s0, rmax, errType,normType,bRectify,bMean, nF,nFix,nFset,btchSz, optMethod,maxiter,tol,bPrint);

AMA.print_cons()
cProfile.run(AMA.optimize())
AMA.print_cons()
AMA.plot_f()

# target cost: 59

# TODO
# P
# test cost
# constraints?
# Ac?
