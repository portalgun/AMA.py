import cProfile
from ..src.wrapper import wrapper
from ..src.wrapper import wrapper
from ..src.engine import engine
import ..src.ama.wrapper as w

stimFname="/home/dambam/Code/mat/projects/ama/AMA_GITHUB/AMAdataDisparity.mat"
alpha=1.3600;
s0=0.230;
rmax=5.70

nF=4
nFix=2
nFset=2
btchSz=0 #XXX
maxiter=100
#stpSize=1.4901161193847656e-08
errType=2
normType=0
bMean=1
bRectify=0
tol=1e-8
optMethod='COBYLA'


f=w.load_filters(stimFname)
f=w.flatten_filters(f)

AMA=wrapper(stimFname,alpha,s0, rmax, errType,normType,bRectify,bMean ,nF,nFix,nFset,btchSz, optMethod,maxiter,tol);
AMA.f=f

#print(AMA.f.shape)
#
cProfile.run('AMA.AMA.objective_fun(f)')
#2.2
#2.2
# 2.477 sec objective fun
#   2.473 engine(get_posterior)
#       2.453 get_Y
#           1.226 sum
#           1.204 wrap reduction
#           1.237 implement_array_function
#           1.174 reduce numpy.ufunc


AMA.plot_f()

#AMA.plot_S(1,3)
#AMA.plot_Ac(1,3)

#AMA.plot_r()
#AMA.plot_R()
#AMA.plot_mean_posteriors()


#AMA.plot_R()
