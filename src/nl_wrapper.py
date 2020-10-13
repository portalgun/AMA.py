import nlopt
import jax.numpy as np

class nl_wrapper:
    def __init__(self,AMA,optOpts,consOpts):
        self.optOpts=optOpts
        self.consOpts=consOpts
        self.AMA=AMA

        self.consOpts.set_filter_ind(self.AMA)

        if self.optOpts.algorithm=='COBYLA':
            self.optOpts.algorithm=nlopt.LN_COBYLA
        elif self.optOpts.algorithm=='BOBYQA':
            self.optOpts.algorithm=nlopt.LN_BOBYQA
        elif self.optOpts.algorithm=='PRAXIS':
            self.optOpts.algorithm=nlopt.LN_PRAXIS
        elif self.optOpts.algorithm=='NELDERMEAD':
            # USE SBLX INSTEAD
            self.optOpts.algorithm=nlopt.LN_NELDERMEAD
        elif self.optOpts.algorithm=='SBPLX':
            self.optOpts.algorithm=nlopt.LN_SBPLX
        elif self.optOpts.algorithm=='SLSQP':
            self.optOpts.algorithm=nlopt.LD_SLSQP

        self.n=self.AMA.Nf*self.AMA.nPix

        self.opt=nlopt.opt(self.optOpts.algorithm,self.n)

        if consOpts.bAug==1:
            self.local_opt=self.opt;
            self.opt=nlopt.opt(nlopt.LD_AUGLAG,self.n)
            self.opt.set_local_optimizer(self.local_opt)


        self.opt.set_min_objective(lambda x,grad: self.AMA.objective_fun(x,grad))

        if self.consOpts.bLB:
            lb=np.repeat(self.consOpts.LB,self.n)
            self.opt.set_lower_bounds(lb)

        if self.consOpts.bUB:
            ub=np.repeat(self.consOpts.UB,self.n)
            self.opt.set_upper_bounds(ub)

        if self.consOpts.bSumE:
            c=lambda res,x,grad: self.consOpts.vec_mag_one_fun(res,x,grad)
            #m=self.n
            m=self.AMA.Nf
            self.opt.add_equality_mconstraint(
                c,
                np.repeat(self.consOpts.sumTol,m))

        if self.optOpts.stopval:
            self.opt.set_stopval(self.optOpts.stopval)
        if self.optOpts.ftolAbs:
            self.opt.set_ftol_rel(self.optOpts.ftolAbs)
        if self.optOpts.ftolRel:
            self.opt.set_ftol_abs(self.optOpts.ftolRel)
        if self.optOpts.xtolAbs:
            self.opt.set_xtol_abs(self.optOpts.xtolAbs)
        if self.optOpts.xtolRel:
            self.opt.set_xtol_rel(self.optOpts.xtolRel)
        if self.optOpts.maxeval:
            self.opt.set_maxeval(self.optOpts.maxeval)
        if self.optOpts.maxtime:
            self.opt.set_maxtime(self.optOpts.maxtime)
        if self.optOpts.step:
            self.opt.set_initial_step(self.optOpts.step)


    def optimize(self):
        f0=self.AMA.get_f0()
        self.fopt    = self.opt.optimize(f0)

    def get_results(self):
        opt_val = self.opt.last_optimum_value()
        result  = self.opt.last_optimize_result()
