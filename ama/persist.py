"""Saving and loading Units (pickle of plain data) and their yaml configurations."""
from ._base import *
from .nrn import Nrn
from .model import Model
from .objective import Objective
from .optimizer import Optimizer


class _Persistence:
    """Unit methods: save/load and yaml configuration"""

    #- save/load
    def save(self,fname):
        """save settings, filters, optimizer state, random keys, and frozen whitening (not the stimuli) with pickle"""
        if not self.nrn.bFinalized:
            raise Exception('train (or finalize) the unit first')
        asnp=lambda tree: None if tree is None else tree_util.tree_map(np.asarray,tree)
        state={'version':1,
               'nrn':_get_copy_dict(self.nrn,_NRN_EXCL),
               'model':_get_copy_dict(self.model),
               'objective':_get_copy_dict(self.objective),
               'optimizer':None if self.optimizer is None else _get_copy_dict(self.optimizer,_OPT_EXCL),
               'finalize':dict(n=self.filter.n,dtype=np.dtype(self.nrn.dtype).name,bFourier=self.nrn.bFourier,
                               bAnalytic=bool(self.nrn.bAnalytic),bSplit=self.nrn.bSplit,
                               stimInd=getattr(self,'_stimInd',None)),
               'out':np.asarray(self.filter.out),
               'last':asnp(self.filter.last),
               'opt_state':asnp(self.opt_state),
               'opt_param_shape':getattr(self,'_opt_param_shape',None),
               'opt_state_key':getattr(self,'_opt_state_key',None),
               'loss_hist':list(getattr(self.optimizer,'loss_hist',[])),
               'restart_costs':self.restart_costs,
               'seed':self.seed,
               'rng':np.asarray(jxrandom.key_data(self.rng)),
               'rng_last':None if self.rng_last is None else np.asarray(jxrandom.key_data(self.rng_last)),
               'W':asnp(self.nrn._W),
               'pool_p':asnp(self.pool_p),
               'nrn_p':asnp(getattr(self,'nrn_p',None)),
               'train_log':list(getattr(self,'train_log',[]) or []),
               'ama_source_sha256':source_sha256(),
               'name':getattr(self,'name',None),
               'multiscale_out':getattr(self,'multiscale_out',None),
               'param_out':getattr(self,'param_out',None),
               'stim_summary':self._stim_summary()}
        with open(fname,'wb') as fh:
            pickle.dump(state,fh)

    @classmethod
    def load(cls,fname,stim):
        """load a unit saved with Unit.save, with its training stimuli"""
        with open(fname,'rb') as fh:
            state=pickle.load(fh)
        asjnp=lambda tree: None if tree is None else tree_util.tree_map(jnp.asarray,tree)
        unit=cls(stim,Nrn(**state['nrn']),Model(**state['model']),Objective(**state['objective'],_bCopy=True),
                 optimizer=None if state['optimizer'] is None else Optimizer(**state['optimizer']),
                 seed=state['seed'],rng=jxrandom.wrap_key_data(state['rng']),
                 rng_last=None if state['rng_last'] is None else jxrandom.wrap_key_data(state['rng_last']),name=state.get('name'))
        fin=state['finalize']
        fourierType=(2 if fin['bAnalytic'] else 1) if fin['bFourier'] else 0
        if unit.optimizer is None:
            unit.optimizer=Optimizer()
        unit._finalize(fin['n'],np.arange(fin['n']),fourierType=fourierType,bSplit=fin['bSplit'],dtype=jnp.dtype(fin['dtype']),
                       stimInd=fin.get('stimInd'))
        if state['optimizer'] is None:
            unit.optimizer=None
        unit.filter.out=jnp.asarray(state['out'])
        unit.filter.last=asjnp(state['last'])
        unit.opt_state=asjnp(state['opt_state'])
        unit._opt_param_shape=state['opt_param_shape']
        unit._opt_state_key=state.get('opt_state_key')                   # older saves: the state is not reused
        if unit.optimizer is not None:
            unit.optimizer.loss_hist=state['loss_hist']
        unit.restart_costs=state['restart_costs']
        unit.nrn._W=asjnp(state['W'])
        unit.pool_p=None if state.get('pool_p') is None else np.asarray(state['pool_p'])
        unit.nrn_p=None if state.get('nrn_p') is None else {k:np.asarray(v) for k,v in state['nrn_p'].items()}
        unit.train_log=list(state.get('train_log') or [])
        unit.name=state.get('name')
        if state.get('multiscale_out') is not None:
            unit.multiscale_out=state['multiscale_out']
        if state.get('param_out') is not None:
            unit.param_out=state['param_out']
        return unit

    #- yaml configuration
    def _stim_summary(self):
        st=self.stim_full
        return {'dims':[int(d) for d in st.dims],'nCtg':int(st.nCtg),'nStim':int(np.sum(np.asarray(st.weights)>0)),
                'Y':_yaml_safe(np.asarray(st.Y)),'Yperiod':_yaml_safe(getattr(st,'Yperiod',None)),
                'bContinuous':bool(getattr(st,'bContinuous',False)),
                'bIsFourier':bool(st.bIsFourier),'nSplit':int(st.nSplit or 0)}

    def config(self):
        """
        this unit's options and settings as a plain dict (yaml-safe): Nrn, Model, Objective and Optimizer settings, the
        seed, the filter layout, a summary of the training stimuli, and the training calls made so far (method and
        arguments, from train_new / train_recurse / train_append / train_parametric / train_multiscale). Rebuild with
        Unit.from_config. Learned filters are not included (use save / load).
        """
        cfg={'ama_config_version':CONFIG_VERSION,'ama_source_sha256':source_sha256(),'name':getattr(self,'name',None),
             'seed':_yaml_safe(self.seed)}
        cfg.update(_components_config(self.nrn,self.model,self.objective,self.optimizer))
        if self.nrn.bFinalized:
            cfg['filters']={'n':int(self.filter.n),'fourierType':(2 if self.nrn.bAnalytic else 1) if self.nrn.bFourier else 0,
                            'dtype':np.dtype(self.nrn.dtype).name,'bSplit':bool(self.nrn.bSplit),
                            'shape':list(np.shape(self.filter.out)),'bPoolWeights':self.pool_p is not None}
        cfg['train']=_yaml_safe(list(getattr(self,'train_log',[]) or []))
        cfg['stim']=self._stim_summary()
        return cfg

    def save_config(self,fname):
        """write config() to a yaml file"""
        import yaml
        with open(fname,'w') as fh:
            yaml.safe_dump(self.config(),fh,sort_keys=False,default_flow_style=None,width=120)

    @classmethod
    def from_config(cls,cfg,stim,bTrain=False,stimVal=None):
        """
        build a unit from a configuration (dict, or the path of a yaml file written by save_config / config_from_saved)
        with these training stimuli. bTrain replays the recorded training calls in order; stimVal is passed to calls that
        were made with validation stimuli. Unknown top-level keys (e.g. project metadata) are ignored.
        """
        if isinstance(cfg,str) or hasattr(cfg,'__fspath__'):
            cfg=load_config(cfg)
        if int(cfg.get('ama_config_version',1))>CONFIG_VERSION:
            raise Exception('config version ' + str(cfg.get('ama_config_version')) + ' is newer than this ama (' + str(CONFIG_VERSION) + ')')
        opt=cfg.get('optimizer')
        unit=cls(stim,Nrn(**cfg['nrn']),Model(**cfg['model']),Objective(**cfg['objective'],_bCopy=True),
                 optimizer=None if opt is None else Optimizer(**opt),seed=cfg.get('seed'),name=cfg.get('name'))
        if bTrain:
            for step in cfg.get('train') or []:
                args=dict(step.get('args') or {})
                if 'stimVal' in args:
                    args['stimVal']=stimVal if args['stimVal'] else None
                if args.get('optimizer') is not None:
                    args['optimizer']=Optimizer(**args['optimizer'])
                if args.get('dtype') is not None:
                    args['dtype']=jnp.dtype(args['dtype'])
                if step['method'] not in ('train_new','train_recurse','train_append','train_parametric','train_multiscale'):
                    raise Exception('unknown training method in config: ' + str(step['method']))
                getattr(unit,step['method'])(**args)
        return unit


__all__=['_Persistence']
