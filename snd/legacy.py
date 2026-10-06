"""
Conversion of the flat model dictionaries of `neural_network_dynamics`
(ntwk) into `snd` model descriptions.

ntwk unit system: ms, mV, pF, nS, pA, Hz

    'Exc_Gl'    -> populations.Exc.params.gL
    'Exc_Cm'    -> C,   'Exc_Trefrac' -> tref,  'Exc_El' -> EL,
    'Exc_Vthre' -> VT,  'Exc_Vreset'  -> Vr,    'Exc_deltaV' -> DeltaT,
    'Exc_a', 'Exc_b', 'Exc_tauw' -> a, b, tauw
    'p_Exc_Inh' -> projections.(Exc, Inh).indegree = int(p*N_Exc)
    'Q_Exc_Inh' -> projections.(Exc, Inh).w
    'psyn_Exc_Inh' -> projections.(Exc, Inh).p_release
    'Erev_Exc', 'Tsyn_Exc' (or 'Erev_<pop>', 'Tsyn_<pop>') -> channels
    'F_AffExc' / 'Farray_AffExc' -> afferents.AffExc.rate

One channel per (Erev, Tsyn) pair: 'AMPA' if Erev > -20 mV, 'GABA' otherwise
(suffixed by the source population name if several pairs collide).
"""
import numpy as np
from brian2 import ms, mV, nS, pF, pA, Hz

CELL_KEYS = {'Gl': ('gL', nS), 'Cm': ('C', pF), 'Trefrac': ('tref', ms),
             'El': ('EL', mV), 'Vthre': ('VT', mV), 'Vreset': ('Vr', mV),
             'deltaV': ('DeltaT', mV), 'a': ('a', nS), 'b': ('b', pA),
             'tauw': ('tauw', ms)}

UNSUPPORTED = ['Ioscill_amp', 'Vclamp', 'alpha_', 'SpatialDecay_', 'Delay_',
               'IncreasingStep']


def _cell(Model, pop):
    old = {k[len(pop)+1:]: v for k, v in Model.items()
           if k.startswith(pop+'_') and k[len(pop)+1:] in CELL_KEYS}
    adaptation = (old.get('a', 0) != 0) or (old.get('b', 0) != 0)
    if old.get('deltaV', 0) == 0:
        kind = 'AdLIF' if adaptation else 'LIF'
        old.pop('deltaV', None)
    else:
        kind = 'AdExp' if adaptation else 'EIF'
    if not adaptation:
        for k in ['a', 'b', 'tauw']:
            old.pop(k, None)
    params = {CELL_KEYS[k][0]: v*CELL_KEYS[k][1] for k, v in old.items()}
    if 'deltaV' in old:  # ntwk: detection at Vthre + 5*deltaV
        params['Vcut'] = params['VT']+5*params['DeltaT']
    return dict(N=int(Model['N_'+pop]), kind=kind, params=params)


def _synapse_type(Model, source):
    """ (Erev, Tsyn) of a source population, as in ntwk """
    def get(key):
        if '%s_%s' % (key, source) in Model:
            return Model['%s_%s' % (key, source)]
        if 'Inh' in source:
            return Model['%s_Inh' % key]
        return Model['%s_Exc' % key]
    return get('Erev'), get('Tsyn')


def from_ntwk(Model, REC_POPS=None, AFF_POPS=None):
    """
    snd model description from a ntwk flat dictionary
    (REC_POPS / AFF_POPS default to Model['REC_POPS'] / Model['AFF_POPS'])
    """
    REC_POPS = list(REC_POPS if REC_POPS is not None else Model['REC_POPS'])
    AFF_POPS = list(AFF_POPS if AFF_POPS is not None else Model.get('AFF_POPS', []))

    unsupported = [k for k in Model if any(u in k for u in UNSUPPORTED)]
    if unsupported:
        raise NotImplementedError('ntwk features not converted: %s' % unsupported)

    # channels
    channels, channel_of = {}, {}
    for source in REC_POPS+AFF_POPS:
        Erev, Tsyn = _synapse_type(Model, source)
        name = 'AMPA' if Erev > -20 else 'GABA'
        spec = dict(kind='exp', E=Erev*mV, tau=Tsyn*ms)
        if name in channels and \
                (channels[name]['E'] != spec['E'] or channels[name]['tau'] != spec['tau']):
            name = '%s_%s' % (name, source)
        channels[name] = spec
        channel_of[source] = name

    new = {'dt': Model['dt']*ms, 'tstop': Model['tstop']*ms,
           'seed': int(Model.get('SEED', 1)),
           'channels': channels,
           'populations': {pop: _cell(Model, pop) for pop in REC_POPS},
           'afferents': {}, 'projections': {}}

    for aff in AFF_POPS:
        if 'Farray_%s' % aff in Model:
            rate = np.asarray(Model['Farray_%s' % aff])*Hz
        else:
            rate = Model.get('F_%s' % aff, 0.)*Hz
        new['afferents'][aff] = dict(N=int(Model['N_'+aff]), rate=rate)

    for pre in REC_POPS+AFF_POPS:
        for post in REC_POPS:
            p = Model.get('p_%s_%s' % (pre, post), 0)
            Q = Model.get('Q_%s_%s' % (pre, post), 0)
            if p > 0 and Q != 0:
                proj = dict(channel=channel_of[pre], w=Q*nS,
                            indegree=int(p*Model['N_'+pre]))
                if 'psyn_%s_%s' % (pre, post) in Model:
                    proj['p_release'] = Model['psyn_%s_%s' % (pre, post)]
                new['projections'][(pre, post)] = proj

    return new
