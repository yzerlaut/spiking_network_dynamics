"""
Layer 2 -- neuronal populations.

`population` returns a plain `brian2.NeuronGroup`, it is equivalent to:

    eqs = equations.neuron('AdExp')['model'] \\
        + equations.channel('AMPA') + equations.channel('GABA') \\
        + equations.synaptic_current(['AMPA', 'GABA'])
    G = b2.NeuronGroup(N, eqs, threshold='V > Vcut', reset='V = Vr; w += b',
                       refractory='tref', method='euler', name='Exc',
                       namespace={'gL': 10*nS, ..., 'E_AMPA': 0*mV, 'tau_AMPA': 5*ms, ...})
    G.V = 'EL'

Parameters are looked up in `G.namespace` (shared by all neurons). A
parameter given as an array of size N becomes a per-neuron constant variable
(e.g. `G.VT`), to model heterogeneous populations.
"""
import numpy as np
import brian2 as b2
from brian2 import mV, ms, nS, pF, pA

from . import equations

DEFAULTS = {'C': 200*pF, 'gL': 10*nS, 'EL': -70*mV,
            'VT': -50*mV, 'DeltaT': 2*mV, 'Vr': -70*mV, 'tref': 5*ms,
            'a': 0*nS, 'b': 0*pA, 'tauw': 500*ms}


def unit_string(value):
    """ unit of a quantity, as written in brian2 equations ('volt', '1', ...) """
    if b2.Quantity(value).is_dimensionless:
        return '1'
    return repr(b2.get_unit(b2.get_dimensions(value)))


def population(name, N,
               kind='AdExp',
               params=None,
               channels=None,
               method='euler',
               **group_kwargs):
    """
    name      : name of the brian2 object (G.name, net[name])
    N         : number of neurons
    kind      : membrane model, one of equations.MEMBRANE ('LIF', 'EIF', 'AdLIF', 'AdExp')
    params    : membrane parameters (brian2 quantities), missing ones are
                taken from DEFAULTS (Vcut defaults to VT + 5*DeltaT)
    channels  : synaptic channels, e.g.
                {'AMPA': dict(kind='exp', E=0*mV, tau=5*ms),
                 'GABA': dict(kind='exp', E=-80*mV, tau=5*ms)}
    other keyword arguments are passed to brian2.NeuronGroup
    """
    params, channels = dict(params or {}), dict(channels or {})

    expected = equations.PARAMETERS[kind]
    unknown = set(params)-set(expected)
    if unknown:
        raise ValueError("unknown parameter(s) %s for a '%s' population "
                         "(expected: %s)" % (sorted(unknown), kind, expected))
    for key in expected:
        if (key not in params) and (key in DEFAULTS):
            params[key] = DEFAULTS[key]
    if ('Vcut' in expected) and ('Vcut' not in params):
        params['Vcut'] = params['VT']+5*params['DeltaT']

    # equations: membrane + channels + summed synaptic current
    neuron = equations.neuron(kind)
    eqs = neuron.pop('model')
    namespace = {}
    for c, spec in channels.items():
        spec = dict(spec)
        ckind = spec.pop('kind', 'exp')
        eqs += equations.channel(c, ckind)
        namespace.update(equations.channel_namespace(c, ckind, **spec))
    eqs += equations.synaptic_current(list(channels))

    # parameters: shared (namespace) or per-neuron (constant variables)
    heterogeneous = {}
    for key, val in params.items():
        if np.size(val) > 1:
            if np.size(val) != N:
                raise ValueError("parameter '%s' has %i values for %i neurons"
                                 % (key, np.size(val), N))
            eqs += b2.Equations('%s : %s (constant)' % (key, unit_string(val)))
            heterogeneous[key] = val
        else:
            namespace[key] = val

    G = b2.NeuronGroup(N, eqs, method=method, name=name,
                       namespace=namespace, **neuron, **group_kwargs)

    for key, val in heterogeneous.items():
        setattr(G, key, val)
    G.V = 'EL'  # at rest, conductances and adaptation at 0

    return G
