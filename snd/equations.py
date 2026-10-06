"""
Layer 1 -- equation blocks.

Plain brian2 model strings with *named* parameters, turned into
`brian2.Equations`. No parameter value is inserted here: values live in the
`namespace` of the group that uses the equations (see `snd.cells`).

Membrane parameter names follow the brian2 AdEx example
(https://brian2.readthedocs.io/en/stable/examples/frompapers.Brette_Gerstner_2005.html):

    C, gL, EL        membrane capacitance, leak conductance, leak reversal
    VT, DeltaT       threshold and slope factor of the exponential term
    Vcut, Vr, tref   spike detection level, reset level, refractory period
    a, b, tauw       subthreshold and spike-triggered adaptation (of the
                     current `w_adapt`, `w` is kept for synaptic weights)

Synaptic channels are written once, with generic names (g, I, E, tau, ...),
and renamed per receptor with brian2's own substitution mechanism:

    brian2.Equations(CHANNEL['exp'], g='g_AMPA', I='I_AMPA', E='E_AMPA', tau='tau_AMPA')
"""
import numpy as np
import brian2 as b2

# ------------------------------------------------------------------ #
#    membrane models
# ------------------------------------------------------------------ #

MEMBRANE = {

    'LIF': '''
dV/dt = (gL*(EL - V) + I_syn + I0)/C : volt (unless refractory)
I0 : amp
''',

    'EIF': '''
dV/dt = (gL*(EL - V) + gL*DeltaT*exp((V - VT)/DeltaT) + I_syn + I0)/C : volt (unless refractory)
I0 : amp
''',

    'AdLIF': '''
dV/dt = (gL*(EL - V) + I_syn + I0 - w_adapt)/C : volt (unless refractory)
dw_adapt/dt = (a*(V - EL) - w_adapt)/tauw : amp
I0 : amp
''',

    'AdExp': '''
dV/dt = (gL*(EL - V) + gL*DeltaT*exp((V - VT)/DeltaT) + I_syn + I0 - w_adapt)/C : volt (unless refractory)
dw_adapt/dt = (a*(V - EL) - w_adapt)/tauw : amp
I0 : amp
''',
}

THRESHOLD = {'LIF': 'V > VT', 'AdLIF': 'V > VT',
             'EIF': 'V > Vcut', 'AdExp': 'V > Vcut'}

RESET = {'LIF': 'V = Vr', 'EIF': 'V = Vr',
         'AdLIF': 'V = Vr; w_adapt += b', 'AdExp': 'V = Vr; w_adapt += b'}

PARAMETERS = {'LIF': ['C', 'gL', 'EL', 'VT', 'Vr', 'tref'],
              'EIF': ['C', 'gL', 'EL', 'VT', 'DeltaT', 'Vcut', 'Vr', 'tref'],
              'AdLIF': ['C', 'gL', 'EL', 'VT', 'Vr', 'tref', 'a', 'b', 'tauw'],
              'AdExp': ['C', 'gL', 'EL', 'VT', 'DeltaT', 'Vcut', 'Vr', 'tref',
                        'a', 'b', 'tauw']}


def neuron(kind='AdExp'):
    """
    keyword arguments of `brian2.NeuronGroup` for a given membrane model:

        b2.NeuronGroup(N, **neuron('AdExp'), namespace=params)

    N.B. `I_syn` is left undefined, see `synaptic_current`
    """
    return dict(model=b2.Equations(MEMBRANE[kind]),
                threshold=THRESHOLD[kind],
                reset=RESET[kind],
                refractory='tref')


# ------------------------------------------------------------------ #
#    synaptic channels (one per receptor type)
# ------------------------------------------------------------------ #

CHANNEL = {

    # instantaneous rise, exponential decay
    'exp': '''
dg/dt = -g/tau : siemens
I = g*(E - V) : amp
''',

    # alpha function, peak (of value w) at t=tau
    'alpha': '''
dg/dt = (e*x - g)/tau : siemens
dx/dt = -x/tau : siemens
I = g*(E - V) : amp
''',

    # difference of exponentials, normalized so that the peak has value w
    'biexp': '''
dg/dt = (norm*x - g)/tau_decay : siemens
dx/dt = -x/tau_rise : siemens
I = g*(E - V) : amp
''',
}

# names local to each channel block (to be suffixed by the channel name)
CHANNEL_NAMES = {'exp': ['g', 'I', 'E', 'tau'],
                 'alpha': ['g', 'x', 'I', 'E', 'tau'],
                 'biexp': ['g', 'x', 'I', 'E', 'tau_rise', 'tau_decay', 'norm']}

# parameters that have to be provided by the user
CHANNEL_PARAMETERS = {'exp': ['E', 'tau'],
                      'alpha': ['E', 'tau'],
                      'biexp': ['E', 'tau_rise', 'tau_decay']}

# variable incremented by a presynaptic event (on_pre = '<var>_post += w')
CHANNEL_INPUT = {'exp': 'g', 'alpha': 'x', 'biexp': 'x'}


def channel(name, kind='exp'):
    """
    equations of the channel `name`, e.g. channel('AMPA') gives:

        dg_AMPA/dt = -g_AMPA/tau_AMPA : siemens
        I_AMPA = g_AMPA*(E_AMPA - V) : amp
    """
    return b2.Equations(CHANNEL[kind],
                        **{n: '%s_%s' % (n, name) for n in CHANNEL_NAMES[kind]})


def channel_input(name, kind='exp'):
    """ variable incremented by presynaptic events, e.g. 'g_AMPA' """
    return '%s_%s' % (CHANNEL_INPUT[kind], name)


def biexp_norm(tau_rise, tau_decay):
    """ factor so that the peak of the 'biexp' channel equals the increment """
    tr, td = float(tau_rise/b2.second), float(tau_decay/b2.second)
    tpeak = tr*td/(td-tr)*np.log(td/tr)
    return (td-tr)/tr/(np.exp(-tpeak/td)-np.exp(-tpeak/tr))


def channel_namespace(name, kind='exp', **params):
    """
    namespace entries of a channel, e.g. {'E_AMPA': 0*mV, 'tau_AMPA': 5*ms}
    """
    missing = set(CHANNEL_PARAMETERS[kind])-set(params)
    unknown = set(params)-set(CHANNEL_PARAMETERS[kind])
    if missing or unknown:
        raise ValueError("channel '%s' (kind '%s') expects %s, got %s" %
                         (name, kind, CHANNEL_PARAMETERS[kind], list(params)))
    namespace = {'%s_%s' % (key, name): val for key, val in params.items()}
    if kind == 'biexp':
        namespace['norm_%s' % name] = biexp_norm(params['tau_rise'],
                                                 params['tau_decay'])
    return namespace


def synaptic_current(channel_names):
    """ I_syn = I_AMPA + I_GABA + ... : amp """
    if len(channel_names) == 0:
        return b2.Equations('I_syn = 0*amp : amp')
    return b2.Equations('I_syn = %s : amp' %
                        ' + '.join('I_%s' % c for c in channel_names))
