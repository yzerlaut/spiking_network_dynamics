"""
Layer 2 -- synaptic projections.

`projection` returns a plain `brian2.Synapses` targeting one channel of the
postsynaptic population, it is equivalent to:

    S = b2.Synapses(Exc, Inh, model='w : siemens',
                    on_pre='g_AMPA_post += w', name='Exc_to_Inh')
    S.connect(i='k for k in sample(N_pre, size=200)')   # fixed in-degree
    S.w = 1*nS

Connectivity rules (one of them):
    indegree=K   : each postsynaptic neuron receives exactly K synapses
                   (distinct presynaptic neurons, no autapse)
    p=0.05       : brian2 connection probability (binomial in-degree)
    i=..., j=... : explicit pairs
    connect=dict(...) : passed as is to `S.connect`
"""
import brian2 as b2

from . import equations


def channels_of(group):
    """ names of the synaptic channels of a population (from its variables) """
    return sorted(v[2:] for v in group.variables
                  if v.startswith('I_') and v not in ['I_syn'])


def input_variable(group, channel):
    """ variable incremented by presynaptic events on `channel`, e.g. 'g_AMPA' """
    for kind in ['alpha', 'exp']:  # 'x_' (alpha, biexp) before 'g_' (exp)
        var = equations.channel_input(channel, kind)
        if var in group.variables:
            return var
    raise ValueError("population '%s' has no channel '%s' (channels: %s)" %
                     (group.name, channel, channels_of(group)))


def projection(pre, post, channel, w,
               indegree=None, p=None, i=None, j=None, connect=None,
               p_release=None,
               delay=None,
               autapses=False,
               name=None,
               model='w : siemens'):
    """
    pre, post  : brian2 groups (NeuronGroup, PoissonGroup, SpikeGeneratorGroup, ...)
    channel    : channel of `post` receiving the events (e.g. 'AMPA')
    w          : synaptic weight (quantity, array, or string expression,
                 e.g. '(1 + 0.2*randn())*nS')
    p_release  : release probability (each event transmitted with proba p_release)
    delay      : synaptic delay (quantity, array or string expression)
    autapses   : allow i==j connections when pre is post
    name       : default '<pre>_to_<post>'
    """
    on_pre = '%s_post += w' % input_variable(post, channel)
    namespace = {}
    if p_release is not None:
        on_pre = '%s_post += w*int(rand() < p_release)' % input_variable(post, channel)
        namespace['p_release'] = p_release

    S = b2.Synapses(pre, post, model=model, on_pre=on_pre,
                    namespace=namespace,
                    name=name or '%s_to_%s' % (pre.name, post.name))

    recurrent = (pre is post) and not autapses
    if connect is not None:
        S.connect(**connect)
    elif indegree is not None:
        if recurrent:  # sample among the N_pre-1 other neurons
            S.connect(i='k + int(k >= j) for k in sample(N_pre - 1, size=%i)'
                      % indegree)
        else:
            S.connect(i='k for k in sample(N_pre, size=%i)' % indegree)
    elif p is not None:
        S.connect(condition='i != j' if recurrent else None, p=p)
    elif i is not None:
        S.connect(i=i, j=j)
    else:
        raise ValueError("projection '%s': no connectivity rule "
                         "(indegree, p, i/j or connect)" % S.name)

    S.w = w
    if delay is not None:
        S.delay = delay

    return S
