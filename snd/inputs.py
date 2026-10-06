"""
Layer 2 -- external inputs.

Two kinds of afferent input:

 - `poisson_drive`: each target neuron receives N independent Poisson
   synapses (no explicit presynaptic population, no shared input). It is an
   operation on the target group (as `brian2.PoissonInput`, but named and
   with possibly time-varying rates):

        G.run_regularly('g_AMPA += w*poisson(N*rate(t)*dt)', when='synapses')

 - `poisson_population`: an explicit population of Poisson neurons
   (`brian2.PoissonGroup`), to be connected with `synapses.projection`
   (shared input, recordable spikes).

Explicit spike times: use `brian2.SpikeGeneratorGroup` directly.

Time-varying rates are given either as a `brian2.TimedArray` or as an array
of rates sampled every `dt`.
"""
import numpy as np
import brian2 as b2

from .synapses import input_variable


def timed_array(rate, dt):
    """ brian2.TimedArray from a rate array sampled every dt """
    if isinstance(rate, b2.TimedArray):
        return rate
    if dt is None:
        raise ValueError('a time-varying rate needs its sampling step `dt`')
    return b2.TimedArray(rate, dt=dt)


def poisson_drive(target, channel, N, rate, w,
                  p_release=1.,
                  dt=None,
                  name=None):
    """
    target    : postsynaptic NeuronGroup
    channel   : channel of `target` receiving the events (e.g. 'AMPA')
    N         : number of afferent synapses per target neuron
    rate      : rate per synapse: quantity (constant), TimedArray or array (with dt)
    w         : synaptic weight (quantity)
    p_release : release probability (thins the Poisson process)
    """
    var = input_variable(target, channel)
    name = name or 'drive_%s_%s' % (channel, target.name)

    # the drive parameters go to the target namespace, prefixed by the name
    # of the operation
    if (not isinstance(rate, b2.TimedArray)) and (np.size(rate) == 1):
        rate_code, rate_value = '{n}_rate', rate
    else:
        rate_code, rate_value = '{n}_rate(t)', timed_array(rate, dt)
    if target.namespace is None:
        target.namespace = {}
    target.namespace.update({'%s_rate' % name: rate_value,
                             '%s_N' % name: N*p_release,
                             '%s_w' % name: w})
    code = ('{var} += {n}_w*poisson({n}_N*%s*dt)' % rate_code).format(var=var, n=name)
    return target.run_regularly(code, when='synapses', name=name)


def poisson_population(name, N, rate, dt=None):
    """
    brian2.PoissonGroup with a constant or time-varying rate
    """
    if (not isinstance(rate, b2.TimedArray)) and (np.size(rate) == 1):
        return b2.PoissonGroup(N, rates=rate, name=name)
    return b2.PoissonGroup(N, rates='rate(t)', name=name,
                           namespace={'rate': timed_array(rate, dt)})
