"""
Layer 2 -- monitors.

`monitors(G, ...)` returns a list of plain brian2 monitors, named after the
group:

    b2.SpikeMonitor(G, name='Exc_spikes')
    b2.PopulationRateMonitor(G, name='Exc_rate')
    b2.StateMonitor(G, ['V', 'w'], record=range(4), name='Exc_state')
"""
import numpy as np
import brian2 as b2


def monitors(group,
             spikes=True,
             rate=False,
             variables=None,
             record=4,
             dt=None):
    """
    spikes    : SpikeMonitor
    rate      : PopulationRateMonitor
    variables : list of variables for a StateMonitor (e.g. ['V', 'g_AMPA'])
    record    : neurons of the StateMonitor: a number n (the first n
                neurons), a list of indices, or True (all)
    dt        : sampling step of the StateMonitor (default: simulation dt)
    """
    out = []
    if spikes:
        out.append(b2.SpikeMonitor(group, name='%s_spikes' % group.name))
    if rate:
        out.append(b2.PopulationRateMonitor(group, name='%s_rate' % group.name))
    if variables:
        if isinstance(record, (int, np.integer)) and not isinstance(record, bool):
            record = np.arange(min(record, len(group)))
        out.append(b2.StateMonitor(group, list(variables), record=record, dt=dt,
                                   name='%s_state' % group.name))
    return out
