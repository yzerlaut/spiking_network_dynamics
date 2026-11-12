"""
Plotting of simulation data (the dictionary returned by `snd.run` or
`snd.io.load`).

Functions take matplotlib axes as arguments (matplotlib is not imported here).

    fig, AX = plt.subplots(3, 1, sharex=True)
    snd.plot.membrane_potential_subplots(data, AX, ['Exc', 'Inh', 'DsInh'],
                                         spike_peak=-10*mV, clip=True)
"""
import numpy as np
from brian2 import ms, mV

from .cells import DEFAULTS


def thin_spikes(i, t, dead_time=100*ms, random=False, seed=None,
                subsampling=1):
    """
    for raster plots at high rates: removes, for each neuron, the spikes
    falling closer than `dead_time` to a kept spike

    i, t      : spike indices and times (as in a SpikeMonitor)
    random    : False: spikes scanned in time order (each kept spike removes
                the spikes in the next `dead_time`)
                True : spikes picked in random order (each picked spike, if
                not yet removed, is kept and removes the spikes within
                `dead_time` before and after it), no bias in time
    seed      : seed of the random order
    subsampling : keep only the spikes of 1 neuron over `subsampling`
                  (neurons with index i % subsampling == 0)

    returns the boolean mask of the kept spikes
    """
    i, t = np.asarray(i), np.asarray(t/ms)
    dead_time = float(dead_time/ms)
    rng = np.random.default_rng(seed)
    keep = np.zeros(len(t), dtype=bool)
    order = np.lexsort((t, i))  # by neuron, then by time
    order = order[i[order] % subsampling == 0]
    if len(order) == 0:
        return keep
    bounds = np.flatnonzero(np.diff(i[order]))+1
    for spikes in np.split(order, bounds):  # spikes of one neuron, in time order
        ts = t[spikes]
        removed = np.zeros(len(ts), dtype=bool)
        picks = rng.permutation(len(ts)) if random else range(len(ts))
        for k in picks:
            if not removed[k]:
                keep[spikes[k]] = True
                lo = np.searchsorted(ts, ts[k]-dead_time, side='right') \
                    if random else k
                hi = np.searchsorted(ts, ts[k]+dead_time, side='left')
                removed[lo:hi] = True
    return keep


def vm_trace(data, pop, k=0, spike_peak=None, clip=None):
    """
    membrane potential of the k-th recorded neuron of `pop`

    spike_peak : None, or a voltage: spikes are returned as segments from V at
                 the spike time up to `spike_peak`
    clip       : None, a duration (V is masked over (tspike, tspike+clip)), or
                 True (clip over the refractory period `tref` of `pop`)

    returns t (ms), V (mV) and the list of spike segments
    [(tspike, V_at_spike, peak), ...] (empty if spike_peak is None), unitless
    """
    state = data['%s_state' % pop]
    t, V = np.asarray(state['t']/ms), np.array(state['V'][k]/mV)
    neuron = state['record'][k]

    segments = []
    if spike_peak is None and clip is None:
        return t, V, segments

    if '%s_spikes' % pop not in data:
        raise KeyError('spikes of %s not recorded, needed for `spike_peak` '
                       'and `clip`' % pop)
    sp = data['%s_spikes' % pop]
    tspikes = np.asarray(sp['t'][sp['i'] == neuron]/ms)
    tspikes = tspikes[(tspikes >= t[0]) & (tspikes <= t[-1])]

    if clip is True:
        clip = data.get('model', {}).get('populations', {}).get(pop, {})\
            .get('params', {}).get('tref', DEFAULTS['tref'])
    if clip is not None:
        clip = float(clip/ms)
        for ts in tspikes:
            V[(t > ts+1e-6) & (t < ts+clip-1e-6)] = np.nan

    if spike_peak is not None:
        # last sample before the spike (monitors record before reset)
        idx = np.searchsorted(t, tspikes+1e-6, side='right')-1
        segments = [(ts, V[i], float(spike_peak/mV))
                    for ts, i in zip(tspikes, idx)]

    return t, V, segments


def membrane_potential_subplots(data, AX, pops,
                                colors=None,
                                neurons=None,
                                tzoom=None,
                                spike_peak=None,
                                spike_style='--',
                                clip=None,
                                lw=1,
                                spike_lw=None,
                                subsampling=1,
                                xlabel=True):
    """
    membrane potential traces of the recorded neurons, one axis per population

    AX          : list of axes (same length as `pops`)
    pops        : population names (with a recorded 'V' variable)
    colors      : list of colors (default: 'C0', 'C1', ...)
    neurons     : indices (in the recorded neurons) to plot, the same for all
                  populations, or a dict {pop: indices} (default: all)
    tzoom       : (tmin, tmax) quantities (default: full trace)
    spike_peak  : None, or a voltage: spikes drawn as a line up to this value
    spike_style : linestyle of spikes ('--' dashed, '-' plain)
    clip        : None, a duration (V masked over (tspike, tspike+clip)), or
                  True (clip over the refractory period of each population)
    lw          : linewidth of traces (and of spikes if `spike_lw` is None)
    subsampling : plot one sample out of `subsampling`
    xlabel      : add the time label on the last axis
    """
    if colors is None:
        colors = ['C%i' % i for i in range(len(pops))]
    if spike_lw is None:
        spike_lw = lw

    for i, (pop, ax) in enumerate(zip(pops, AX)):

        if '%s_state' % pop not in data or 'V' not in data['%s_state' % pop]:
            ax.annotate('V of %s not recorded' % pop, (.5, .5),
                        xycoords='axes fraction', color=colors[i],
                        ha='center', va='center')
            continue

        ks = neurons.get(pop) if isinstance(neurons, dict) else neurons
        if ks is None:
            ks = range(len(data['%s_state' % pop]['record']))

        for k in np.atleast_1d(ks):
            t, V, segments = vm_trace(data, pop, k,
                                      spike_peak=spike_peak, clip=clip)
            cond = np.ones(len(t), dtype=bool)
            if tzoom is not None:
                cond = (t >= float(tzoom[0]/ms)) & (t <= float(tzoom[1]/ms))
            ax.plot(t[cond][::subsampling], V[cond][::subsampling],
                    '-', lw=lw, color=colors[i])
            for ts, v0, peak in segments:
                if tzoom is None or \
                        float(tzoom[0]/ms) <= ts <= float(tzoom[1]/ms):
                    ax.plot([ts, ts], [v0, peak], linestyle=spike_style,
                            lw=spike_lw, color=colors[i])

        ax.annotate(' %s' % pop, (1., .5), xycoords='axes fraction',
                    color=colors[i], va='center')
        ax.set_ylabel('V (mV)')
        if tzoom is not None:
            ax.set_xlim(float(tzoom[0]/ms), float(tzoom[1]/ms))

    if xlabel:
        AX[len(pops)-1].set_xlabel('time (ms)')

    return AX


def stacked_membrane_potentials(data, ax, traces,
                                shift=20*mV,
                                tzoom=None,
                                spike_peak=None,
                                spike_style='--',
                                clip=None,
                                lw=1,
                                spike_lw=None,
                                subsampling=1,
                                Vbar=10*mV,
                                Tbar=None,
                                bar_loc='bottom left',
                                labels=True):
    """
    membrane potential traces on a single axis, shifted vertically (first
    trace on top), with scale bars instead of axes (see `scale_bars`)

    traces : list of (pop, k, color), k the index in the recorded neurons
    shift  : vertical spacing between traces
    Vbar   : size of the vertical scale bar (None: no bar)
    Tbar   : size of the horizontal (time) scale bar, replacing the time axis
             (None: keep the time axis)
    bar_loc: location of the scale bars, 'bottom left' or 'top left'
    labels : write the population name at the right of each trace
    (other arguments as in `membrane_potential_subplots`)
    """
    if spike_lw is None:
        spike_lw = lw
    shift = float(shift/mV)
    t0, t1 = np.inf, -np.inf

    for n, (pop, k, color) in enumerate(traces):
        offset = (len(traces)-1-n)*shift
        t, V, segments = vm_trace(data, pop, k,
                                  spike_peak=spike_peak, clip=clip)
        cond = np.ones(len(t), dtype=bool)
        if tzoom is not None:
            cond = (t >= float(tzoom[0]/ms)) & (t <= float(tzoom[1]/ms))
        t0, t1 = min(t0, t[cond][0]), max(t1, t[cond][-1])
        ax.plot(t[cond][::subsampling], V[cond][::subsampling]+offset,
                '-', lw=lw, color=color)
        for ts, v0, peak in segments:
            if t[cond][0] <= ts <= t[cond][-1]:
                ax.plot([ts, ts], [v0+offset, peak+offset],
                        linestyle=spike_style, lw=spike_lw, color=color)
        if labels:
            ax.annotate(' %s' % pop, (t[cond][-1], np.nanmedian(V[cond])+offset),
                        color=color, va='center', annotation_clip=False)

    ax.set_xlim(t0, t1)
    scale_bars(ax, Vbar=Vbar, Tbar=Tbar, loc=bar_loc)
    if Tbar is None:
        ax.set_xlabel('time (ms)')

    return ax


def scale_bars(ax, Vbar=10*mV, Tbar=None, Vunit=mV, loc='bottom left'):
    """
    replace the y-axis (and the x-axis if `Tbar` is given) of `ax` by scale
    bars at the left corners (axis units: `Vunit` and ms)

    Vbar  : size of the vertical bar (None: no bar), e.g. 10*mV or 1*Hz
    Tbar  : size of the horizontal (time) bar (None: keep the time axis)
    Vunit : unit of the y-axis data (e.g. mV, Hz)
    loc   : 'bottom left' or 'top left'
    """
    for side in ['left', 'right', 'top']:
        ax.spines[side].set_visible(False)
    ax.set_yticks([])
    if Tbar is not None:
        ax.spines['bottom'].set_visible(False)
        # (no set_xticks([]), that would also remove the ticks of shared axes)
        ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)

    (t0, t1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    if loc not in ['bottom left', 'top left']:
        raise ValueError("loc should be 'bottom left' or 'top left'")
    top = (loc == 'top left')
    xbar = t0-0.01*(t1-t0)
    ybar = y1 if top else y0  # corner of the bars
    if Vbar is not None:
        Vbar = float(Vbar/Vunit)
        yV = ybar-Vbar if top else ybar  # bottom of the vertical bar
        ax.plot([xbar, xbar], [yV, yV+Vbar], 'k-', lw=1.5, clip_on=False)
        ax.annotate('%g%s ' % (Vbar, Vunit), (xbar, yV+Vbar/2), ha='right', va='center',
                    annotation_clip=False)
    if Tbar is not None:
        Tbar = float(Tbar/ms)
        ax.plot([xbar, xbar+Tbar], [ybar, ybar], 'k-', lw=1.5, clip_on=False)
        ax.annotate('%gms' % Tbar, (xbar+Tbar/2, ybar),
                    ha='center', va='bottom' if top else 'top',
                    xytext=(0, 3 if top else -3), textcoords='offset points',
                    annotation_clip=False)
    ax.set_xlim(t0, t1)
    ax.set_ylim(y0, y1)
