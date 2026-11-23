"""
Mini network of 7 cells illustrating, in four phases, what spiking network
models capture:

  1. cellular diversity   : current steps in a regular-spiking (RS, adapting),
                            a fast-spiking (FS) and an intrinsically bursting
                            (IB) cell, one after the other
  2. synaptic integration : a triplet of presynaptic spikes onto a pyramidal
                            cell (Pyr) from a cell with AMPA-only synapses
                            (subthreshold), then from a cell with AMPA+NMDA
                            synapses (the AMPA depolarization relieves the Mg2+
                            block of NMDA: supralinear summation -> spike)
  3. inhibition           : an interneuron alone (small IPSP at rest, near
                            E_GABA), then vetoing the AMPA+NMDA spike
                            (conductance-based: strong when depolarized)
  4. background activity  : excitatory/inhibitory Poisson synaptic
                            bombardment, in vivo-like fluctuating Vm and
                            irregular firing

    python demo/mini-network.py         # run, save to data/mini-network.h5
    python demo/mini-network.py plot    # plot

Connectivity (the 7 cells are wired throughout, every spike gives PSPs):

    PreA  -> PreAN, Pyr, IB  (AMPA)       PreAN -> PreA, RS, Pyr  (AMPA+NMDA)
    RS    -> PreA, FS        (AMPA)       Pyr   -> Inh      (AMPA)
    IB    -> Inh, FS         (AMPA)
    Inh   -| Pyr, IB         (GABA)       FS    -| RS, Pyr  (GABA)

Each cell is its own population (N=1), with its own membrane model.
"""
import sys, pathlib
import numpy as np
import brian2 as b2
from brian2 import ms, mV, nS, pF, pA, Hz, mM

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # repo root, for `snd`
import snd

membrane = dict(gL=10*nS, C=200*pF, EL=-70*mV, VT=-50*mV, tref=2*ms)

# phases (for the plot): name, start, end
PHASES = [('1. cellular\ndiversity', 0*ms, 700*ms),
          ('2. synaptic\nintegration', 700*ms, 1100*ms),
          ('3. inhibition', 1100*ms, 1450*ms),
          ('4. background\nactivity', 1450*ms, 1950*ms)]

def triplets(*starts, duration=60*ms, amplitude=340*pA):
    """ one current step per start time, making a LIF cell fire 3 spikes
    (20ms apart: 18ms to threshold + 2ms refractory) """
    onset = [float(start/ms) for start in starts]
    return dict(onset=onset*ms, duration=[float(duration/ms)]*len(onset)*ms,
                amplitude=[float(amplitude/pA)]*len(onset)*pA)


def step(onset, duration, amplitude=400*pA):
    """ one current step """
    return dict(onset=[float(onset/ms)]*ms, duration=[float(duration/ms)]*ms,
                amplitude=[float(amplitude/pA)]*pA)


def connect(channel, w):
    """ projection between two single-cell populations """
    return dict(channel=channel, w=w, i=[0], j=[0])


Model = {
    'dt': 0.1*ms, 'tstop': 1950*ms, 'seed': 3,

    'channels': {'AMPA': dict(kind='exp', E=0*mV, tau=3*ms),
                 'NMDA': dict(kind='nmda', E=0*mV, Mg=1*mM,
                              tau_rise=2*ms, tau_decay=80*ms),
                 'GABA': dict(kind='exp', E=-80*mV, tau=10*ms)},

    'populations': {
        # excitatory
        'RS': dict(N=1, kind='AdExp',
                   params=dict(membrane, DeltaT=2*mV, Vr=-65*mV,
                               a=4*nS, b=80*pA, tauw=150*ms)),
        'IB': dict(N=1, kind='AdExp',
                   params=dict(membrane, DeltaT=2*mV, Vr=-46*mV,
                               a=2*nS, b=80*pA, tauw=80*ms)),
        'PreA': dict(N=1, kind='LIF', params=dict(membrane, Vr=-70*mV)),
        'PreAN': dict(N=1, kind='LIF', params=dict(membrane, Vr=-70*mV)),
        'Pyr': dict(N=1, kind='EIF',
                    params=dict(membrane, DeltaT=2*mV, Vr=-65*mV, tref=5*ms)),
        # inhibitory (fast-spiking)
        'FS': dict(N=1, kind='EIF',
                   params=dict(membrane, DeltaT=0.5*mV, Vr=-65*mV)),
        'Inh': dict(N=1, kind='EIF',
                    params=dict(membrane, DeltaT=0.5*mV, Vr=-65*mV)),
    },

    'projections': {
        # onto Pyr: the circuit of phases 2-3
        ('PreA', 'Pyr'): connect('AMPA', 10*nS),
        ('PreAN', 'Pyr'): connect('AMPA', 10*nS),
        ('PreAN', 'Pyr', 'NMDA'): connect('NMDA', 13*nS),
        ('Inh', 'Pyr'): connect('GABA', 15*nS),
        # the rest of the circuit: visible, subthreshold PSPs
        ('PreA', 'PreAN'): connect('AMPA', 4*nS),
        ('PreA', 'IB'): connect('AMPA', 4*nS),
        ('PreAN', 'RS'): connect('AMPA', 4*nS),
        ('PreAN', 'RS', 'NMDA'): connect('NMDA', 4*nS),
        ('PreAN', 'PreA'): connect('AMPA', 4*nS),
        ('PreAN', 'PreA', 'NMDA'): connect('NMDA', 4*nS),
        ('RS', 'PreA'): connect('AMPA', 4*nS),
        ('RS', 'FS'): connect('AMPA', 4*nS),
        ('IB', 'Inh'): connect('AMPA', 2*nS),
        ('IB', 'FS'): connect('AMPA', 2*nS),
        ('Pyr', 'Inh'): connect('AMPA', 4*nS),
        ('Inh', 'IB'): connect('GABA', 10*nS),
        ('FS', 'RS'): connect('GABA', 10*nS),
        ('FS', 'Pyr'): connect('GABA', 10*nS),
    },

    # phase 4: background synaptic bombardment (rates set below)
    'afferents': {'BgExc': dict(N=1), 'BgInh': dict(N=1)},

    # current steps per population: onsets, durations, amplitudes
    'stim': {
        'RS': step(50*ms, 180*ms),
        'FS': step(260*ms, 180*ms),
        'IB': step(470*ms, 180*ms),
        'PreA': triplets(760*ms),
        'PreAN': triplets(920*ms, 1300*ms),
        'Inh': dict(onset=[1150, 1290]*ms, duration=[50, 80]*ms,
                    amplitude=[400, 400]*pA),
    },
}

POPS = list(Model['populations'])
EXC, INH = ['RS', 'IB', 'PreA', 'PreAN', 'Pyr'], ['FS', 'Inh']
ORDER = ['Pyr', 'IB', 'RS', 'PreAN', 'PreA', 'FS', 'Inh']  # traces, top to bottom

# background: excitatory and inhibitory Poisson drive onto every cell, on in phase 4
rate_dt = 1*ms
t_rate = np.arange(int(Model['tstop']/rate_dt))*rate_dt
on = np.clip((t_rate-PHASES[3][1]-20*ms)/(100*ms), 0, 1)  # 100ms ramp
Model['afferents']['BgExc'].update(rate=on*650*Hz, rate_dt=rate_dt)
Model['afferents']['BgInh'].update(rate=on*300*Hz, rate_dt=rate_dt)
for pop in POPS:
    # (Pyr: half, the rest comes from its inputs PreA, PreAN, Inh)
    Model['projections'][('BgExc', pop)] = dict(channel='AMPA', indegree=1,
                                                w=(1.5 if pop == 'Pyr' else 3)*nS,
                                                name='BgExc_to_%s' % pop)
    Model['projections'][('BgInh', pop)] = dict(channel='GABA', w=3*nS, indegree=1,
                                                name='BgInh_to_%s' % pop)
Model['record'] = {pop: dict(spikes=True, variables=['V'], record=1) for pop in POPS}


def stim_current(Model, pop):
    """ current steps of `pop` as a TimedArray I(t) """
    dt = Model['dt']
    t = np.arange(int(Model['tstop']/dt)+1)*dt
    I = np.zeros(len(t))
    stim = Model['stim'].get(pop)
    if stim is not None:
        for onset, duration, amp in zip(stim['onset'], stim['duration'],
                                        stim['amplitude']):
            I[(t >= onset) & (t < onset+duration)] += float(amp/pA)
    return b2.TimedArray(I*pA, dt=dt, name='I_stim_%s' % pop)


if sys.argv[-1] == 'plot':

    import matplotlib.pyplot as plt
    data = snd.io.load('data/mini-network.h5')
    shift = 40*mV
    order = ORDER
    exc = [pop for pop in order if pop in EXC]  # green shades, darkest on top
    colors = dict(zip(exc, plt.cm.Greens(np.linspace(0.95, 0.65, len(exc)))))
    colors.update(zip(INH, plt.cm.Reds([0.9, 0.6])))

    fig, (ax, ax_raster) = plt.subplots(2, 1, figsize=(8.1, 6.6), sharex=True,
                                        gridspec_kw=dict(height_ratios=[5, 1.1],
                                                         hspace=0.05))
    snd.plot.stacked_membrane_potentials(
        data, ax, [(pop, 0, colors[pop]) for pop in order],
        shift=shift, spike_peak=20*mV, spike_style='-',
        lw=1, Vbar=20*mV, Tbar=100*ms, bar_loc='top right', labels=False)

    for n, pop in enumerate(order):
        offset = float((len(order)-1-n)*shift/mV)
        rest = float(membrane['EL']/mV)+offset
        stim = data['model']['stim'].get(pop)
        if stim is not None:  # current steps, under the trace
            for onset, duration in zip(stim['onset'], stim['duration']):
                t0 = float(onset/ms)
                ax.plot([t0, t0+float(duration/ms)], [rest-4]*2,
                        '-', color='0.5', lw=2.5, solid_capstyle='butt')

    # raster plot, one row per cell (same order and colors as the traces)
    for n, pop in enumerate(order):
        tspikes = data['%s_spikes' % pop]['t']/ms
        ax_raster.plot(tspikes, np.full(len(tspikes), len(order)-1-n), '|',
                       color=colors[pop], ms=5, mew=1.5)
    y0, y1 = -0.7, len(order)-0.3
    ax_raster.set_ylim(y0, y1)
    ax_raster.axis('off')

    # colorbar of the neurons, in an inset aligned with the raster rows
    from matplotlib.colors import ListedColormap, BoundaryNorm
    cmap = ListedColormap([colors[pop] for pop in order[::-1]])  # bottom row first
    norm = BoundaryNorm(np.arange(len(order)+1)-0.5, len(order))
    cax = ax_raster.inset_axes([1.01, (-0.5-y0)/(y1-y0), 0.02, len(order)/(y1-y0)])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.set_ticks([])
    cb.minorticks_off()
    cax.tick_params(which='both', length=0)
    cb.outline.set_visible(False)

    for name, t0, t1 in PHASES[1:]:  # phase separators
        for a in [ax, ax_raster]:
            a.axvline(float(t0/ms), color='0.7', lw=0.8, ls='--')
    plt.show()

else:

    net = snd.build(Model)
    for pop in POPS:
        I_stim = stim_current(Model, pop)
        net[pop].run_regularly('I0 = %s(t)' % I_stim.name,
                               dt=Model['dt'], when='start')
        net[pop].namespace[I_stim.name] = I_stim
    net.run(Model['tstop'])
    data = snd.io.collect(net)
    data['model'] = Model
    snd.io.save('data/mini-network.h5', data)

    print(' --- spike times (ms) ---')
    for pop in POPS:
        print('%5s:' % pop, np.round(data['%s_spikes' % pop]['t']/ms).astype(int))
