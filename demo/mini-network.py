"""
Toy network of 7 neurons (cells 1-5 excitatory, cells 6-7 inhibitory) with an
explicit, hand-written connectivity. Each cell is made to fire a single spike
by a current step, one after the other, and the postsynaptic potentials show
the connectivity.

    python demo/mini-network.py         # run, save to data/mini-network.h5
    python demo/mini-network.py plot    # plot

Connectivity (cell numbers, 1-5: Exc[0-4], 6-7: Inh[0-1]):
    1 -> 2, 4, 5       2 -> 3, 4       3 -> 1, 7
    4 -> 6             5 -> 6, 7
    6 -| 4, 5          7 -| 3
"""
import sys, pathlib
import numpy as np
import brian2 as b2
from brian2 import ms, mV, nS, pF, pA

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # repo root, for `snd`
import snd

params = dict(gL=10*nS, C=200*pF, EL=-70*mV, VT=-50*mV, Vr=-70*mV, tref=5*ms)

Model = {
    'dt': 0.1*ms, 'tstop': 875*ms, 'seed': 1,

    'channels': {'AMPA': dict(kind='exp', E=0*mV, tau=3*ms),
                 'GABA': dict(kind='exp', E=-80*mV, tau=10*ms)},

    'populations': {'Exc': dict(N=5, kind='LIF', params=params),
                    'Inh': dict(N=2, kind='LIF', params=params)},

    # explicit pairs (i: presynaptic, j: postsynaptic), 0-based indices
    'projections': {
        ('Exc', 'Exc'): dict(channel='AMPA', w=20*nS,
                             i=[0, 0, 0, 1, 1, 2], j=[1, 3, 4, 2, 3, 0]),
        ('Exc', 'Inh'): dict(channel='AMPA', w=20*nS,
                             i=[2, 3, 4, 4], j=[1, 0, 0, 1]),
        ('Inh', 'Exc'): dict(channel='GABA', w=40*nS,
                             i=[1, 0, 0], j=[2, 3, 4]),
    },

    'record': {'Exc': dict(spikes=True, variables=['V'], record=5),
               'Inh': dict(spikes=True, variables=['V'], record=2)},

    # current steps, onset per neuron (firing order: 1, 7, 2, 6, 3, 5, 4)
    'stim': dict(amplitude=300*pA, duration=25*ms,
                 onsets=dict(Exc=[125, 320, 515, 710, 610]*ms,
                             Inh=[415, 220]*ms)),
}

# cell number (as in the figure) -> (pop, index)
CELLS = [('Exc', k) for k in range(5)] + [('Inh', k) for k in range(2)]


def stim_current(Model, pop):
    """ current steps of `pop` as a TimedArray I(t, i) """
    stim, dt = Model['stim'], Model['dt']
    nt = int(Model['tstop']/dt)+1
    I = np.zeros((nt, Model['populations'][pop]['N']))
    t = np.arange(nt)*dt
    for k, onset in enumerate(stim['onsets'][pop]):
        I[(t >= onset) & (t < onset+stim['duration']), k] = stim['amplitude']/pA
    return b2.TimedArray(I*pA, dt=dt, name='I_stim_%s' % pop)


if sys.argv[-1] == 'plot':

    import matplotlib.pyplot as plt
    data = snd.io.load('data/mini-network.h5')
    stim = data['model']['stim']
    colors = {'Exc': 'tab:green', 'Inh': 'tab:red'}
    shift = 25*mV

    fig, ax = plt.subplots(figsize=(8.1, 3.5))
    snd.plot.stacked_membrane_potentials(
        data, ax, [(pop, k, colors[pop]) for pop, k in CELLS],
        shift=shift, spike_peak=0*mV, spike_style='-', clip=True,
        lw=1.5, Vbar=20*mV, Tbar=50*ms, bar_loc='top left', labels=False)

    for n, (pop, k) in enumerate(CELLS):
        offset = float((len(CELLS)-1-n)*shift/mV)
        rest = float(params['EL']/mV)+offset
        ax.annotate('%i ' % (n+1), (0, rest), ha='right', va='center')
        ax.plot([0, float(Model['tstop']/ms)], [rest, rest], ':',
                color=colors[pop], lw=1)
        t0 = float(stim['onsets'][pop][k]/ms)  # current step, under the trace
        ax.plot([t0, t0+float(stim['duration']/ms)], [rest-4]*2,
                'k-', lw=4, solid_capstyle='butt')
    ax.set_title('membrane potential dynamics (black bars: current steps)',
                 fontsize=9)
    plt.show()

else:

    net = snd.build(Model)
    for pop in ['Exc', 'Inh']:
        I_stim = stim_current(Model, pop)
        net[pop].run_regularly('I0 = %s(t, i)' % I_stim.name,
                               dt=Model['dt'], when='start')
        net[pop].namespace[I_stim.name] = I_stim
    net.run(Model['tstop'])
    data = snd.io.collect(net)
    data['model'] = Model
    snd.io.save('data/mini-network.h5', data)

    print(' --- spike times (ms) ---')
    for n, (pop, k) in enumerate(CELLS):
        sp = data['%s_spikes' % pop]
        print('cell %i (%s %i):' % (n+1, pop, k), np.round(sp['t'][sp['i'] == k]/ms, 1))
