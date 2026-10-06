"""
Recurrent network of regular-spiking (RS, AdExp) excitatory and
fast-spiking (FS, EIF) inhibitory neurons, driven by a time-varying
afferent excitation. Port of `neural_network_dynamics/demo/RS-FS.py`.

    python demo/RS-FS.py         # run, save to data/RS-FS.h5
    python demo/RS-FS.py plot    # plot
"""
import sys, pathlib
import numpy as np
from brian2 import ms, mV, nS, pF, pA, Hz, second

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # repo root, for `snd`
import snd

Model = {
    'dt': 0.1*ms, 'tstop': 1500*ms, 'seed': 3,

    'channels': {'AMPA': dict(kind='exp', E=0*mV, tau=5*ms),
                 'GABA': dict(kind='exp', E=-80*mV, tau=5*ms)},

    'populations': {
        'RecExc': dict(N=4000, kind='AdExp',
                       params=dict(gL=10*nS, C=200*pF, tref=5*ms, EL=-70*mV,
                                   VT=-50*mV, Vr=-70*mV, DeltaT=2*mV,
                                   a=0*nS, b=40*pA, tauw=500*ms)),
        'RecInh': dict(N=1000, kind='EIF',
                       params=dict(gL=10*nS, C=200*pF, tref=5*ms, EL=-70*mV,
                                   VT=-50*mV, Vr=-70*mV, DeltaT=0.5*mV)),
    },

    'afferents': {'AffExc': dict(N=400)},  # rate set below

    'projections': {
        ('RecExc', 'RecExc'): dict(channel='AMPA', w=1*nS, indegree=200),
        ('RecExc', 'RecInh'): dict(channel='AMPA', w=1*nS, indegree=200),
        ('RecInh', 'RecExc'): dict(channel='GABA', w=4*nS, indegree=50),
        ('RecInh', 'RecInh'): dict(channel='GABA', w=4*nS, indegree=50),
        ('AffExc', 'RecExc'): dict(channel='AMPA', w=1*nS, indegree=200),
        ('AffExc', 'RecInh'): dict(channel='AMPA', w=1*nS, indegree=200),
    },

    'record': {'RecExc': dict(spikes=True, rate=True, variables=['V'], record=3),
               'RecInh': dict(spikes=True, rate=True, variables=['V'], record=3)},
}

# afferent rate: baseline + gaussian bump at t=1s
t = np.arange(int(Model['tstop']/Model['dt']))*Model['dt']
Model['afferents']['AffExc']['rate'] = (4+4*np.exp(-(t-1*second)**2/2/(100*ms)**2))*Hz


if sys.argv[-1] == 'plot':

    import matplotlib.pyplot as plt
    data = snd.io.load('data/RS-FS.h5')
    fig, AX = plt.subplots(4, 1, figsize=(6, 6), sharex=True)
    rate = data['model']['afferents']['AffExc']['rate']
    AX[0].plot(t/ms, rate/Hz, 'k')
    AX[0].set_ylabel('aff. rate (Hz)')
    for pop, c in [('RecExc', 'tab:green'), ('RecInh', 'tab:red')]:
        sp = data['%s_spikes' % pop]
        AX[1].plot(sp['t']/ms, sp['i']+(4000 if pop == 'RecInh' else 0), '.',
                   color=c, ms=1)
        r = data['%s_rate' % pop]
        kernel = np.ones(50)/50  # 5ms smoothing
        AX[2].plot(r['t']/ms, np.convolve(r['rate']/Hz, kernel, 'same'), color=c)
        AX[3].plot(data['%s_state' % pop]['t']/ms, data['%s_state' % pop]['V'][0]/mV,
                   color=c)
    AX[1].set_ylabel('neuron')
    AX[2].set_ylabel('rate (Hz)')
    AX[3].set_ylabel('V (mV)')
    AX[3].set_xlabel('time (ms)')
    plt.show()

else:

    net, data = snd.run(Model, filename='data/RS-FS.h5', report='text')
    snd.describe(net)
    print()
    print(' --- simulations results ---')
    for pop in ['RecExc', 'RecInh']:
        sp = data['%s_spikes' % pop]
        print('%s: %.2f Hz' % (pop, len(sp['t'])/sp['N']/float(Model['tstop']/second)))
