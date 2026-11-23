import numpy as np
import pytest
import brian2 as b2
from brian2 import ms, mV, nS, pF, pA, Hz, second

import snd

b2.prefs.codegen.target = 'numpy'

CHANNELS = {'AMPA': dict(kind='exp', E=0*mV, tau=5*ms),
            'GABA': dict(kind='exp', E=-80*mV, tau=5*ms)}


# ---------- layer 1 ---------- #

@pytest.mark.parametrize('kind, params', [('exp', dict(tau=5*ms)),
                                          ('alpha', dict(tau=5*ms)),
                                          ('biexp', dict(tau_rise=1*ms, tau_decay=8*ms)),
                                          ('nmda', dict(tau_rise=2*ms, tau_decay=80*ms,
                                                        Mg=1*b2.mM))])
def test_channel_peak_equals_weight(kind, params):
    b2.defaultclock.dt = 0.01*ms
    G = snd.cells.population('G', 1, kind='LIF', params=dict(VT=100*mV),
                             channels={'X': dict(kind=kind, E=0*mV, **params)})
    src = b2.SpikeGeneratorGroup(1, [0], [1]*ms)
    S = snd.synapses.projection(src, G, 'X', w=2*nS, indegree=1)
    M = b2.StateMonitor(G, 'g_X', record=0)
    b2.Network(G, src, S, M).run(40*ms)
    assert float(M.g_X[0].max()/nS) == pytest.approx(2., rel=1e-2)


def test_nmda_mg_block():
    """ NMDA current blocked at rest, (mostly) unblocked when depolarized """
    b2.defaultclock.dt = 0.1*ms
    G = snd.cells.population('G', 2, kind='LIF', params=dict(VT=100*mV),
                             channels={'NMDA': dict(kind='nmda', E=0*mV, Mg=1*b2.mM,
                                                    tau_rise=2*ms, tau_decay=80*ms)})
    G.g_NMDA = 1*nS
    G.V = [-70, -10]*mV
    ratio = np.asarray(G.I_NMDA[:]/(G.g_NMDA[:]*(0*mV-G.V[:])))  # unblocked fraction
    assert ratio[0] < 0.1 and ratio[1] > 0.5


# ---------- layer 2 ---------- #

def test_population_params():
    G = snd.cells.population('Exc', 10, params=dict(b=40*pA), channels=CHANNELS)
    assert G.namespace['b'] == 40*pA
    assert G.namespace['Vcut'] == -50*mV+5*2*mV
    assert np.all(G.V == -70*mV)
    with pytest.raises(ValueError):
        snd.cells.population('X', 10, params=dict(Gl=10*nS))


def test_heterogeneous_params():
    VT = np.linspace(-55, -45, 10)*mV
    G = snd.cells.population('Exc', 10, kind='LIF', params=dict(VT=VT))
    assert np.allclose(G.VT[:]/mV, VT/mV)


def test_fixed_indegree_no_autapse():
    G = snd.cells.population('Exc', 100, channels=CHANNELS)
    S = snd.synapses.projection(G, G, 'AMPA', w=1*nS, indegree=20)
    assert np.all(np.bincount(S.j[:]) == 20)
    assert not np.any(S.i[:] == S.j[:])
    assert len(set(zip(S.i[:], S.j[:]))) == len(S)


def test_unknown_channel():
    G = snd.cells.population('Exc', 10, channels=CHANNELS)
    with pytest.raises(ValueError, match='NMDA'):
        snd.synapses.projection(G, G, 'NMDA', w=1*nS, indegree=2)


@pytest.mark.parametrize('rate', [10*Hz, np.r_[np.zeros(500), 20*np.ones(500)]*Hz])
def test_poisson_drive(rate):
    b2.defaultclock.dt = 0.1*ms
    G = snd.cells.population('G', 200, kind='LIF', params=dict(VT=100*mV),
                             channels={'AMPA': dict(kind='exp', E=0*mV, tau=1e6*ms)})
    D = snd.inputs.poisson_drive(G, 'AMPA', N=50, rate=rate, w=1*nS, dt=0.1*ms)
    b2.Network(G, D).run(100*ms)
    expected = 50*float(np.mean(rate)/Hz)*0.1  # events per neuron
    assert float(np.mean(G.g_AMPA/nS)) == pytest.approx(expected, rel=0.05)


# ---------- io ---------- #

def test_io_roundtrip(tmp_path):
    data = {'model': {'dt': 0.1*ms, 'name': 'test', 'N': 4,
                      'rates': np.arange(3)*Hz, 'none': None,
                      'projections': {('Exc', 'Inh'): {'w': 1*nS}},
                      'list': ['a', 'b']}}
    snd.io.save(tmp_path/'d.h5', data)
    out = snd.io.load(tmp_path/'d.h5')['model']
    assert out['dt'] == 0.1*ms and out['name'] == 'test' and out['N'] == 4
    assert np.all(out['rates'] == np.arange(3)*Hz)
    assert out['projections'][('Exc', 'Inh')]['w'] == 1*nS
    assert out['none'] is None and out['list'] == ['a', 'b']


# ---------- layer 3 ---------- #

def small_model():
    return {'dt': 0.1*ms, 'tstop': 200*ms, 'seed': 1,
            'channels': CHANNELS,
            'populations': {'Exc': dict(N=400, kind='AdExp', params=dict(b=40*pA)),
                            'Inh': dict(N=100, kind='EIF', params=dict(DeltaT=0.5*mV))},
            'afferents': {'AffExc': dict(N=100, rate=np.linspace(5, 10, 2000)*Hz)},
            'projections': {('Exc', 'Exc'): dict(channel='AMPA', w=1*nS, indegree=20),
                            ('Exc', 'Inh'): dict(channel='AMPA', w=1*nS, indegree=20),
                            ('Inh', 'Exc'): dict(channel='GABA', w=4*nS, indegree=5),
                            ('Inh', 'Inh'): dict(channel='GABA', w=4*nS, indegree=5),
                            ('AffExc', 'Exc'): dict(channel='AMPA', w=1*nS, indegree=50),
                            ('AffExc', 'Inh'): dict(channel='AMPA', w=1*nS, indegree=50)},
            'record': {'Exc': dict(spikes=True, rate=True, variables=['V'], record=2)}}


def test_build_run_save(tmp_path):
    net, data = snd.run(small_model(), filename=tmp_path/'sim.h5')
    assert isinstance(net['Exc'], b2.NeuronGroup)
    assert isinstance(net['Exc_to_Inh'], b2.Synapses)
    assert data['Exc_state']['V'].shape == (2, 2000)
    out = snd.io.load(tmp_path/'sim.h5')
    assert out['Exc_spikes']['t'].dim == second.dim
    assert out['model']['projections'][('Exc', 'Inh')]['indegree'] == 20
    assert 'g_AMPA' in snd.describe(net, show=False)


def test_labelled_projections(tmp_path):
    """ (pre, post, label) keys: several projections between the same populations """
    M = small_model()
    M['channels'] = dict(CHANNELS, NMDA=dict(kind='nmda', E=0*mV, Mg=1*b2.mM,
                                             tau_rise=2*ms, tau_decay=80*ms))
    M['projections'][('Exc', 'Exc', 'NMDA')] = dict(channel='NMDA', w=1*nS, indegree=10)
    net, data = snd.run(M, filename=tmp_path/'sim.h5')
    assert isinstance(net['Exc_to_Exc_NMDA'], b2.Synapses)
    assert isinstance(net['Exc_to_Exc'], b2.Synapses)
    out = snd.io.load(tmp_path/'sim.h5')
    assert out['model']['projections'][('Exc', 'Exc', 'NMDA')]['indegree'] == 10


def test_build_is_reproducible():
    _, d1 = snd.run(small_model())
    _, d2 = snd.run(small_model())
    assert np.array_equal(d1['Exc_spikes']['i'], d2['Exc_spikes']['i'])


def test_explicit_afferent():
    M = small_model()
    M['afferents']['AffExc']['explicit'] = True
    net = snd.build(M)
    assert isinstance(net['AffExc'], b2.PoissonGroup)
    assert np.all(np.bincount(net['AffExc_to_Exc'].j[:]) == 50)


def test_spike_times_afferent():
    M = small_model()
    M['afferents']['Stim'] = dict(N=2, indices=[0, 1], times=[50, 100]*ms)
    M['projections'][('Stim', 'Exc')] = dict(channel='AMPA', w=100*nS,
                                             i=[0, 0, 1], j=[3, 4, 5])
    M['record']['Stim'] = dict(spikes=True)
    net, data = snd.run(M)
    assert isinstance(net['Stim'], b2.SpikeGeneratorGroup)
    assert list(net['Stim_to_Exc'].j[:]) == [3, 4, 5]
    assert len(data['Stim_spikes']['t']) == 2


def test_scan(tmp_path):
    M = small_model()
    M['tstop'] = 20*ms
    grid = {('populations', 'Exc', 'params', 'b'): [0, 40]*pA, 'seed': [1, 2]}
    files = snd.scan.run(M, grid, folder=str(tmp_path/'scan'), processes=1)
    assert len(files) == 4
    S = snd.scan.load(str(tmp_path/'scan'))
    assert len(S['values']) == 4
    assert snd.io.load(files[1])['model']['seed'] == 2


# ---------- plot ---------- #

def test_vm_trace_spikes_and_clip():
    M = small_model()
    M['tstop'] = 500*ms
    M['afferents']['AffExc']['rate'] = 20*Hz
    _, data = snd.run(M)
    sp, state = data['Exc_spikes'], data['Exc_state']
    k = int(np.argmax([np.sum(sp['i'] == n) for n in state['record']]))
    tspikes = sp['t'][sp['i'] == state['record'][k]]/ms
    assert len(tspikes) > 0
    t, V, segments = snd.plot.vm_trace(data, 'Exc', k, spike_peak=0*mV, clip=True)
    assert len(segments) == len(tspikes)
    for ts, v0, peak in segments:
        i = np.searchsorted(t, ts+1e-6)-1
        assert v0 < -40 and peak == 0      # V before reset, below Vcut
        assert np.all(np.isnan(V[i+1:i+50]))  # 5ms clipped (tref)
        if i+50 < len(V):
            assert not np.isnan(V[i+50])


def test_thin_spikes():
    i = np.array([0, 1, 0, 0, 1, 0])
    t = np.array([0, 5, 50, 120, 300, 210])*ms
    keep = snd.plot.thin_spikes(i, t, dead_time=100*ms)
    assert list(keep) == [True, True, False, True, True, False]
    keep = snd.plot.thin_spikes(i, t, dead_time=100*ms, subsampling=2)
    assert list(keep) == [True, False, False, True, False, False]
    # random order: kept spikes separated by more than dead_time, and no time
    # bias (uniform kept density for a regular train)
    t = np.arange(0, 100000, 1.)*ms
    keep = snd.plot.thin_spikes(np.zeros(len(t)), t, dead_time=100*ms,
                                random=True, seed=1)
    assert np.all(np.diff(t[keep]/ms) >= 100)
    assert np.sum(keep[:50000]) == pytest.approx(np.sum(keep[50000:]), rel=0.05)


def test_membrane_potential_subplots():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    _, data = snd.run(small_model())
    fig, AX = plt.subplots(2, 1)
    snd.plot.membrane_potential_subplots(data, AX, ['Exc', 'Inh'],
                                         spike_peak=-10*mV, clip=5*ms,
                                         tzoom=(50*ms, 150*ms))
    assert len(AX[0].lines) >= 2           # 2 recorded neurons
    assert 'not recorded' in AX[1].texts[0].get_text()
    plt.close(fig)


@pytest.mark.parametrize('loc', ['bottom left', 'top left', 'bottom right', 'top right'])
def test_scale_bars_corners(loc):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot([0, 100], [-70, -50])
    snd.plot.scale_bars(ax, Vbar=10*mV, Tbar=20*ms, loc=loc)
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    xV = ax.lines[1].get_xdata()[0]  # vertical bar, outside the data range
    assert (xV > x1) if loc.endswith('right') else (xV < x0)
    yT = ax.lines[2].get_ydata()[0]  # time bar, at the top or bottom
    assert yT == (y1 if loc.startswith('top') else y0)
    plt.close(fig)
