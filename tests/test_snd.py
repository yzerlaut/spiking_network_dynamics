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
                                          ('biexp', dict(tau_rise=1*ms, tau_decay=8*ms))])
def test_channel_peak_equals_weight(kind, params):
    b2.defaultclock.dt = 0.01*ms
    G = snd.cells.population('G', 1, kind='LIF', params=dict(VT=100*mV),
                             channels={'X': dict(kind=kind, E=0*mV, **params)})
    src = b2.SpikeGeneratorGroup(1, [0], [1]*ms)
    S = snd.synapses.projection(src, G, 'X', w=2*nS, indegree=1)
    M = b2.StateMonitor(G, 'g_X', record=0)
    b2.Network(G, src, S, M).run(40*ms)
    assert float(M.g_X[0].max()/nS) == pytest.approx(2., rel=1e-2)


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


def test_scan(tmp_path):
    M = small_model()
    M['tstop'] = 20*ms
    grid = {('populations', 'Exc', 'params', 'b'): [0, 40]*pA, 'seed': [1, 2]}
    files = snd.scan.run(M, grid, folder=str(tmp_path/'scan'), processes=1)
    assert len(files) == 4
    S = snd.scan.load(str(tmp_path/'scan'))
    assert len(S['values']) == 4
    assert snd.io.load(files[1])['model']['seed'] == 2
