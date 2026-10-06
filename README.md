# Spiking Network Dynamics

> Numerical simulations and theoretical analysis of the emergent dynamics in spiking neural networks

A thin, layered package on top of [**brian2**](https://brian2.readthedocs.io/en/stable/).
Every layer returns **plain brian2 objects** (named `NeuronGroup`, `Synapses`, monitors, ...),
so that you can drop down to brian2 code at any point.

| layer | module | what it gives | brian2 equivalent |
|---|---|---|---|
| 1 | `snd.equations` | equation blocks with *named* parameters | `brian2.Equations` |
| 2 | `snd.cells.population` | a neuronal population with synaptic channels | `brian2.NeuronGroup` |
| 2 | `snd.synapses.projection` | a projection onto one channel, with a connectivity rule | `brian2.Synapses` + `connect` |
| 2 | `snd.inputs.poisson_drive` / `poisson_population` | afferent Poisson input (constant or time-varying) | `run_regularly` / `brian2.PoissonGroup` |
| 2 | `snd.record.monitors` | named monitors | `SpikeMonitor`, `PopulationRateMonitor`, `StateMonitor` |
| 3 | `snd.build` / `snd.run` | network from a model description | `brian2.Network` |

plus `snd.io` (save/load with units, HDF5), `snd.scan` (parameter scans),
`snd.describe(net)` (print the brian2 content of a network) and
`snd.legacy.from_ntwk` (convert `neural_network_dynamics` model dictionaries).

## Installation

```
pip install -e .
```

## Usage

### Layer 3: model description

A model is a nested dictionary of brian2 quantities (units everywhere):

```python
import numpy as np
from brian2 import ms, mV, nS, pF, pA, Hz
import snd

Model = {
    'dt': 0.1*ms, 'tstop': 1500*ms, 'seed': 3,
    # synaptic channels, one per receptor type (default for all populations)
    'channels': {'AMPA': dict(kind='exp', E=0*mV, tau=5*ms),
                 'GABA': dict(kind='exp', E=-80*mV, tau=5*ms)},
    'populations': {
        'Exc': dict(N=4000, kind='AdExp', params=dict(b=40*pA, DeltaT=2*mV)),
        'Inh': dict(N=1000, kind='EIF', params=dict(DeltaT=0.5*mV)),
    },
    'afferents': {'AffExc': dict(N=400, rate=5*Hz)},
    'projections': {
        ('Exc', 'Exc'): dict(channel='AMPA', w=1*nS, indegree=200),
        ('Exc', 'Inh'): dict(channel='AMPA', w=1*nS, indegree=200),
        ('Inh', 'Exc'): dict(channel='GABA', w=4*nS, indegree=50),
        ('Inh', 'Inh'): dict(channel='GABA', w=4*nS, indegree=50),
        ('AffExc', 'Exc'): dict(channel='AMPA', w=1*nS, indegree=200),
        ('AffExc', 'Inh'): dict(channel='AMPA', w=1*nS, indegree=200),
    },
}

net, data = snd.run(Model, filename='data/sim.h5')   # build + run + save
data['Exc_spikes']['t']                              # spike times (quantity, in seconds)
```

or build only, and modify with brian2 code before running:

```python
net = snd.build(Model)
net['Exc'].V = '-70*mV + 5*mV*rand()'          # NeuronGroup 'Exc'
net['Inh_to_Exc'].w = '(4 + randn())*nS'        # Synapses 'Inh_to_Exc'
net.add(b2.StateMonitor(net['Exc'], 'w_adapt', record=[0]))
net.run(Model['tstop'])
snd.describe(net)                               # equations, namespaces, on_pre code
```

### Layer 2: brian2 objects

```python
import brian2 as b2

Exc = snd.cells.population('Exc', 4000, kind='AdExp', params=dict(b=40*pA),
                           channels={'AMPA': dict(kind='exp', E=0*mV, tau=5*ms),
                                     'GABA': dict(kind='exp', E=-80*mV, tau=5*ms)})
# -> b2.NeuronGroup, parameters in Exc.namespace
S = snd.synapses.projection(Exc, Exc, 'AMPA', w=1*nS, indegree=200)
# -> b2.Synapses(Exc, Exc, 'w : siemens', on_pre='g_AMPA_post += w', name='Exc_to_Exc')
drive = snd.inputs.poisson_drive(Exc, 'AMPA', N=200, rate=5*Hz, w=1*nS)
net = b2.Network(Exc, S, drive, *snd.record.monitors(Exc, rate=True))
```

### Layer 1: equations

```python
snd.equations.neuron('AdExp')     # model, threshold, reset, refractory of a NeuronGroup
snd.equations.channel('AMPA')     # dg_AMPA/dt = -g_AMPA/tau_AMPA ; I_AMPA = g_AMPA*(E_AMPA - V)
```

## Conventions

- Membrane parameters follow the brian2 AdEx example: `C, gL, EL, VT, DeltaT, Vcut, Vr, tref, a, b, tauw`.
  The adaptation current is `w_adapt` (`w` is the synaptic weight).
- Synaptic channels are per receptor type, of kind `exp`, `alpha` or `biexp`
  (peak conductance = synaptic weight). For different time constants, declare another channel.
- `indegree=K`: each neuron receives exactly K synapses from distinct presynaptic neurons (no autapse).
  `p=...` is brian2's connection probability.

## Demo

```
python demo/RS-FS.py        # run
python demo/RS-FS.py plot   # plot
```

## Tests

```
pytest tests
```
