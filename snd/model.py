"""
Layer 3 -- network models from a model description.

A model is a plain (nested) dictionary of quantities, numbers and strings,
so that it can be saved, copied and scanned:

    Model = {
        'dt': 0.1*ms, 'tstop': 1500*ms, 'seed': 3,

        # synaptic channels (default for all populations)
        'channels': {'AMPA': dict(kind='exp', E=0*mV, tau=5*ms),
                     'GABA': dict(kind='exp', E=-80*mV, tau=5*ms)},

        'populations': {
            'Exc': dict(N=4000, kind='AdExp', params=dict(b=40*pA, ...)),
            'Inh': dict(N=1000, kind='EIF', params=dict(DeltaT=0.5*mV, ...)),
        },

        'afferents': {
            # rate: constant, or array sampled every `rate_dt` (default: dt)
            # explicit=False: independent Poisson drive (see inputs.poisson_drive)
            # explicit=True : PoissonGroup connected with synapses.projection
            'AffExc': dict(N=400, rate=5*Hz),
        },

        'projections': {
            ('Exc', 'Inh'): dict(channel='AMPA', w=1*nS, indegree=200),
            ('AffExc', 'Exc'): dict(channel='AMPA', w=1*nS, indegree=200),
            ...
        },

        # monitors per population (default: spikes and population rate)
        'record': {'Exc': dict(spikes=True, rate=True, variables=['V'], record=4)},
    }

`build(Model)` returns a `brian2.Network` whose objects are named after the
model: net['Exc'] (NeuronGroup), net['Exc_to_Inh'] (Synapses),
net['AffExc_to_Exc'] (drive), net['Exc_spikes'] (SpikeMonitor), ...
Everything can be modified with brian2 code before `net.run(...)`.
"""
import brian2 as b2

from . import cells, synapses, inputs, record, io

DEFAULT_RECORD = dict(spikes=True, rate=True)


def build(Model):
    """ brian2.Network from a model description """
    if 'seed' in Model:
        b2.seed(Model['seed'])
    b2.defaultclock.dt = Model['dt']

    groups = {}
    for name, spec in Model['populations'].items():
        spec = dict(spec)
        spec.setdefault('channels', Model.get('channels', {}))
        groups[name] = cells.population(name, **spec)

    afferents = Model.get('afferents', {})
    for name, spec in afferents.items():
        if spec.get('explicit', False):
            groups[name] = inputs.poisson_population(
                name, spec['N'], spec['rate'],
                dt=spec.get('rate_dt', Model['dt']))

    objects = list(groups.values())

    for (pre, post), spec in Model.get('projections', {}).items():
        spec = dict(spec)
        name = spec.pop('name', '%s_to_%s' % (pre, post))
        if (pre in afferents) and not afferents[pre].get('explicit', False):
            if set(spec)-{'channel', 'w', 'indegree', 'p_release'} or \
                    ('indegree' not in spec):
                raise ValueError("drive '%s' (afferent '%s' not explicit) accepts "
                                 "channel, w, indegree, p_release" % (name, pre))
            objects.append(inputs.poisson_drive(
                groups[post], spec['channel'], N=spec['indegree'],
                rate=afferents[pre]['rate'], w=spec['w'],
                p_release=spec.get('p_release', 1.),
                dt=afferents[pre].get('rate_dt', Model['dt']),
                name=name))
        else:
            objects.append(synapses.projection(groups[pre], groups[post],
                                               name=name, **spec))

    to_record = Model.get('record', {})
    for name, group in groups.items():
        if (name in to_record) or (name in Model['populations']):
            objects += record.monitors(group, **to_record.get(name, DEFAULT_RECORD))

    return b2.Network(*objects)


def run(Model, filename=None, report=None):
    """
    build, run for Model['tstop'], collect the monitors' data
    (and save it with the model in `filename`)

    returns (net, data)
    """
    net = build(Model)
    net.run(Model['tstop'], report=report)
    data = io.collect(net)
    data['model'] = Model
    if filename is not None:
        io.save(filename, data)
    return net, data
