"""
Print the brian2-level content of a network: equations, thresholds, resets,
namespaces, synaptic code. To see what a model built with `snd` amounts to in
plain brian2.
"""
import brian2 as b2


def _namespace(obj):
    ns = getattr(obj, 'namespace', None) or {}
    return '\n'.join('    %s = %s' % (k, v) for k, v in sorted(ns.items()))


def describe(net, show=True):
    lines = []
    for obj in sorted(net.objects, key=lambda o: o.name):
        if isinstance(obj, b2.PoissonGroup):
            lines += ['', '[PoissonGroup] %s  (N=%i)' % (obj.name, len(obj)),
                      _namespace(obj)]
        elif isinstance(obj, b2.NeuronGroup):
            lines += ['', '[NeuronGroup] %s  (N=%i)' % (obj.name, len(obj)),
                      '  model:', str(obj.equations),
                      '  threshold: %s' % obj.events.get('spike'),
                      '  reset:     %s' % obj.event_codes.get('spike'),
                      '  refractory: %s' % obj._refractory,
                      '  namespace:', _namespace(obj)]
        elif isinstance(obj, b2.Synapses):
            lines += ['', '[Synapses] %s  (%s -> %s, %i synapses)' %
                      (obj.name, obj.source.name, obj.target.name, len(obj)),
                      '  model:', str(obj.equations),
                      '  on_pre: %s' % obj.pre.code]
            if obj.namespace:
                lines += ['  namespace:', _namespace(obj)]
        elif isinstance(obj, (b2.SpikeMonitor, b2.PopulationRateMonitor,
                              b2.StateMonitor)):
            lines += ['[%s] %s  (on %s)' % (type(obj).__name__, obj.name,
                                            obj.source.name)]
        elif isinstance(obj, b2.groups.group.CodeRunner):
            lines += ['', '[run_regularly] %s  (on %s)' % (obj.name, obj.group.name),
                      '  %s' % obj.abstract_code]
    text = '\n'.join(lines)
    if show:
        print(text)
    return text
