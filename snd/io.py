"""
Saving / loading with units.

`collect(net)` gathers the data of all monitors of a brian2 Network into a
dictionary (quantities keep their units):

    {'Exc_spikes': {'i': ..., 't': ... * second, 'N': 4000},
     'Exc_rate':   {'t': ..., 'rate': ... * hertz},
     'Exc_state':  {'t': ..., 'record': [0, 1, 2, 3], 'V': ... * volt}}

`save` / `load` write / read any nested dictionary of quantities, arrays,
numbers and strings to HDF5. Quantities are stored as SI values with their
dimensions as an attribute, and come back as brian2 quantities.
Tuple keys (e.g. projections ('Exc', 'Inh')) are preserved.
"""
import os, pathlib
import numpy as np
import h5py
import brian2 as b2
from brian2.units.fundamentalunits import get_or_create_dimension


# ------------------------------------------------------------------ #
#    monitors -> dictionary
# ------------------------------------------------------------------ #

def collect(net):
    """ data of all monitors of a brian2.Network, keyed by monitor name """
    data = {}
    for obj in net.objects:
        if isinstance(obj, b2.SpikeMonitor):
            data[obj.name] = {'i': np.array(obj.i[:]), 't': obj.t[:],
                              'N': len(obj.source)}
        elif isinstance(obj, b2.PopulationRateMonitor):
            data[obj.name] = {'t': obj.t[:], 'rate': obj.rate[:]}
        elif isinstance(obj, b2.StateMonitor):
            data[obj.name] = {'t': obj.t[:], 'record': np.array(obj.record)}
            for var in obj.record_variables:
                data[obj.name][var] = getattr(obj, var)[:]
    return data


# ------------------------------------------------------------------ #
#    dictionary <-> HDF5
# ------------------------------------------------------------------ #

TUPLE_SEP = '->'


def _key(key):
    if isinstance(key, tuple):
        return TUPLE_SEP.join(key), True
    return str(key), False


def _write(grp, key, val):
    key, is_tuple = _key(key)
    if isinstance(val, dict):
        node = grp.create_group(key)
        for k, v in val.items():
            _write(node, k, v)
    elif val is None:
        node = grp.create_dataset(key, data=h5py.Empty('f'))
    elif isinstance(val, str):
        node = grp.create_dataset(key, data=val)
    elif isinstance(val, b2.Quantity) and not val.is_dimensionless:
        node = grp.create_dataset(key, data=np.asarray(val))
        node.attrs['dim'] = np.array(val.dim._dims)
    elif isinstance(val, (list, tuple)) and len(val) > 0 and \
            all(isinstance(v, str) for v in val):
        node = grp.create_dataset(key, data=np.array(val, dtype=h5py.string_dtype()))
        node.attrs['str_list'] = True
    else:
        node = grp.create_dataset(key, data=np.asarray(val))
    if is_tuple:
        node.attrs['tuple_key'] = True


def _read(node):
    if isinstance(node, h5py.Group):
        out = {}
        for k, v in node.items():
            key = tuple(k.split(TUPLE_SEP)) if v.attrs.get('tuple_key', False) else k
            out[key] = _read(v)
        return out
    if node.shape is None:  # h5py.Empty
        return None
    if node.attrs.get('str_list', False):
        return [s.decode() if isinstance(s, bytes) else s for s in node[()]]
    val = node[()]
    if isinstance(val, bytes):
        return val.decode()
    if 'dim' in node.attrs:
        return b2.Quantity(val, dim=get_or_create_dimension(tuple(node.attrs['dim'])))
    return val


def save(filename, data):
    """ write a nested dictionary to an HDF5 file """
    pathlib.Path(os.path.dirname(os.path.abspath(filename))).mkdir(parents=True,
                                                                   exist_ok=True)
    with h5py.File(filename, 'w') as f:
        for key, val in data.items():
            _write(f, key, val)


def load(filename):
    """ read an HDF5 file written by `save` """
    with h5py.File(filename, 'r') as f:
        return _read(f)
