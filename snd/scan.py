"""
Parameter scans.

Parameters are addressed by their path in the model description, as a tuple
of keys (or a dotted string when all keys are strings):

    grid = {('populations', 'Exc', 'params', 'b'): [0, 20, 40]*pA,
            ('projections', ('AffExc', 'Exc'), 'w'): [1, 2]*nS,
            'seed': [1, 2, 3]}
    scan.run(Model, grid, folder='data/scan')
    S = scan.load('data/scan')   # S['grid'], S['filenames'], S['values']

Each configuration is simulated with `model.run` (in parallel processes) and
saved as `<folder>/<i>.h5`. The grid is saved as `<folder>/scan.h5`.
"""
import copy, itertools, os
import multiprocessing as mp
import numpy as np
import brian2 as b2

from . import io, model


def _path(key):
    return tuple(key.split('.')) if isinstance(key, str) else tuple(key)


def set_value(Model, path, value):
    d = Model
    for k in path[:-1]:
        d = d[k]
    d[path[-1]] = value


def configurations(Model, grid):
    """ list of (values, model) for all combinations of the grid """
    keys = list(grid)
    out = []
    for values in itertools.product(*[list(grid[k]) for k in keys]):
        M = copy.deepcopy(Model)
        for key, val in zip(keys, values):
            set_value(M, _path(key), val)
        out.append((values, M))
    return out


def _run_one(args):
    M, filename = args
    model.run(M, filename=filename)
    return filename


def run(Model, grid, folder='scan',
        processes=None,
        fix_missing_only=False):
    """
    processes : number of parallel processes (default: number of cores,
                1 to run sequentially)
    fix_missing_only : only run the configurations whose file is missing
    """
    os.makedirs(folder, exist_ok=True)
    configs = configurations(Model, grid)
    filenames = [os.path.join(folder, '%i.h5' % i) for i in range(len(configs))]

    io.save(os.path.join(folder, 'scan.h5'),
            {'model': Model, 'keys': [str(k) for k in grid],
             'grid': {'%i' % i: grid[k] if isinstance(grid[k], b2.Quantity)
                      else np.asarray(grid[k]) for i, k in enumerate(grid)},
             'filenames': filenames})

    todo = [(M, fn) for (_, M), fn in zip(configs, filenames)
            if not (fix_missing_only and os.path.isfile(fn))]
    if processes == 1:
        for args in todo:
            _run_one(args)
    else:
        with mp.Pool(processes) as pool:
            pool.map(_run_one, todo)
    return filenames


def load(folder):
    """ grid description and filenames of a scan """
    S = io.load(os.path.join(folder, 'scan.h5'))
    S['grid'] = {k: S['grid']['%i' % i] for i, k in enumerate(S['keys'])}
    S['values'] = list(itertools.product(*S['grid'].values()))
    return S
