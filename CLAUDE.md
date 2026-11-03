# Instructions for Claude

## `snd`: rewrite of `ntwk`

This repo (package `snd`, remote github.com/yzerlaut/spiking_network_dynamics, branch main) rewrites the sibling submodule `../neural_network_dynamics` (package `ntwk`), which stays untouched. The old API hides brian2 (floats baked into equation strings, flat string-keyed Model dict, objects by list index); each `snd` layer should map clearly onto brian2 code.

Layers:

1. `snd.equations`: brian2 Equations, named params
2. `cells`, `synapses`, `inputs`, `record`: return named brian2 objects
3. `snd.model.build/run`: nested model dict -> `b2.Network`

Design decisions:

- brian2 units everywhere, model dict included; unitless only at the theory/plot boundary
- Synaptic channels per receptor type (AMPA, GABA...), weights per projection, an extra channel for a different tau
- The adaptation variable is `w_adapt` (`w` clashes with the synaptic `w`)
- First pass is the simulation core (cells, synapses, inputs, record, io, scan, legacy converter); theory, transfer_functions, nrn and vision are ported later

Porting `ntwk` demos:

- Old `p_X_Y` (pconn) is NOT brian2 `connect(p=)`: `ntwk` draws a fixed indegree `int(p*N_pre)` per cell, with replacement, so port it as `indegree=`. Using `p=` made the Zerlaut 2019 network saturate (~190 Hz vs ~50 Hz).
- Afferent `p*N_aff` becomes the drive `indegree` (may be fractional).
- Running old `ntwk` needs shims: `scipy.integrate.simps = simpson` to import with current scipy, `np.string_ = np.bytes_` to save.
