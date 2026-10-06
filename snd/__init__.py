"""
spiking_network_dynamics (snd) -- layers on top of brian2

  layer 1  equations : brian2 equation blocks with named parameters
  layer 2  cells, synapses, inputs, record : thin constructors returning
           named brian2 objects (NeuronGroup, Synapses, PoissonInput, monitors)
  layer 3  model : model description (nested dict with units) -> brian2.Network

  io       : save/load data and models with units (HDF5)
  scan     : parameter scans
  legacy   : conversion of `neural_network_dynamics` (ntwk) model dictionaries
"""
from . import equations, cells, synapses, inputs, record, io, model, scan, legacy
from .model import build, run
from .describe import describe
