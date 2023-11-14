from diffsimgen.scripts import helper_functions
from diffsimgen.scripts import generate_training_data
import numpy as np

def diffsimrun(model,bval,bvec,S0,SNR,numofsim=100000,delta=None,Delta=None,TE=None):

  '''

  Function for generating training data (normalized signal and microstructure parameters)
  Current models accepted:
  - NODDI_watson
  bval: 
    path to .bval file or numpy array with bvalues.
  bvec: 
    path to .bvec file or numpy array with bvectors.
  S0: 
    Signal. Example -> 100, [10,100], [10,20,30,40,50]
    This can be a single value or array. If you specify two S0 values ([10,100])
    then a random S0 value will be drawn within that range with equal probability.
    This can help if you want to cover the range of signals that might be present
    on a voxel-to-voxel basis. If you have multi-shell data, then you can specify
    the signal at each shell ([10,20,30,40,50]), then the data will be simulated
    using only those specified values. 
  SNR: 
    Signal to noise ratio. Example -> 40, [10,50], [10,20,30,40,50]
    Idea is the same as the S0 variable above.
    *Gaussian noise will be added to real and imag signal then combined
  numofsim: 
    Number of simulations to perform (i.e. how many random microstructural
    environments should data be simulated from) (default=100000).
  delta: 
    Pulse duration time in seconds (default=None)
  Delta: 
    Pulse seperation time in seconds (default=None)
  TE: 
    Echo time (default=None)
  Note, if you want to change fixed parameters (for example, in NODDI the free
  diffusivity and parallel diffusivity are fixed) this can be adjusted
  in the models.py file

  returns:
    normalized signal (S/S0), microstructure parameters, and the name of the 
    microstructural parameters

  '''

  if type(SNR) == int or len(SNR) == 1:
    SNRarr = np.tile(SNR,numofsim)
  elif len(SNR) == 2:
    SNRarr = np.random.uniform(SNR[0],SNR[1],size=[numofsim,1]) #if len(SNR) == 2 then assume that someone wants to randomly sample SNR between low and high
  else:
    SNRarr = np.tile(SNR,int(np.ceil(numofsim/len(SNR)))) #if len(SNR) > 2 then assume that someone wants to generate signal using the provided SNR values
  
  if type(S0) == int or len(S0) == 1:
    S0arr = np.tile(S0,numofsim)
  elif len(S0) == 2:
    S0arr = np.random.uniform(S0[0],S0[1],size=[numofsim,1]) #if len(S0) == 2 then assume that someone wants to randomly sample S0 between low and high
  else:
    S0arr = np.tile(S0,int(np.ceil(numofsim/len(S0)))) #if len(S0) > 2 then assume that someone wants to generate signal using all provided S0 values

  acq_scheme = helper_functions.get_acq_scheme(bval,bvec,delta,Delta,TE)
  function = getattr(generate_training_data, f'{model}')
    
  signal,parameters,parameter_names = function(numofsim,acq_scheme,S0arr,SNRarr)

  return signal,parameters,parameter_names
