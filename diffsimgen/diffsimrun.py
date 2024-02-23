from diffsimgen.scripts import helper_functions
from diffsimgen.scripts import generate_training_data
import numpy as np

def diffsimrun(model,bval,bvec,S0,SNR,numofsim=100000,noise_type='rician',delta=None,Delta=None,TE=None):

  '''

  Function for generating training data (signal and microstructure parameters)
  Current models accepted:
  - NODDI_watson
  - ball
  - stick
  - ball_stick
  bval: str or array (N,)
    path to .bval file or numpy array with bvalues.
  bvec: str or array (N,3)
    path to .bvec file or numpy array with bvectors.
  S0: float or array 
    Unweights signal value (b0). Example -> 100, [10,100], [10,20,30,40,50]
    This can be a single value or array. If you specify two S0 values ([10,100])
    then a random S0 value will be drawn within that range with equal probability.
    This can help if you want to cover the range of signals that might be present
    on a voxel-to-voxel basis. If you have multi-shell data, then you can specify
    the signal at each shell ([10,20,30,40,50]), then the data will be simulated
    using only those specified values. 
  SNR: float or array
    Signal to noise ratio. Example -> 40, [10,50], [10,20,30,40,50]
    Idea is the same as the S0 variable above.
  numofsim: int
    Number of simulations to perform (i.e. how many random microstructural
    environments should data be simulated from) (default=100000).
  noise_type: str
    Type of noise to inject. One of 'rician' or 'gaussian'
  delta: float (optional)
    Pulse duration time in seconds (default=None)
  Delta: float (optional)
    Pulse seperation time in seconds (default=None)
  TE: float (optional)
    Echo time (default=None)
  Note, if you want to change fixed parameters (for example, in NODDI the free
  diffusivity and parallel diffusivity are fixed) this can be adjusted
  in the models.py file

  returns:
    - Noisy signal
    - Ground-truth (noiseless) signal
    - Microstructure parameters
    - Name of the microstructural parameters
    - SNR used for each simulation
    - S0 used for each simulation
    
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
  noise_type = noise_type
  function = getattr(generate_training_data, f'{model}')
    
  signal,parameters,parameter_names,signal_noiseless = function(numofsim,acq_scheme,S0arr,SNRarr,noise_type)

  return signal, signal_noiseless, parameters, parameter_names, SNRarr, S0arr
