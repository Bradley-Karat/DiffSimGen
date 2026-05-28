from diffsimgen.scripts import helper_functions
from diffsimgen.scripts import generate_training_data
import numpy as np
import pickle
import os

def diffsimrun(model,bval,bvec,SNR,output='Default',numofsim=100000,noise_type='rician',delta=None,Delta=None,TE=None,parameter_distributions=[]):

  '''

  Function for generating training data (signal and microstructure parameters)
  Current models accepted:
  - SANDI
  - NODDI_watson
  - ball
  - stick
  - ball_stick
  - zeppelin
  - zeppelin_stick
  bval: str or array (N,)
    path to .bval file or numpy array with bvalues.
  bvec: str or array (N,3)
    path to .bvec file or numpy array with bvectors.
  output: str
    Output path/filename for saving as a .pkl. If 'Default' saves in 
    current working directory. If None does not save.
  SNR: float or array
    Signal to noise ratio. Example -> [40], [10,50], [10,20,30,40,50]
    If you specify two SNR values ([10,100])
    then a random SNR value will be drawn within that range with equal probability.
    This can help if you want to cover the range of signals that might be present
    on a voxel-to-voxel basis. If you have multi-shell data, then you can specify
    the SNR at each shell ([10,20,30,40,50]), then the data will be simulated
    using only those specified values. 
  numofsim: int
    Number of simulations to perform (i.e. how many random microstructural
    environments should data be simulated from) (default=100000).
  noise_type: str
    Type of noise to inject. One of 'rician' or 'gaussian'
  delta: float (optional unless SANDI model chosen)
    Pulse duration time in seconds (default=None)
  Delta: float (optional unless SANDI model chosen)
    Pulse seperation time in seconds (default=None)
  TE: float (optional)
    Echo time (default=None)
  parameter_distributions: Array (optional)
    Array of model parameters which will be used to simulate signal as opposed
    to uniform sampling on the whole parameter range. Useful if you want to 
    simulate signal from realistic parameter distributions in the brain. *Note*
    the index of the parameters in the array provided has to match the order of
    the parameters seen in the <model>.json file in the resources folder. For
    example, if you are providing a parameter distribution for the stick model,
    the array will be [numofparameters,3] where the 0th index is theta, the 1st
    index is phi, and the 2nd index is Dpar (as seen in stick_parameter_range.json)
    numofsim samples will be drawn from the provided parameter distributions with
    replacement.
  Note, if you want to change fixed parameters (for example, in NODDI the free
  diffusivity and parallel diffusivity are fixed) this can be adjusted
  in the models.py file

  returns:
    - Normalized noisy signal
    - Normalized ground-truth (noiseless) signal
    - Microstructure parameters
    - Name of the microstructural parameters
    - SNR used for each simulation    

  '''

  if len(SNR) == 1:
    SNRarr = np.tile(SNR,numofsim)
  elif len(SNR) == 2:
    SNRarr = np.random.uniform(SNR[0],SNR[1],size=[numofsim,1]) #if len(SNR) == 2 then assume that someone wants to randomly sample SNR between low and high
  else:
    SNRarr = np.tile(SNR,int(np.ceil(numofsim/len(SNR)))) #if len(SNR) > 2 then assume that someone wants to generate signal using the provided SNR values
  
  #acq_scheme = helper_functions.get_acq_scheme(bval,bvec,delta,Delta,TE)
  acq_scheme = helper_functions.get_acq_scheme(bval,bvec)

  noise_type = noise_type
  function = getattr(generate_training_data, f'{model}')

  if model=='SANDI': # SANDI has some extra considerations for simulating signal. Provide powder averaged and directional signal
    if np.logical_or(delta==None,Delta==None):
      raise TypeError("SANDI model was chosen but delta and/or Delta were not specified. These are required to fit the SANDI model.")
    else:
        signal_directional,signal_powder_avg,parameters,parameter_names,signal_noiseless = function(numofsim,bval,bvec,delta,Delta,SNRarr,noise_type,parameter_distributions)

        modeldict = {"signal":signal_directional,"signal_podwer_avg":signal_powder_avg,"signal_noiseless":signal_noiseless,"parameters":parameters,"parameter_names":parameter_names,"SNRarr":SNRarr}

        if output!=None:
          if output=="Default":
            fpath = os.getcwd()
            output = f'{fpath}/{model}_diffusion_simulations.pkl' 
            
          with open(output, 'wb') as fp:
            pickle.dump(modeldict, fp)

        return signal_directional, signal_powder_avg, signal_noiseless, parameters, parameter_names, SNRarr

  else:

    signal,parameters,parameter_names,signal_noiseless = function(numofsim,acq_scheme,SNRarr,noise_type,parameter_distributions)

    modeldict = {"signal":signal,"signal_noiseless":signal_noiseless,"parameters":parameters,"parameter_names":parameter_names,"SNRarr":SNRarr}
  
    if output!=None:
      if output=="Default":
        fpath = os.getcwd()
        output = f'{fpath}/{model}_diffusion_simulations.pkl' 
          
      with open(output, 'wb') as fp:
        pickle.dump(modeldict, fp)

    return signal, signal_noiseless, parameters, parameter_names, SNRarr
