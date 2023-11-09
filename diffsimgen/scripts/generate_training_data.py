from diffsimgen.scripts.generate_parameter_array import generate_model_parameter_array
from diffsimgen.scripts import models
from diffsimgen.scripts import simulate_signal
import numpy as np

def NODDI_watson(numofsim,acq_scheme,S0arr,SNRarr): #Generate random array of NODDI parameters to then get our simulated signals

  numofacq = acq_scheme.number_of_measurements
  simulated_data = np.empty((numofsim,numofacq))

  parameter_array,parameter_names = generate_model_parameter_array(numofsim).NODDI_watson() #numofsim x 6 array

  for ii in range(numofsim):
    mu = parameter_array[ii,0:2]
    ODI = parameter_array[ii,2]
    watsonstickfrac = parameter_array[ii,3]
    ballfrac = parameter_array[ii,4]
    watsonfrac = parameter_array[ii,5]

    model,param_vector = models.NODDI_watson(mu,ODI,watsonstickfrac,ballfrac,watsonfrac).make_model()
    simulated_data[ii,:] = simulate_signal.simulate_SNR_signal(model,param_vector,S0arr[ii],SNRarr[ii],acq_scheme).simulate_noisy_signal()

  return simulated_data,parameter_array,parameter_names