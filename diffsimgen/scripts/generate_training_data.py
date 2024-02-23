from diffsimgen.scripts.generate_parameter_array import generate_model_parameter_array
from diffsimgen.scripts import models
from diffsimgen.scripts import simulate_signal
import numpy as np

def NODDI_watson(numofsim,acq_scheme,S0arr,SNRarr,noise_type): #Generate random array of NODDI parameters to then get our simulated signals

  numofacq = acq_scheme.number_of_measurements
  simulated_data = np.empty((numofsim,numofacq))
  simulated_data_nonoise = np.empty((numofsim,numofacq))
  noise_type = noise_type


  parameter_array,parameter_names = generate_model_parameter_array(numofsim).NODDI_watson() #numofsim x 6 array

  for ii in range(numofsim):
    mu = parameter_array[ii,0:2]
    ODI = parameter_array[ii,2]
    watsonstickfrac = parameter_array[ii,3]
    ballfrac = parameter_array[ii,4]
    watsonfrac = parameter_array[ii,5]

    model,param_vector = models.NODDI_watson(mu,ODI,watsonstickfrac,ballfrac,watsonfrac).make_model()
    simulated_data[ii,:] = simulate_signal.simulate_SNR_signal(model,param_vector,S0arr[ii],SNRarr[ii],acq_scheme,noise_type).simulate_noisy_signal()
    simulated_data_nonoise[ii,:] = simulate_signal.simulate_SNR_signal(model,param_vector,S0arr[ii],SNRarr[ii],acq_scheme,noise_type).simulate_true_signal()


  return simulated_data,parameter_array,parameter_names,simulated_data_nonoise


def ball(numofsim,acq_scheme,S0arr,SNRarr,noise_type): #Generate random array of ball parameters to then get our simulated signals

  numofacq = acq_scheme.number_of_measurements
  simulated_data = np.empty((numofsim,numofacq))
  simulated_data_nonoise = np.empty((numofsim,numofacq))
  noise_type = noise_type

  parameter_array,parameter_names = generate_model_parameter_array(numofsim).ball()

  for ii in range(numofsim):
    Diso = parameter_array[ii]

    model,param_vector = models.ball(Diso).make_model()
    simulated_data[ii,:] = simulate_signal.simulate_SNR_signal(model,param_vector,S0arr[ii],SNRarr[ii],acq_scheme,noise_type).simulate_noisy_signal()
    simulated_data_nonoise[ii,:] = simulate_signal.simulate_SNR_signal(model,param_vector,S0arr[ii],SNRarr[ii],acq_scheme,noise_type).simulate_true_signal()


  return simulated_data,parameter_array,parameter_names,simulated_data_nonoise


def stick(numofsim,acq_scheme,S0arr,SNRarr,noise_type): #Generate random array of stick parameters to then get our simulated signals

  numofacq = acq_scheme.number_of_measurements
  simulated_data = np.empty((numofsim,numofacq))
  simulated_data_nonoise = np.empty((numofsim,numofacq))
  noise_type = noise_type

  parameter_array,parameter_names = generate_model_parameter_array(numofsim).stick() 

  for ii in range(numofsim):
    mu = parameter_array[ii,0:2]
    Dpar = parameter_array[ii,2]

    model,param_vector = models.stick(mu,Dpar).make_model()
    simulated_data[ii,:] = simulate_signal.simulate_SNR_signal(model,param_vector,S0arr[ii],SNRarr[ii],acq_scheme,noise_type).simulate_noisy_signal()
    simulated_data_nonoise[ii,:] = simulate_signal.simulate_SNR_signal(model,param_vector,S0arr[ii],SNRarr[ii],acq_scheme,noise_type).simulate_true_signal()


  return simulated_data,parameter_array,parameter_names,simulated_data_nonoise


def ball_stick(numofsim,acq_scheme,S0arr,SNRarr,noise_type): 
  numofacq = acq_scheme.number_of_measurements
  simulated_data = np.empty((numofsim,numofacq))
  simulated_data_nonoise = np.empty((numofsim,numofacq))
  noise_type = noise_type

  parameter_array,parameter_names = generate_model_parameter_array(numofsim).ball_stick() 

  for ii in range(numofsim):
    mu = parameter_array[ii,0:2]
    Dpar = parameter_array[ii,2]
    Diso = parameter_array[ii,3]
    stickfrac = parameter_array[ii,4]
    ballfrac = parameter_array[ii,5]

    model,param_vector = models.ball_stick(mu,Dpar,stickfrac,Diso,ballfrac).make_model()
    simulated_data[ii,:] = simulate_signal.simulate_SNR_signal(model,param_vector,S0arr[ii],SNRarr[ii],acq_scheme,noise_type).simulate_noisy_signal()
    simulated_data_nonoise[ii,:] = simulate_signal.simulate_SNR_signal(model,param_vector,S0arr[ii],SNRarr[ii],acq_scheme,noise_type).simulate_true_signal()


  return simulated_data,parameter_array,parameter_names,simulated_data_nonoise


