import numpy as np
import dmipy
from contextlib import contextmanager
import os
import sys

class simulate_SNR_signal:
  def __init__(self,model,parametervector,SNR,acq_scheme,noise_type):
    self.model = model
    self.parametervector = parametervector
    self.SNR = SNR
    self.acq_scheme = acq_scheme
    self.noise_type = noise_type
  def simulate_true_signal(self):
    self.numofdata = self.acq_scheme.number_of_measurements
    self.simulated_data = np.empty((self.numofdata))
    self.simulated_data = self.model.simulate_signal(self.acq_scheme,self.parametervector) #multiply by self.S0 here if want signal 
    #on original scale defined by S0. Need to do this before adding noise
    return self.simulated_data
  def simulate_noisy_signal(self):
    self.numofdata = self.acq_scheme.number_of_measurements
    sigma = 1 / self.SNR #SNR = S0 / std(noise)
    if self.noise_type == 'rician':
      real = self.simulate_true_signal() + np.random.normal(0, sigma, size=self.numofdata)
      imag = np.random.normal(0, sigma, size=self.numofdata)
      self.noisy_signal = np.sqrt(real**2 + imag**2)
    elif self.noise_type == 'gaussian':
      self.noisy_signal = self.simulate_true_signal() + np.random.normal(0, sigma, size=self.numofdata)
    return self.noisy_signal

#class simulate_noisemap_signal:
 # def __init__(self,model,noisemap,acq_scheme):
 #   self.model = model
 #   self.SNR = SNR
 #   self.bvals = bvals
 #   self.bvecs = bvecs
 # def simulate_true_signal(self):
 #   self.numofdata = self.bvals.shape[0]
 #   self.simulated_data = np.empty((self.numofdata))
 #   for i in range(self.numofdata):
 #     self.simulated_data[i] = self.model.signal_representation(self.S0,self.bvals[i],self.bvecs[i,:])
 #   return self.simulated_data
 # def simulate_noisy_signal(self):
 #   self.numofdata = self.bvals.shape[0]
 #   sigma = self.S0 / self.SNR #SNR = S0 / std(noise)
 #   noise1 = np.random.normal(0, sigma, size=self.numofdata)
 #   self.noisy_signal = self.simulate_true_signal() + noise1
 #   return(self.noisy_signal)