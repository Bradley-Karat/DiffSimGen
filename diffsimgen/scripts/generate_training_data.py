from diffsimgen.scripts.generate_parameter_array import generate_model_parameter_array
from diffsimgen.scripts import models
from diffsimgen.scripts import simulate_signal
import numpy as np
from diffsimgen.scripts import my_murdaycotts
from diffsimgen.scripts import drchrnd
import scipy

def NODDI_watson(numofsim,acq_scheme,SNRarr,noise_type,parameter_distributions): #Generate random array of NODDI parameters to then get our simulated signals

  numofacq = acq_scheme.number_of_measurements
  simulated_data = np.empty((numofsim,numofacq))
  simulated_data_nonoise = np.empty((numofsim,numofacq))
  noise_type = noise_type
  parameter_distributions=parameter_distributions

  parameter_array,parameter_names = generate_model_parameter_array(numofsim,parameter_distributions).NODDI_watson() #numofsim x 6 array

  for ii in range(numofsim):

    mu = parameter_array[ii,0:2]
    ODI = parameter_array[ii,2]
    watsonstickfrac = parameter_array[ii,3]
    ballfrac = parameter_array[ii,4]
    watsonfrac = parameter_array[ii,5]

    model,param_vector = models.NODDI_watson(mu,ODI,watsonstickfrac,ballfrac,watsonfrac).make_model()
    simulated_data[ii,:] = simulate_signal.simulate_SNR_signal(model,param_vector,SNRarr[ii],acq_scheme,noise_type).simulate_noisy_signal()
    simulated_data_nonoise[ii,:] = simulate_signal.simulate_SNR_signal(model,param_vector,SNRarr[ii],acq_scheme,noise_type).simulate_true_signal()

  return simulated_data,parameter_array,parameter_names,simulated_data_nonoise


def SANDI(numofsim,bval,bvec,delta,Delta,SNRarr,noise_type,parameter_distributions): #Generate random array of NODDI parameters to then get our simulated signals

  Dsoma = 3
  Din_UB = 3
  De_UB = 3
  fsphere = lambda p,x: my_murdaycotts.my_murdaycotts(Delta,delta,p,Dsoma,x)
  Rsoma_UB = scipy.optimize.minimize((lambda p: np.abs(1 - fsphere(p,1))),10) # Automatically set the Rsoma to a max value given the diffusion time and the set Dsoma so to not exceed an apparent diffusivity = 1. This threshold has been chosen given realistic soma sizes and extracellular diffusivities. 
  Rsoma_UB = Rsoma_UB.x[0]

  paramsrange = np.array(([0,1],[0,1],[Dsoma/6,Din_UB],[1,Rsoma_UB],[Dsoma/6,De_UB]))

  parameter_distributions = np.zeros((numofsim,5))

  T = drchrnd.drchrnd([1,1,1],np.round(numofsim))
  parameter_distributions[:,0:2] = T[:,0:2]

  for i in range(2,5):
    parameter_distributions[:,i] = (paramsrange[i,1]-paramsrange[i,0])*np.random.uniform(size=(numofsim)) + paramsrange[i,0]

  parameter_array,parameter_names = generate_model_parameter_array(numofsim,parameter_distributions).SANDI() #numofsim x 6 array

  simulated_data = np.empty((numofsim,len(bval)))
  simulated_data_powder_avg = np.empty((numofsim,len(np.unique(bval))))
  simulated_data_nonoise = np.empty((numofsim,len(bval)))
  noise_type = noise_type

  fstick = lambda p,x,costheta: np.exp(-x*p*costheta**2) #np.exp(-x*p[0]*costheta**2)
  fsphere = lambda p,x: np.exp(-(my_murdaycotts.my_murdaycotts(Delta,delta,p,Dsoma,x)))
  fball = lambda p,x: np.exp(-p*x)

  fcomb = lambda p,x,costheta: p[0]*fstick(p[2],x,costheta) + p[1]*fsphere(p[3],x) + (1-p[0]-p[1])*fball(p[4],x)

  # Load bvals and bvecs
  bval = np.round(bval/100)*100
  bval[bval==0] = 1e-6
  bunique = np.unique(bval)

  Ndirs_per_shell = np.zeros((len(bunique),1))
  for i in range(len(bunique)):
      Ndirs_per_shell[i] = np.sum(bval == bunique[i])

  bvec = bvec/np.linalg.norm(bvec) #normalize bvecs
  bvec[np.isnan(bvec)] = 0

  phi = np.random.rand(numofsim,1)*2*np.pi # *2pi to get directions across whole sphere
  u = 2*np.random.rand(numofsim,1)-1 # cos(theta): ranging from -1 and +1
  term1 = np.sqrt(1-u**2)*np.cos(phi)
  term2 = np.sqrt(1-u**2)*np.sin(phi)

  fibre_orientation = np.transpose(np.concatenate((term1,term2,u),axis=1))

  costheta = np.zeros((numofsim,len(bval)))

  for ii in range(numofsim): # SANDI gets more manual signal simulation since not DIPY
    simfibre = np.tile(fibre_orientation[:,i],[len(bval),1])
    costheta[i,:] = (simfibre*np.transpose(bvec)).sum(1) #np.dot(simfibre,bvecs) - element-wise multiplication then sum across columnn

  for i in range(numofsim):
    simulated_data_nonoise[i,:] = fcomb(parameter_distributions[i,:], bval/1000, costheta[i,:])
    sigma = 1 / SNRarr[ii]
    if noise_type=='rician':
      real = simulated_data_nonoise[i,:] + np.random.normal(0, sigma, size=len(bval))
      imag = np.random.normal(0, sigma, size=len(bval))
      simulated_data[i,:] = np.sqrt(real**2 + imag**2)
    elif self.noise_type == 'gaussian':
      simulated_data[i,:] = simulated_data_nonoise[i,:] + np.random.normal(0, sigma, size=len(bval))

  for i in range(np.size(bunique)):
      simulated_data_powder_avg[:,i] = np.nanmean(simulated_data[:,bval==bunique[i]],1)

  return simulated_data,simulated_data_powder_avg,parameter_distributions,parameter_names,simulated_data_nonoise


def ball(numofsim,acq_scheme,SNRarr,noise_type,parameter_distributions): #Generate random array of ball parameters to then get our simulated signals

  numofacq = acq_scheme.number_of_measurements
  simulated_data = np.empty((numofsim,numofacq))
  simulated_data_nonoise = np.empty((numofsim,numofacq))
  noise_type = noise_type

  parameter_array,parameter_names = generate_model_parameter_array(numofsim,parameter_distributions).ball()

  for ii in range(numofsim):
    Diso = parameter_array[ii]

    model,param_vector = models.ball(Diso).make_model()
    simulated_data[ii,:] = simulate_signal.simulate_SNR_signal(model,param_vector,SNRarr[ii],acq_scheme,noise_type).simulate_noisy_signal()
    simulated_data_nonoise[ii,:] = simulate_signal.simulate_SNR_signal(model,param_vector,SNRarr[ii],acq_scheme,noise_type).simulate_true_signal()


  return simulated_data,parameter_array,parameter_names,simulated_data_nonoise


def stick(numofsim,acq_scheme,SNRarr,noise_type,parameter_distributions): #Generate random array of stick parameters to then get our simulated signals

  numofacq = acq_scheme.number_of_measurements
  simulated_data = np.empty((numofsim,numofacq))
  simulated_data_nonoise = np.empty((numofsim,numofacq))
  noise_type = noise_type

  parameter_array,parameter_names = generate_model_parameter_array(numofsim,parameter_distributions).stick() 

  for ii in range(numofsim):
    mu = parameter_array[ii,0:2]
    Dpar = parameter_array[ii,2]

    model,param_vector = models.stick(mu,Dpar).make_model()
    simulated_data[ii,:] = simulate_signal.simulate_SNR_signal(model,param_vector,SNRarr[ii],acq_scheme,noise_type).simulate_noisy_signal()
    simulated_data_nonoise[ii,:] = simulate_signal.simulate_SNR_signal(model,param_vector,SNRarr[ii],acq_scheme,noise_type).simulate_true_signal()


  return simulated_data,parameter_array,parameter_names,simulated_data_nonoise

def zeppelin(numofsim,acq_scheme,SNRarr,noise_type,parameter_distributions): #Generate random array of zeppelin parameters to then get our simulated signals

  numofacq = acq_scheme.number_of_measurements
  simulated_data = np.empty((numofsim,numofacq))
  simulated_data_nonoise = np.empty((numofsim,numofacq))
  noise_type = noise_type

  parameter_array,parameter_names = generate_model_parameter_array(numofsim,parameter_distributions).zeppelin() 

  for ii in range(numofsim):
    mu = parameter_array[ii,0:2]
    Dpar = parameter_array[ii,2]
    Dperp = parameter_array[ii,3]

    model,param_vector = models.zeppelin(mu,Dpar,Dperp).make_model()
    simulated_data[ii,:] = simulate_signal.simulate_SNR_signal(model,param_vector,SNRarr[ii],acq_scheme,noise_type).simulate_noisy_signal()
    simulated_data_nonoise[ii,:] = simulate_signal.simulate_SNR_signal(model,param_vector,SNRarr[ii],acq_scheme,noise_type).simulate_true_signal()

  return simulated_data,parameter_array,parameter_names,simulated_data_nonoise

def ball_stick(numofsim,acq_scheme,SNRarr,noise_type,parameter_distributions): 
  numofacq = acq_scheme.number_of_measurements
  simulated_data = np.empty((numofsim,numofacq))
  simulated_data_nonoise = np.empty((numofsim,numofacq))
  noise_type = noise_type

  parameter_array,parameter_names = generate_model_parameter_array(numofsim,parameter_distributions).ball_stick() 

  for ii in range(numofsim):
    mu = parameter_array[ii,0:2]
    Dpar = parameter_array[ii,2]
    Diso = parameter_array[ii,3]
    stickfrac = parameter_array[ii,4]
    ballfrac = parameter_array[ii,5]

    model,param_vector = models.ball_stick(mu,Dpar,stickfrac,Diso,ballfrac).make_model()
    simulated_data[ii,:] = simulate_signal.simulate_SNR_signal(model,param_vector,SNRarr[ii],acq_scheme,noise_type).simulate_noisy_signal()
    simulated_data_nonoise[ii,:] = simulate_signal.simulate_SNR_signal(model,param_vector,SNRarr[ii],acq_scheme,noise_type).simulate_true_signal()


  return simulated_data,parameter_array,parameter_names,simulated_data_nonoise
  
def zeppelin_stick(numofsim,acq_scheme,SNRarr,noise_type,parameter_distributions): 
  numofacq = acq_scheme.number_of_measurements
  simulated_data = np.empty((numofsim,numofacq))
  simulated_data_nonoise = np.empty((numofsim,numofacq))
  noise_type = noise_type

  parameter_array,parameter_names = generate_model_parameter_array(numofsim,parameter_distributions).zeppelin_stick() 

  for ii in range(numofsim):
    mu = parameter_array[ii,0:2]
    Dpar = parameter_array[ii,2]
    stickfrac = parameter_array[ii,3]

    model,param_vector = models.zeppelin_stick(mu,Dpar,stickfrac).make_model()
    simulated_data[ii,:] = simulate_signal.simulate_SNR_signal(model,param_vector,SNRarr[ii],acq_scheme,noise_type).simulate_noisy_signal()
    simulated_data_nonoise[ii,:] = simulate_signal.simulate_SNR_signal(model,param_vector,SNRarr[ii],acq_scheme,noise_type).simulate_true_signal()


  return simulated_data,parameter_array,parameter_names,simulated_data_nonoise


