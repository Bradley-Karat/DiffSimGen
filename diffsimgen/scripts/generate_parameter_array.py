import numpy as np

class generate_model_parameter_array:
    def __init__(self,numofsim):
        self.numofsim = numofsim

    def NODDI_watson(self):
        theta = np.random.uniform(0,np.pi,size=[self.numofsim,1])
        phi = np.random.uniform(0,2*np.pi,size=[self.numofsim,1])
        mu = np.c_[theta,phi]
        ODI = np.random.uniform(0.001,1,size=[self.numofsim,1])
        ballfrac = np.random.uniform(0.001,1,size=[self.numofsim,1])
        watsonfrac = 1 - ballfrac
        watsonstickfrac = np.random.uniform(0.001,1,size=[self.numofsim,1])

        parameter_names = ['theta','phi','ODI','stick_fraction_within_watson','ball_fraction','total_watson_fraction']
        parameter_array = np.c_[mu,ODI,watsonstickfrac,ballfrac,watsonfrac]

        return parameter_array,parameter_names