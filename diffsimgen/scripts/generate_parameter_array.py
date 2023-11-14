import numpy as np
import json

class generate_model_parameter_array:
    def __init__(self,numofsim):
        self.numofsim = numofsim

    def NODDI_watson(self):
        with open('resources/NODDI_watson_parameter_range.json', 'r') as f:
            param_range = json.load(f)
        theta = np.random.uniform(param_range['theta'][0],param_range['theta'][1],size=[self.numofsim,1])
        phi = np.random.uniform(param_range['phi'][0],param_range['phi'][1],size=[self.numofsim,1])
        mu = np.c_[theta,phi]
        ODI = np.random.uniform(param_range['odi'][0],param_range['odi'][1],size=[self.numofsim,1])
        ballfrac = np.random.uniform(param_range['ballfrac'][0],param_range['ballfrac'][1],size=[self.numofsim,1])
        watsonfrac = 1 - ballfrac
        watsonstickfrac = np.random.uniform(param_range['watsonstickfrac'][0],param_range['watsonstickfrac'][1],size=[self.numofsim,1])

        parameter_names = ['theta','phi','ODI','stick_fraction_within_watson','ball_fraction','total_watson_fraction']
        parameter_array = np.c_[mu,ODI,watsonstickfrac,ballfrac,watsonfrac]

        return parameter_array,parameter_names
