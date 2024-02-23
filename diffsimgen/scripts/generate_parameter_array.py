import numpy as np
import json
import os

pathname = os.path.dirname(os.path.abspath(__file__))

class generate_model_parameter_array:
    def __init__(self,numofsim):
        self.numofsim = numofsim

    def NODDI_watson(self):
        with open(f'{pathname}/../../resources/NODDI_watson_parameter_range.json', 'r') as f:
            param_range = json.load(f)
        theta = np.random.uniform(param_range['theta'][0],param_range['theta'][1],size=[self.numofsim,1])
        phi = np.random.uniform(param_range['phi'][0],param_range['phi'][1],size=[self.numofsim,1])
        mu = np.c_[theta,phi]
        ODI = np.random.uniform(param_range['odi'][0],param_range['odi'][1],size=[self.numofsim,1])
        #sampfrac = np.random.dirichlet(np.ones(2),size=[self.numofsim])
        #ballfrac = sampfrac[:,0]
        #watsonfrac = sampfrac[:,1]
        ballfrac = np.random.uniform(param_range['ballfrac'][0],param_range['ballfrac'][1],size=[self.numofsim,1])
        watsonfrac = 1 - ballfrac
        watsonstickfrac = np.random.uniform(param_range['watsonstickfrac'][0],param_range['watsonstickfrac'][1],size=[self.numofsim,1])

        parameter_names = ['theta','phi','ODI','stick_fraction_within_watson','ball_fraction','total_watson_fraction']
        parameter_array = np.c_[mu,ODI,watsonstickfrac,ballfrac,watsonfrac]

        return parameter_array,parameter_names

    def ball(self):
        with open(f'{pathname}/../../resources/ball_parameter_range.json', 'r') as f:
            param_range = json.load(f)
        Diso = np.random.uniform(param_range['Diso'][0],param_range['Diso'][1],size=[self.numofsim,1])

        parameter_names = ['Diso']
        parameter_array = Diso

        return parameter_array,parameter_names
    
    def stick(self):
        with open(f'{pathname}/../../resources/stick_parameter_range.json', 'r') as f:
            param_range = json.load(f)
        theta = np.random.uniform(param_range['theta'][0],param_range['theta'][1],size=[self.numofsim,1])
        phi = np.random.uniform(param_range['phi'][0],param_range['phi'][1],size=[self.numofsim,1])
        mu = np.c_[theta,phi]
        Dpar = np.random.uniform(param_range['Dpar'][0],param_range['Dpar'][1],size=[self.numofsim,1])

        parameter_names = ['theta','phi','Dpar']
        parameter_array = np.c_[mu,Dpar]

        return parameter_array,parameter_names

    def ball_stick(self):
        with open(f'{pathname}/../../resources/ball_stick_parameter_range.json', 'r') as f:
            param_range = json.load(f)
        theta = np.random.uniform(param_range['theta'][0],param_range['theta'][1],size=[self.numofsim,1])
        phi = np.random.uniform(param_range['phi'][0],param_range['phi'][1],size=[self.numofsim,1])
        mu = np.c_[theta,phi]
        Dpar = np.random.uniform(param_range['Dpar'][0],param_range['Dpar'][1],size=[self.numofsim,1])
        Diso = np.random.uniform(param_range['Diso'][0],param_range['Diso'][1],size=[self.numofsim,1])
        #sampfrac = np.random.dirichlet(np.ones(2),size=[self.numofsim])
        #ballfrac = sampfrac[:,0]
        #stickfrac = sampfrac[:,1]
        #stickfrac = np.random.uniform(param_range['stickfrac'][0],param_range['stickfrac'][1],size=[self.numofsim,1])
        ballfrac = np.random.uniform(param_range['ballfrac'][0],param_range['ballfrac'][1],size=[self.numofsim,1])
        stickfrac = 1-ballfrac

        parameter_names = ['theta','phi','Dpar','Diso','stick_fraction','ball_fraction']
        parameter_array = np.c_[mu,Dpar,Diso,stickfrac,ballfrac]

        return parameter_array,parameter_names