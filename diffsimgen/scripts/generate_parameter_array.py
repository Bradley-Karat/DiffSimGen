import numpy as np
import json
import os

pathname = os.path.dirname(os.path.abspath(__file__))

class generate_model_parameter_array:
    def __init__(self,numofsim,parameter_distributions):
        self.numofsim = numofsim
        self.parameter_distributions = parameter_distributions
    
    def NODDI_watson(self):
        if len(self.parameter_distributions)==0: #no parameter distribution provided, sample randomly
            
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

        elif len(self.parameter_distributions)!=0: #use provided parameter distribution
            theta = np.random.choice(np.squeeze(self.parameter_distributions[:,0]),size=[self.numofsim,1])
            phi = np.random.choice(np.squeeze(self.parameter_distributions[:,1]),size=[self.numofsim,1])
            mu = np.c_[theta,phi]
            ODI = np.random.choice(np.squeeze(self.parameter_distributions[:,2]),size=[self.numofsim,1])
            ODI[ODI==0] = 0.01
            watsonstickfrac = np.random.choice(np.squeeze(self.parameter_distributions[:,3]),size=[self.numofsim,1])
            watsonstickfrac[watsonstickfrac==0] = 0.01
            ballfrac = np.random.choice(np.squeeze(self.parameter_distributions[:,4]),size=[self.numofsim,1])
            ballfrac[ballfrac==0] = 0.01
            watsonfrac = 1 - ballfrac

            parameter_names = ['theta','phi','ODI','stick_fraction_within_watson','ball_fraction','total_watson_fraction']
            parameter_array = np.c_[mu,ODI,watsonstickfrac,ballfrac,watsonfrac]

        return parameter_array,parameter_names

    def ball(self):
        if len(self.parameter_distributions)==0: #no parameter distribution provided, sample randomly

            with open(f'{pathname}/../../resources/ball_parameter_range.json', 'r') as f:
                param_range = json.load(f)
            Diso = np.random.uniform(param_range['Diso'][0],param_range['Diso'][1],size=[self.numofsim,1])

            parameter_names = ['Diso']
            parameter_array = Diso

        elif len(self.parameter_distributions)!=0: #use provided parameter distribution

            Diso = np.random.choice(np.squeeze(self.parameter_distributions[:,0]),size=[self.numofsim,1])
            
            parameter_names = ['Diso']
            parameter_array = Diso
        
        return parameter_array,parameter_names
    
    def stick(self):
        if len(self.parameter_distributions)==0: #no parameter distribution provided, sample randomly

            with open(f'{pathname}/../../resources/stick_parameter_range.json', 'r') as f:
                param_range = json.load(f)
            theta = np.random.uniform(param_range['theta'][0],param_range['theta'][1],size=[self.numofsim,1])
            phi = np.random.uniform(param_range['phi'][0],param_range['phi'][1],size=[self.numofsim,1])
            mu = np.c_[theta,phi]
            Dpar = np.random.uniform(param_range['Dpar'][0],param_range['Dpar'][1],size=[self.numofsim,1])

            parameter_names = ['theta','phi','Dpar']
            parameter_array = np.c_[mu,Dpar]

        elif len(self.parameter_distributions)!=0: #use provided parameter distribution

            theta = np.random.choice(np.squeeze(self.parameter_distributions[:,0]),size=[self.numofsim,1])
            phi = np.random.choice(np.squeeze(self.parameter_distributions[:,1]),size=[self.numofsim,1])
            mu = np.c_[theta,phi]
            Dpar = np.random.choice(np.squeeze(self.parameter_distributions[:,2]),size=[self.numofsim,1])

            parameter_names = ['theta','phi','Dpar']
            parameter_array = np.c_[mu,Dpar]

        return parameter_array,parameter_names
    
    def zeppelin(self):
        if len(self.parameter_distributions)==0:

            ###### CHECK THE PARAMETER RANGE FILE CONTENTS - should Dperp be same range a Dpar?? ######
            with open(f'{pathname}/../../resources/zeppelin_parameter_range.json', 'r') as f:
                param_range = json.load(f)
            theta = np.random.uniform(param_range['theta'][0],param_range['theta'][1],size=[self.numofsim,1])
            phi = np.random.uniform(param_range['phi'][0],param_range['phi'][1],size=[self.numofsim,1])
            mu = np.c_[theta,phi]

            # sample two diffusivities
            Dpar = np.random.uniform(param_range['Dpar'][0],param_range['Dpar'][1],size=[self.numofsim,1])
            Dperp = np.random.uniform(param_range['Dperp'][0],param_range['Dperp'][1],size=[self.numofsim,1])
            D = np.c_[Dpar,Dperp]

            # Enforce constraint Dpar > Dperp
            Dpar = np.max(D, axis=1)
            Dperp = np.min(D, axis=1)

            parameter_names = ['theta','phi','Dpar','Dperp']
            parameter_array = np.c_[mu,Dpar,Dperp]

        elif len(self.parameter_distributions)!=0: #use provided parameter distribution

            theta = np.random.choice(np.squeeze(self.parameter_distributions[:,0]),size=[self.numofsim,1])
            phi = np.random.choice(np.squeeze(self.parameter_distributions[:,1]),size=[self.numofsim,1])
            mu = np.c_[theta,phi]
            Dpar = np.random.choice(np.squeeze(self.parameter_distributions[:,2]),size=[self.numofsim,1])
            Dperp = np.random.choice(np.squeeze(self.parameter_distributions[:,3]),size=[self.numofsim,1])
            D = np.c_[Dpar,Dperp]

            # Enforce constraint Dpar > Dperp
            Dpar = np.max(D, axis=1)
            Dperp = np.min(D, axis=1)

            parameter_names = ['theta','phi','Dpar','Dperp']
            parameter_array = np.c_[mu,Dpar,Dperp]

        return parameter_array,parameter_names

    def ball_stick(self):
        if len(self.parameter_distributions)==0: #no parameter distribution provided, sample randomly

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

        elif len(self.parameter_distributions)!=0: #use provided parameter distribution

            theta = np.random.choice(np.squeeze(self.parameter_distributions[:,0]),size=[self.numofsim,1])
            phi = np.random.choice(np.squeeze(self.parameter_distributions[:,1]),size=[self.numofsim,1])
            mu = np.c_[theta,phi]
            Dpar = np.random.choice(np.squeeze(self.parameter_distributions[:,2]),size=[self.numofsim,1])
            Diso = np.random.choice(np.squeeze(self.parameter_distributions[:,3]),size=[self.numofsim,1])
            ballfrac = np.random.choice(np.squeeze(self.parameter_distributions[:,4]),size=[self.numofsim,1])
            ballfrac[ballfrac==0] = 0.01
            stickfrac = 1-ballfrac

            parameter_names = ['theta','phi','Dpar','Diso','stick_fraction','ball_fraction']
            parameter_array = np.c_[mu,Dpar,Diso,stickfrac,ballfrac]

        return parameter_array,parameter_names