import dmipy
from dmipy.signal_models import cylinder_models, gaussian_models
from dmipy.distributions.distribute_models import SD1WatsonDistributed
from dmipy.core.modeling_framework import MultiCompartmentModel
from contextlib import contextmanager
import os
import sys

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout

class NODDI_watson: #Input to class NODDI is all parameters needed to build the NODDI model
    def __init__(self,mu,ODI,watson_stick_frac,ball_frac,watson_frac,ball_diff=3e-9,par_free_diff=1.7e-9):
        '''
        ball_diff: fixed ball diffusivity (default=3e-9 m^2/s)
        mu: Orientation of watson distribution
        ODI: orientation dispersion index (between 0 and 1)
        watson_stick_frac: normalized fraction of stick w/i Watson bundle
        ball_frac: ball signal fraction
        watson_frac: fraction of watson component (combined intra- extracellular)
        par_free_diff: fixed parallel free diffusivity (default=1.7e-9 m^2/s)
        '''
        self.ball_diff = ball_diff
        self.mu = mu
        self.ODI = ODI
        self.watson_stick_frac = watson_stick_frac
        self.ball_frac = ball_frac
        self.watson_frac = watson_frac
        self.par_free_diff = par_free_diff

    def make_model(self):
        
        ball = gaussian_models.G1Ball()
        stick = cylinder_models.C1Stick()
        zeppelin = gaussian_models.G2Zeppelin()
        watson_bundle = SD1WatsonDistributed(models=[stick, zeppelin])

        watson_bundle.set_tortuous_parameter('G2Zeppelin_1_lambda_perp','C1Stick_1_lambda_par','partial_volume_0')
        watson_bundle.set_equal_parameter('G2Zeppelin_1_lambda_par', 'C1Stick_1_lambda_par')
        watson_bundle.set_fixed_parameter('G2Zeppelin_1_lambda_par', self.par_free_diff) 
        
        with suppress_stdout():
            NODDI_watson_model = MultiCompartmentModel(models=[ball, watson_bundle])
        
        NODDI_watson_model.set_fixed_parameter('G1Ball_1_lambda_iso', self.ball_diff)

        model_vector = NODDI_watson_model.parameters_to_parameter_vector(SD1WatsonDistributed_1_SD1Watson_1_mu=self.mu,
            SD1WatsonDistributed_1_SD1Watson_1_odi=self.ODI,SD1WatsonDistributed_1_partial_volume_0=self.watson_stick_frac, 
            partial_volume_0=self.ball_frac,partial_volume_1=self.watson_frac)
        
        return NODDI_watson_model,model_vector

class stick: 
    def __init__(self,mu,Dpar): #Orientation and parallel diffusivity
        '''
        mu: Orientation of stick
        Dpar: Stick parallel diffusivity
        '''
        self.mu = mu
        self.Dpar = Dpar

    def make_model(self):
        
        stick = cylinder_models.C1Stick()
        
        with suppress_stdout():
            stick_model = MultiCompartmentModel(models=[stick])
        
        model_vector = stick_model.parameters_to_parameter_vector(
            C1Stick_1_lambda_par=self.Dpar, C1Stick_1_mu=self.mu)
        
        return stick_model,model_vector

class ball: 
    def __init__(self,Diso): #Orientation and parallel diffusivity
        '''
        Diso: Ball isotropic diffusivity
        '''
        self.Diso = Diso

    def make_model(self):
        ball = gaussian_models.G1Ball()

        with suppress_stdout():
            ball_model = MultiCompartmentModel(models=[ball])
        
        model_vector = ball_model.parameters_to_parameter_vector(
            G1Ball_1_lambda_iso=self.Diso)
        
        return ball_model,model_vector

class ball_stick:
    def __init__(self,mu,Dpar,stick_frac,Diso,ball_frac):
        '''
        mu: Orientation of stick
        Dpar: Stick parallel diffusivity
        stick_frac: stick signal fraction
        Diso: Ball isotropic diffusivity
        ball_frac: ball signal fraction
        '''
        self.mu = mu
        self.Dpar = Dpar
        self.stick_frac = stick_frac
        self.Diso = Diso
        self.ball_frac = ball_frac

    def make_model(self):
        ball = gaussian_models.G1Ball()
        stick = cylinder_models.C1Stick()

        with suppress_stdout():
            ball_stick_model = MultiCompartmentModel(models=[ball,stick])
        
        model_vector = ball_stick_model.parameters_to_parameter_vector(C1Stick_1_mu=self.mu,
            C1Stick_1_lambda_par=self.Dpar, partial_volume_0=self.ball_frac,
            partial_volume_1=self.stick_frac,G1Ball_1_lambda_iso=self.Diso)
        
        return ball_stick_model,model_vector
