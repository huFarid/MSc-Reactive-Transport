import numpy as np
from utils import config
# Function that represents the distance from equilibrium for the acetic acid in initial state before injection into the system
def AcidDissociation_Function(x_molal_H, keq_acid, Macid):
   
    
    I=0.5*(x_molal_H+x_molal_H) # Ionic activity : Hydrogen and Acetate (same molarity)
    gama1=np.exp((-0.51*(I**0.5))/(1+0.3294*9*(I**0.5))) # activity coefficient for hydrogen
    gama2=np.exp((-0.51*(I**0.5))/(1+0.3294*4.5*(I**0.5))) # activity coefficient of acetate

    difference=(keq_acid)-gama1*gama2*(x_molal_H**2)/(Macid - x_molal_H) # representative of distance from equilibrium concentrations
    config.gama1 = gama1
    
    return(difference)