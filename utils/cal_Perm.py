
from utils import config
def cal_K(i,por0):
    beta = 5;
    
    
    K0 = config.K0
    Porosity_i = config.Porosity_i
    
    if i==-1:
        o1=Porosity_i
        o2=por0[0,0]
    else:
        o1=Porosity_i
        o2=por0[0,i]

    permeability=(K0/o1)*o2*(o2*(1-o1)/o1/(1-o2))**(2*beta)# 

    return permeability
