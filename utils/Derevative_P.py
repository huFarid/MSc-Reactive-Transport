
import numpy as np

from utils.Equation_P import cal_P_Equation

def cal_Derevative_P(i,zz,P,por0,P_or_phi):
    
    deP=0.00000001;
    de_por=0.000000001;
    Pmin=np.copy(P)
    Pplus=np.copy(P)
    por_plus=por0;   
    por_min=por0;

    if P_or_phi==1:

        Pmin[0,zz]=Pmin[0,zz]-deP
        Pplus[0,zz]=Pplus[0,zz]+deP
        
        P_Derevative=(cal_P_Equation(i,Pplus,por_plus)-cal_P_Equation(i,Pmin,por_min))/2/deP;
   
    elif P_or_phi==2:
    
        por_plus[zz]=por_plus[zz]+de_por;   
        por_min[zz]=por_min[zz]-de_por;
        P_Derevative=(cal_P_Equation(i,Pplus,por_plus)-cal_P_Equation(i,Pmin,por_min))/2/de_por;
    
    return(P_Derevative)