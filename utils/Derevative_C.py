# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 17:07:20 2024

@author: hosyo
"""
import numpy as np

from utils.Equation_C import cal_C_Equation


def cal_C_Derevative(j,k,i,zz,P,C,por0,C_or_P):
    
    deP=0.00000001
    deC=0.00000000001
    de_phi=0.000000001
    
    Pplus=np.copy(P)
    Pmin=np.copy(P)
    
    Cplus=np.copy(C)
    Cmin=np.copy(C)

    por_plus=np.copy(por0)  
    por_min=np.copy(por0)

    if C_or_P==1:  # Pressure
    
        Pplus[0,zz]=Pplus[0,zz]+deP;
        Pmin[0,zz]=Pmin[0,zz]-deP;
        
        Derevative_C=(cal_C_Equation(j,i,Pplus,Cplus,por_plus)-cal_C_Equation(j,i,Pmin,Cmin,por_min))/2/deP;
    
    elif  C_or_P==2:  # Concentration
        Cplus[k,zz]=C[k,zz]+deC;
        Cmin[k,zz]=C[k,zz]-deC;
        Derevative_C=(cal_C_Equation(j,i,Pplus,Cplus,por_plus)-cal_C_Equation(j,i,Pmin,Cmin,por_min))/2/deC;
    elif C_or_P==3: # Porosity
        
        por_plus[zz]=por_plus[zz]+de_phi;
        por_min[zz]=por_min[zz]-de_phi;
        Derevative_C=(cal_C_Equation(j,i,Pplus,Cplus,por_plus)-cal_C_Equation(j,i,Pmin,Cmin,por_min))/2/de_phi;
        
    return Derevative_C


