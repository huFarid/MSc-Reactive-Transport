
from utils.cal_Perm import cal_K
from utils import config
# import numpy as np

def cal_P_Equation(i,P,por0):
    
    
    dx    = config.dx
    vis   = config.vis
    dt    = config.dt
    Q     = config.Q
    Ax    = config.Ax
    n     = config.n
    Patm  = config.Patm
    roi   = config.roi
    por0f = config.por0f
    
    # pp=vis*dx**2/dt;
    # Cr=np.copy(Cl);#
    
    
    B1=2*dx*Q*vis/cal_K(i,por0)/Ax ; ## 
    if i==0:
        P1=P[0,i+1]+B1;
        P2=P[0,i];
        P3=P[0,i+1];
    elif i==n-1:
        P1=P[0,i-1];
        P2=P[0,i];
        P3=Patm;
    elif i>0 and i<n-1:
        P1=P[0,i-1];
        P2=P[0,i];
        P3=P[0,i+1];
        
    TM=(-roi/vis/dx**2)*(2*cal_K(i,por0)*cal_K(i+1,por0))/(cal_K(i,por0)+cal_K(i+1,por0))
    Tm=(-roi/vis/dx**2)*(2*cal_K(i,por0)*cal_K(i-1,por0))/(cal_K(i,por0)+cal_K(i-1,por0))
    P_Equation=TM*(P3-P2)-Tm*(P2-P1)+roi*(por0[0,i]-por0f[0,i])/dt
    return P_Equation

'''##########################################################################################################'''
