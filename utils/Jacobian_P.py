

import numpy as np
from utils.Derevative_P import cal_Derevative_P
from utils import config
## %%%%%%%%%%%%%%%   Pressure   %%%%%%%%%%%%
def cal_P_Jacobian(P,por0):
    n = config.n

    P_Jacobian=np.zeros([n,n])
    for i in range(n):   #% the last grid has constant P and will not be appeared in here
        ss=i;
        if i==0:
            
            P_Jacobian[ss,ss]=cal_Derevative_P(i,i,P,por0,1);
            P_Jacobian[ss,ss+1]=cal_Derevative_P(i,i+1,P,por0,1);
            #%                 a(i,kk)=fprim(i,i,P,por0,2);
        elif i>0 and i<n-1:
            
            P_Jacobian[ss,ss+1]=cal_Derevative_P(i,i+1,P,por0,1);
            P_Jacobian[ss,ss]=cal_Derevative_P(i,i,P,por0,1);
            P_Jacobian[ss,ss-1]=cal_Derevative_P(i,i-1,P,por0,1);
            #%                 a(i,kk)=fprim(i,i,P,por0,2);
        elif i==n-1:
            P_Jacobian[ss,ss]=cal_Derevative_P(i,i,P,por0,1);
            P_Jacobian[ss,ss-1]=cal_Derevative_P(i,i-1,P,por0,1);
    return P_Jacobian 
