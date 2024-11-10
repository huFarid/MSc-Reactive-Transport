
from utils.Derevative_C import cal_C_Derevative
from utils import config

import numpy as np
def cal_C_P_Jacobian(j,k,P,C,por0):
    n = config.n
    C_P_Jacobian=np.zeros([n,n])
 
    
    for i in range(1,n+1):# 
         
        ss=i-1; 
                     
        
        if i==1:
            C_P_Jacobian[ss,i-1]=cal_C_Derevative(j,k,i,i-1,P,C,por0,1);
            C_P_Jacobian[ss,i]=cal_C_Derevative(j,k,i,i,P,C,por0,1);
            C_P_Jacobian[ss,i+1]=cal_C_Derevative(j,k,i,i+1,P,C,por0,1);
            
        elif i>1 and i<n-1:
            C_P_Jacobian[ss,i-2]=cal_C_Derevative(j,k,i,i-2,P,C,por0,1);
            C_P_Jacobian[ss,i-1]=cal_C_Derevative(j,k,i,i-1,P,C,por0,1);
            C_P_Jacobian[ss,i]=cal_C_Derevative(j,k,i,i,P,C,por0,1);
            C_P_Jacobian[ss,i+1]=cal_C_Derevative(j,k,i,i+1,P,C,por0,1);
            
    
        elif i==n-1:
            C_P_Jacobian[ss,i-2]=cal_C_Derevative(j,k,i,i-2,P,C,por0,1);
            C_P_Jacobian[ss,i-1]=cal_C_Derevative(j,k,i,i-1,P,C,por0,1);
            C_P_Jacobian[ss,i]=cal_C_Derevative(j,k,i,i,P,C,por0,1);
    
        elif i==n:
           
            C_P_Jacobian[ss,i-2]=cal_C_Derevative(j,k,i,i-2,P,C,por0,1);
            C_P_Jacobian[ss,i-1]=cal_C_Derevative(j,k,i,i-1,P,C,por0,1);

    return C_P_Jacobian  