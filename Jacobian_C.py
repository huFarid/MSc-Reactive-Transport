

import numpy as np
from utils.Derevative_C import cal_C_Derevative
from utils import config
def cal_C_Jacobian(j,k,P,C,por0):
    n = config.n
    C_Jacobian=np.zeros([n,n])

    
    for i in range(1,n+1):
        ss=i-1
        
        if i==1:
            C_Jacobian[ss,ss]=cal_C_Derevative(j,k,i,i,P,C,por0,2)
            C_Jacobian[ss,ss+1]=cal_C_Derevative(j,k,i,i+1,P,C,por0,2)
        
        elif i>1 and i<n:
            
            C_Jacobian[ss,ss-1]=cal_C_Derevative(j,k,i,i-1,P,C,por0,2)
            C_Jacobian[ss,ss]=cal_C_Derevative(j,k,i,i,P,C,por0,2)
            C_Jacobian[ss,ss+1]=cal_C_Derevative(j,k,i,i+1,P,C,por0,2)

        elif i==n:
            C_Jacobian[ss,ss-1]=cal_C_Derevative(j,k,i,i-1,P,C,por0,2)
            C_Jacobian[ss,ss]=cal_C_Derevative(j,k,i,i,P,C,por0,2)
    
    return C_Jacobian 