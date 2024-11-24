
import numpy as np
from utils import config
from utils.cal_Perm import cal_K
def cal_C_Equation(j,i,P,C,por0): 
    # global KK,rme,endM,neg_rme,Xi,km,R,rmef,rmeb,k123,LandaP,LandaS,k123,Xinj,k1,k2,k3,IAP
    
    KK         = config.KK
    k1         = config.k1
    k2         = config.k2
    k3         = config.k3
    IAP        = config.IAP
    Xinj       = config.Xinj
    k123       = config.k123
    LandaS     = config.LandaS
    LandaP     = config.LandaP
    R          = config.R
    rmef       = config.rmef
    rmeb       = config.rmeb
    rme        = config.rme
    Xi         = config.Xi
    dt         = config.dt
    por0f      = config.por0f
    Pformer    = config.Pformer
    Uformer    = config.Uformer
    # As         = config.As
    Cinj       = config.Cinj
    n          = config.n
    Cl         = config.Cl
    npr        = config.npr
    nMineral   = config.nMineral
    Keq        = config.Keq
    cunit      = config.cunit
    vrj        = config.vrj
    Ksolid     = config.Ksolid
    vmi        = config.vmi
    vsolid     = config.vsolid
    ns         = config.ns
    vrjf       = config.vrjf
    vrjb       = config.vrjb
    Nvf        = config.Nvf
    Nvb        = config.Nvb
    dx         = config.dx
    Q          = config.Q
    vis        = config.vis
    Ax         = config.Ax
    aa         = config.aa
    uinj       = config.uinj
    bb         = config.bb
    Pi         = config.Pi
    Patm       = config.Patm
    km         = config.km
    por0       = config.por0

    
    Cl = config.Cl
    
    Cr=np.copy(Cl);

    U=np.zeros([npr,n+1]);

    config.R=np.zeros([npr,n+1])
    
    
    if nMineral>0:   # is there any hetro reaction or not?

        rm=[]

        ''' P: 'H+',    'Ca2+' ,    'H2CO3*' ,CH3COO- '''
        ''' S:  CH3COOH,  CO3--,  HCO3-,  OH-    '''     

        aH     = (abs(LandaP[0,i]*C[0,i]/cunit))
        aCa    = (abs(LandaP[1,i]*C[1,i]/cunit))
        aH2CO3 = (abs(LandaP[2,i]*C[2,i]/cunit))
        
        IAP    =aH2CO3   * aCa / (aH**2)
        
        k1=k123[0]* aH     ** k123[3]
        k2=k123[1]* aH2CO3 ** k123[4]
        k3=k123[2]

        KK[0,i]=np.copy(k1)
        KK[1,i]=np.copy(k2)
        KK[2,i]=np.copy(k3)
        KK[3,i]=np.copy(IAP) 
        
        
        
        kAll=k1+k2+k3
        rNet=kAll * ( 1 - IAP / Ksolid)  
        
        rm=np.array([rNet])

        rmi=(vrj[:,j]*((rm*(vmi.transpose())).sum(1))).sum()

        rme[:,i]=np.copy(rm);


        config.R[j,i]=(sum(vsolid[:,j]*rme[:,i])+rmi);   
  
    elif nMineral==0:
        config.R[j,i]=0
    
    if ns==0:
        U=np.copy(C);
    elif ns>0:
        U[:,0]=Cinj+(vrj.transpose()*Xinj).sum(axis=1)
        
        if i==1:
            
            for ee in range(i,i+2):
                
                Xee=config.Xi[:,ee]
                Xnon0=[1 if wq!=0 else 0 for wq in Xee]
                Ncf=(((vrjf*C[:,ee])!=0).sum(1))+Xnon0 
                Ncb=((vrjb*C[:,ee])!=0).sum(1)         
                for react in range(ns):
                    if Ncf[react]==Nvf[react] or Ncb[react]==Nvb[react]:
                        config.Xi[react,ee]=(cunit/Keq[react]/LandaS[react,ee])*np.prod((C[:,ee]*LandaP[:,ee]/cunit)**vrj[react,:].transpose())     
                    zigma=0
                    cal=vrj.transpose()*config.Xi[:,ee]
                    zigma=cal.sum(axis=1)
                    U[:,ee]=C[:,ee]+zigma
          
        elif (i>1) and (i<n):
            for ee in range(i-1,i+2):
                
                Xee=config.Xi[:,ee]
                Xnon0=[1 if wq!=0 else 0 for wq in Xee]
                Ncf=(((vrjf*C[:,ee])!=0).sum(1))+Xnon0 
                Ncb=((vrjb*C[:,ee])!=0).sum(1)         
                for react in range(ns):
                    if Ncf[react]==Nvf[react] or Ncb[react]==Nvb[react]:
                        config.Xi[react,ee]=(cunit/Keq[react]/LandaS[react,ee])*np.prod((C[:,ee]*LandaP[:,ee]/cunit)**vrj[react,:].transpose())      
                    zigma=0
                    cal=vrj.transpose()*config.Xi[:,ee]
                    zigma=cal.sum(axis=1)
                    U[:,ee]=C[:,ee]+zigma

        elif i==n:
            for ee in range(i-1,i+1):
                Xee=config.Xi[:,ee]
                Xnon0=[1 if wq!=0 else 0 for wq in Xee]
                Ncf=(((vrjf*C[:,ee])!=0).sum(1))+Xnon0 
                Ncb=((vrjb*C[:,ee])!=0).sum(1)         
                for react in range(ns):
                    if Ncf[react]==Nvf[react] or Ncb[react]==Nvb[react]:
                        config.Xi[react,ee]=(cunit/Keq[react]/LandaS[react,ee])*np.prod((C[:,ee]*LandaP[:,ee]/cunit)**vrj[react,:].transpose())      
                    zigma=0          
                    cal=vrj.transpose()*config.Xi[:,ee]
                    zigma=cal.sum(axis=1)
                    U[:,ee]=C[:,ee]+zigma     
                    
    B1=2*dx*Q*vis/cal_K(0,por0)/Ax ; ## %
    if i==1:
        P1=P[0,i]+B1;
        P2=P[0,i-1];
        P3=P[0,i];
        P4=P[0,i+1];
        C1=U[j,i-1];
        C2=U[j,i];
        C3=U[j,i+1];
        KKM=4* cal_K(i-1,por0)*cal_K(i,por0)*cal_K(i+1,por0)/( cal_K(i,por0)* cal_K(i+1,por0)+ cal_K(i,por0)* cal_K(i-1,por0)+2* cal_K(i+1,por0)* cal_K(i-1,por0))

        C_Equation=aa*(C2*KKM*(P4-P2))-uinj*C1/dx+bb*(C3-2*C2+C1)+\
        C2*(por0[0,i]*(1+Cr*(P3-Pi)))/dt-por0f[0,i]*(1+Cr*(Pformer[0,i]-Pi))*Uformer[j,i]/dt-config.R[j,i];
        

    elif (i>1) and (i<n-1):
        
        P1=P[0,i-2];
        P2=P[0,i-1];
        P3=P[0,i];
        P4=P[0,i+1];
        
        C1=U[j,i-1];
        C2=U[j,i];
        C3=U[j,i+1];
        
        KKM=4* cal_K(i-1,por0)* cal_K(i,por0)* cal_K(i+1,por0)/( cal_K(i,por0)* cal_K(i+1,por0)+ cal_K(i,por0)* cal_K(i-1,por0)+2* cal_K(i+1,por0)* cal_K(i-1,por0))
        KKm=4* cal_K(i-2,por0)* cal_K(i-1,por0)* cal_K(i,por0)/( cal_K(i-1,por0)* cal_K(i,por0)+ cal_K(i-1,por0)* cal_K(i-2,por0)+2* cal_K(i,por0)* cal_K(i-2,por0))
        C_Equation=aa*(C2*KKM*(P4-P2)-C1*KKm*(P3-P1))+bb*(C3-2*C2+C1)+\
        C2*(por0[0,i]*(1+Cr*(P3-Pi)))/dt-por0f[0,i]*(1+Cr*(Pformer[0,i]-Pi))*Uformer[j,i]/dt-config.R[j,i];

    elif i==n-1:
        P1=P[0,i-2];
        P2=P[0,i-1];
        P3=P[0,i];
        P4=Patm;
        C1=U[j,i-1];
        C2=U[j,i];
        C3=U[j,i+1];
        KKM=4* cal_K(i-1,por0)* cal_K(i,por0)* cal_K(i+1,por0)/( cal_K(i,por0)* cal_K(i+1,por0)+ cal_K(i,por0)* cal_K(i-1,por0)+2* cal_K(i+1,por0)* cal_K(i-1,por0))
        KKm=4* cal_K(i-2,por0)* cal_K(i-1,por0)* cal_K(i,por0)/( cal_K(i-1,por0)* cal_K(i,por0)+ cal_K(i-1,por0)* cal_K(i-2,por0)+2* cal_K(i,por0)* cal_K(i-2,por0))
        C_Equation=aa*(C2*KKM*(P4-P2)-C1*KKm*(P3-P1))+bb*(C3-2*C2+C1)+\
        C2*(por0[0,i]*(1+Cr*(P3-Pi)))/dt-por0f[0,i]*(1+Cr*(Pformer[0,i]-Pi))*Uformer[j,i]/dt-config.R[j,i];        

    elif i==n:
        
        P1=P[0,i-2];
        P2=P[0,i-1];
        P3=P[0,i];
        P4=2*P[0,i]-P[0,i-1];
        
        C1=U[j,i-1];
        C2=U[j,i];
        C3=U[j,i-1];
        
        KKM=2* cal_K(i-1,por0)* cal_K(i,por0)/( cal_K(i,por0)+ cal_K(i-1,por0))
        KKm=4* cal_K(i-2,por0)* cal_K(i-1,por0)* cal_K(i,por0)/( cal_K(i-1,por0)*cal_K(i,por0)+ cal_K(i-1,por0)*cal_K(i-2,por0)+2* cal_K(i,por0)* cal_K(i-2,por0))
        C_Equation=aa*(C2*KKM*(P4-P2)-C1*KKm*(P3-P1))+bb*(C3-2*C2+C1)+\
        C2*(por0[0,i]*(1+Cr*(P3-Pi)))/dt-por0f[0,i]*(1+Cr*(Pformer[0,i]-Pi))*Uformer[j,i]/dt- config.R[j,i];

    if i>0:
        return C_Equation