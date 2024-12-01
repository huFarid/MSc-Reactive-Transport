"""
@author: Hossein
"""
    
# ----------------------------------------------------------------------------------------
# Import the external functions
# ----------------------------------------------------------------------------------------

import time
import numpy as np
import pickle

from utils.Jacobian_C import cal_C_Jacobian
from utils.Jacobian_P import cal_P_Jacobian
from utils.Jacobian_C_P import cal_C_P_Jacobian
from utils.Equation_C import cal_C_Equation
from utils.Equation_P import cal_P_Equation
from utils.cal_Perm import cal_K
from utils import config


N_Exp_Data=10;



# ----------------------------------------------------------------------------------------
# Initialize Variables and Begin Jacobian Calculation
# ----------------------------------------------------------------------------------------



def f_landa(x_molal_H):
    global Macid,keq_acid,gama1
    fff=1
    I=0.5*(x_molal_H+x_molal_H)*fff
    gama1=np.exp((-fff*0.51*(I**0.5))/(1+0.3294*9*(I**0.5)))
    gama2=np.exp((-fff*0.51*(I**0.5))/(1+0.3294*4.5*(I**0.5)))

    numm=(keq_acid)-gama1*gama2*(x_molal_H**2)/(Macid-x_molal_H)
    
    return(numm)
'''-----------------------------------------------------------------------------------------------------------'''

 
def fprim_landa(x_molal_H):
    
    dx=0.00000001;
    xmin=x_molal_H-dx
    xplus=x_molal_H+dx
    
    prim=(f_landa(xplus)-f_landa(xmin))/2/dx;
    
    return(prim)    
    
'''-----------------------------------------------------------------------------------------------------------'''
# ----------------------------------------------------------------------------------------
# Initialize the variables
# ----------------------------------------------------------------------------------------


StartTime=time.time();
dict_all_data={}
dict_all_data_necessary={}
    
i_kRate = 300
''' Field Scale Parameters '''

# ----------------------------------------------------------------------------------------
# Defining the domain dimentions
# ----------------------------------------------------------------------------------------


kInjectionRate = 5;        # gal/min
kArea          = 1.053589; # m2
kLength        = 141;      # m
k_Dt1          = 0.01;     # min
K_CriticalDt   = 10;       # Maximum value of time step
n              = 70;       # number of grid blocks - 1







lll     = 0
NPV     = 2   # number of the pore volume of fluid to be injected
NPV_NaCl= 10**1000
pvs     = 13   # started point for  tuning   0,1,2,3,....
pvf     = 30  #  


L       = kLength*100; #
dx      = L/n;
Ax      = kArea*10000#



Porosity_i = 0.39
volfrac0   = np.array([1 - Porosity_i])
MV_m       = np.array([36.93 * 10**-6])

dt1        = k_Dt1 * 60       
dtc        = K_CriticalDt * 60 # the upper limit for dt
orderdt    = 2                # we will multiply dt with this whenever necessary
perdt      = 5                #
dt_reduce  = orderdt          # when the Concentration is negative, we decrease dt value
ncor       = 1
cR         = 1                # convert mol/m3/s to mmole/cm3/s in rate equation


ns   = 8      # number of secondary species
nMineral = 1  # number of the minerals.   (It is connected to As, volfrac, vsolid,...)
npr  = 4      # number of primary species
MaxRem = 10**10
nR     = ns+nMineral
cunit  = 1000  # convert mol/lit   into  mmol/lit

''' P: 'H+',    'Ca2+' ,    'H2CO3*' ,CH3COO- '''

C0=[10**-7,0,0.00039,0]

Nhydro0=0
C0[:]=[number*cunit for number in C0]   #''' Convert to mmole per litr
  

# ----------------------------------------------------------------------------------------
# Parameters related to the reaction network
# ----------------------------------------------------------------------------------------

const_1 = 2900


k_1     = 10**-1.08
k_2     = 10**-3.96
k_3     = 10**-4.82
k_4     = 1
k_5     = 1

k123=[const_1* k_1  ,  const_1*k_2   ,   const_1*k_3  , k_4,  k_5] # Reaction rate constants




km=[0.001,0.001,0.001,1]; # 

Ksolid=10**8.23


#acetic
''' P: 'H+',    'Ca2+' ,    'H2CO3*' ,CH3COO- '''
''' S:  CH3COOH,  CO3--,  HCO3-,  OH- ,Ca(Acet)- , CaCO3 (aq), Ca(OH)+,  Ca(HCO3)-  '''
'''  1            ,   2           , 3       , 4         , 5      , 6         ,  7        ,  8 '''  
Keq=[10**-4.757   ,   10**16.67   , 10**6.34, 10**13.99 , 10**-0.77 , 10**13.35 ,  10**12.85,  10**5.3]

''' Homogeneous Reactions'''

''' P:            'H+',    'Ca2+' ,'H2CO3*' ,CH3COO- '''
Stoichiometry=np.asarray([[    1 ,    0,      0  ,   1],  #1 CH3COOH
                [   -2 ,    0,      1  ,   0],  #2 CO3--
                [   -1 ,    0,      1  ,   0],  #3 HCO3-
                [   -1 ,    0,      0  ,   0],  #4 OH-
#                [    0 ,    1,      0  ,   2],  #5 Ca(Acet)2
                [    0 ,    1,      0  ,   1],  #6 Ca(Acet)-
                [   -2 ,    1,      1  ,   0],  #7 CaCO3 (aq)
                [   -1 ,    1,      0  ,   0],  #8 Ca(OH)+
                [   -1 ,    1,      1  ,   0]]) #9 Ca(HCO3)-
Nforacid = 0   
' The first form # x2 + (Keq+H0)*x - Keq*Macid=0'

x_molal_H       = 0.004          # first geuss
keq_acid        = Keq[Nforacid]
density_of_acid = 1050           #g/Lit
wp_acid         = 5              # 
M_weight_acid   = 60.052         # g/mol
Macid           = (wp_acid/M_weight_acid)/((100-wp_acid)/1000) #

criteria=True
while (criteria==True):
    aa1=f_landa(x_molal_H)
    aa2=fprim_landa(x_molal_H)
    
    x_molal_H_new=x_molal_H-aa1/aa2
    delta=x_molal_H_new-x_molal_H
    if abs(delta)<0.00000001:
        criteria=False
    else:
        x_molal_H=x_molal_H_new

xPV=[]

inletpH=-np.log10(x_molal_H*gama1)

''' S:  CH3COOH,  CO3--,  HCO3-,  OH- ,Ca(Acet)- , CaCO3 (aq), Ca(OH)+,  Ca(HCO3)-  '''
X0     = np.array([0,0,0,10**-7,0.,0,0,0])
Xinj   = np.array([Macid-x_molal_H,0,0,10**-7,0.,0.,0.,0.])
X0[:]  = [number*cunit for number in X0]
Xinj[:]= [number*cunit for number in Xinj]


''' P: 'H+',    'Ca2+' ,    'H2CO3*' ,CH3COO- '''
Cinj   = np.array([x_molal_H,0,0,x_molal_H])
Cinj[:]= [number*cunit for number in Cinj]   


vB=Ax*dx;

density = [2.71];
M       = [100.09];
Clast   = np.zeros([npr,1])
Xilast  = np.zeros([ns,1])
As0     = volfrac0*[10**4]

K0 =2.3;    # % [darcy]

As      = np.ones([nMineral,n+1]);
volfrac = np.ones([nMineral,n+1]);


for ee in range(n+1):
    As[:,ee]=As0;
    volfrac[:,ee]=volfrac0;

ntotal=ns+npr

''' P:      'H+',    'Ca2+' ,    'H2CO3*' ,CH3COO- '''
vsolid=np.array([[ -2  ,  1    ,   1,  0 ]])


''' S:  CH3COOH,  CO3--,  HCO3-,  OH- ,Ca(Acet)- , CaCO3 (aq), Ca(OH)+,  Ca(HCO3)-  '''
vmi=np.array([[ 0. , 0., 0.,0.,0.,0.,0.,0.]]) 
    
vmif=np.copy(abs((vmi-abs(vmi))/2))
vmib=np.copy(abs((vmi+abs(vmi))/2))

vf=np.zeros(vsolid.shape);
vf=np.copy(abs((vsolid-abs(vsolid))/2))
vb=np.copy(abs((vsolid+abs(vsolid))/2))

Nvf=np.zeros((0,ns))
Nvb=np.zeros((0,ns))

vrjf=np.copy(abs((Stoichiometry-abs(Stoichiometry))/2))
vrjb=np.copy((Stoichiometry+abs(Stoichiometry))/2)

Nvf=((vrjf!=0).sum(1))+1
Nvb=(vrjb!=0).sum(1)

vm=np.array([[1]]); #

C=np.arange(npr*(n+1)).reshape((npr,n+1))
C=C*[0.]

Cnd=np.zeros((npr*n,npr,n))
for yy in range(npr*n):
    Cnd[yy,]

for ee in range(n+1):
    C[:,ee]=C0
C[:,0]=Cinj


D  = 5*10**-5; #   % Diffusion coefficient [cm2/s]

Cl = 46.4*10**-6; 

por0=np.ones([1,n+1])*Porosity_i;

permP = []
vis   = 1;      #   % cp
Pi    = 2175;      # % initial Pressure   [atm]
Patm  = 2175;


roi = 1;    #% initial density  [g/cm3]
ro  = np.ones([1,n+1])*roi;

P=np.ones([1,n+1])*Pi;

aa=-1/vis/dx**2/2;
bb=-D/dx**2;


Pformer = np.copy(P);

Cformer = np.copy(C);
por0f   = np.copy(por0);
U       = np.zeros(C.shape)
Cpv     = np.zeros((npr,1))
numpv   = 0
C_all   = []
rme     = np.zeros([nMineral,n+1])
rmef    = np.zeros([nMineral,n+1])
rmeb    = np.zeros([nMineral,n+1])
KK      = np.zeros([4, n+1])

eltime   = 0
ntime    = 0
counterdt = 0
dt       = np.copy(dt1)

Xi       = np.zeros((ns,n+1))
Xiformer = np.copy(Xi)
Xee      = np.zeros((ns,1))
timestep = 0
neg_dt   = False # Is dt so high that cause getting negative P or C values? True or False?
Pnew     = np.zeros(P.shape)
Pnew[0,-1] = Patm

Cnew      = np.zeros(C.shape)
Cnew[:,0] = np.copy(Cinj)
counterE   = 0
inj_Pvol  = 0
inj_Pvol_not_round=0;
ShutInTime=0;
TotalShutInTime = 0.5*24*60*60;


wew     =-1
shNaCl  = 0
critVOL = 0
Dvol    = 10**-8
VV      = np.ones((1,n+1))*critVOL

TtoPV  = 0
shNaCl = 0

Asf = np.zeros(As.shape)
Asb = np.zeros(As.shape)


dtmin = 10**-20
shPH  = 0
Adam  = 1000
        
Q    = 75.7682*kInjectionRate;  
uinj = Q/Ax

times=NPV*Ax*L*Porosity_i/Q+10
kkkkk=1

print(' rate constans:  ', k123,'\n', 'ns=',ns, '\n', 'np=',npr, '\n','Ksolid: ', Ksolid)
print(' Macid:',Macid,'\n', 'wp_acid',wp_acid,'\n','inletpH', inletpH)
print(' perm0:',K0,'\n n: ',n)
''' Field Scale Parameters '''
   

print(' Area (m2):',kArea,'\n', "Length(m):",kLength,'\n', 'Rate (gal/min):', kInjectionRate,'\n', "MaxDt(min):",K_CriticalDt ,'\n','Velocity (m/s):', uinj/100,'\n','Velocity (m/day):', uinj*86400/100)


'''-----------------------------------------------------------------------------------------------------------'''
ith_point = 0
Clast_ave = np.zeros([npr,1])
Qdt  = 0
CQdt = 0
'''-----------------------------------------------------------------------------------------------------------'''




Cpv=np.copy(C[:,-1].reshape((npr,1)))
pHpv=np.copy(C[1,-1])  




'Initialize the Global variables'

config.dt   = dt
config.Q    = Q
config.Ax   = Ax
config.n    = n
config.Patm = Patm
config.roi  = roi
config.por0f = por0f
config.Cl   = Cl
config.KK   = KK
config.Porosity_i = Porosity_i
config.Xinj       = Xinj
config.k123       = k123
config.Macid      = Macid
config.keq_acid   = keq_acid
config.gama1      = gama1
config.rmef       = rmef
config.rmeb       = rmeb
config.rme        = rme
config.Xi         = Xi
config.por0f      = por0f
config.Pformer    = Pformer
config.As         = As
config.Cinj       = Cinj
config.n          = n
config.Cl         = Cl
config.npr        = npr
config.nMineral   = nMineral
config.Keq        = Keq
config.cunit      = cunit
config.Stoichiometry= Stoichiometry
config.vf         = vf
config.vmif       = vmif
config.vb         = vb
config.vmib       = vmib
config.Ksolid     = Ksolid
config.cR         = cR
config.vmi        = vmi
config.vsolid     = vsolid
config.ns         = ns
config.vrjf       = vrjf
config.vrjb       = vrjb
config.Nvf        = Nvf
config.Nvb        = Nvb
config.dx         = dx
config.vis        = vis
config.K0         = K0
config.Ax         = Ax
config.aa         = aa
config.uinj       = uinj
config.bb         = bb
config.Pi         = Pi
config.km         = km
config.por0       = por0




while inj_Pvol_not_round < NPV:
    '''-----------------------------------------------------------------------------------------------------------'''
    
    # Electrical charge
    # Primary and secondary charges
    # P: 'H+', 'Ca2+', 'H2CO3*', CH3COO-
    # S: CH3COOH, CO3--, HCO3-, OH-, Ca(Acet)-, CaCO3 (aq), Ca(OH)+, Ca(HCO3)-
    
    pri_charge = np.array([1, 2, 0, 1])
    sec_charge = np.array([0, 2, 1, 1, 1, 0, 1, 1])
    
    aPri = np.array([9, 6, 0, 4.5])
    bPri = np.array([0, 0.165, 0, 0])
    
    aSec = np.array([0, 4.5, 4, 3.5, 4.4, 0, 4.4, 4.4])
    bSec = np.array([0, 0, 0, 0, 0, 0, 0, 0])
    
    ' Initialize arrays of ionic activities'
    I_P = np.zeros([1, n + 1])
    I_S = np.zeros([1, n + 1])
    I = np.zeros([1, n + 1])
    
    config.LandaP = np.ones([npr, n + 1])
    LandaS = np.ones([ns, n + 1])
    config.LandaS = LandaS
    
    ' Ionic Strength Calculation'
    for ee in range(n + 1):
        I_P[0, ee] = np.copy((0.5 * abs(C[:, ee]) / cunit * (pri_charge**2)).sum())
        I_S[0, ee] = np.copy((0.5 * abs(config.Xi[:, ee]) / cunit * (sec_charge**2)).sum())
        I[0, ee] = I_P[0, ee] + I_S[0, ee]
    
    ' Compute Landa of Primary Species'
    for ee in range(n + 1):
        for j in range(npr):
            config.LandaP[j, ee] = np.exp(
                -0.51 * (pri_charge[j]**2) * (I[0, ee]**0.5) /
                (1 + 0.3294 * aPri[j] * (I[0, ee]**0.5)) + bPri[j] * I[0, ee]
            )
    
    ' Compute Landa of Secondary Species'
    for ee in range(n + 1):
        for j in range(ns):
            config.LandaS[j, ee] = np.exp(
                -0.51 * (sec_charge[j]**2) * (I[0, ee]**0.5) /
                (1 + 0.3294 * aSec[j] * (I[0, ee]**0.5)) + bSec[j] * I[0, ee]
            )
    
    '''-----------------------New ------------------------------------------------------------------------------------'''
    
    ' Calculate the Reactive Surface Area'
    for yym in range(nMineral):
        for grid in range(n + 1):
            Asf[yym, grid] = As0[yym] * ((volfrac[yym, grid] / volfrac0[yym]) ** (2 / 3))
            Asb[yym, grid] = As0[yym] * ((config.por0[0, grid] / config.Porosity_i) ** (2 / 3))
    
    ' Boundary Condition for Surface Area'
    Asb[0, :] = 0
    
    ' Injection and Configuration Adjustments'
    if inj_Pvol >= NPV_NaCl:
        shNaCl += 1
        config.Cinj = np.copy(Cinj)
        config.Keq = np.copy(config.Keq)
        C[:, 0] = config.Cinj
        if shNaCl == 1:
            config.dt = np.copy(dt1)
    
    ' Store Previous States'
    Xiformer = np.copy(config.Xi)
    Cformer = np.copy(C)
    config.Pformer = np.copy(P)
    config.por0f = np.copy(config.por0)
    
    ' Time Step Updates'
    if neg_dt == True or counterE == 1:
        counterE = 0
        config.dt /= dt_reduce
        neg_dt = False
        timestep -= 1
        counterdt = 0
    
    timestep += 1
    counterdt += 1
    
    ' Adjust Time Step Size'
    if config.dt < dtc:  # Start refinement
        if counterdt % perdt == 0:
            counterdt = 0
            config.dt *= orderdt
    
    if config.dt > dtc:
        config.dt = np.copy(dtc)
    
    if config.dt < dtmin:
        print('This timestep size was not good.')
        break

    'Calculate the total concentrations'
    if ns>0:
        for ee in range (n+1):
            if ee==0:
                U[:,0]=config.Cinj+(Stoichiometry.transpose()*config.Xinj).sum(axis=1)
            else:
                Xee=config.Xi[:,ee]
                Xnon0=[1 if wq!=0 else 0 for wq in Xee]
                Ncf=(((vrjf*C[:,ee])!=0).sum(1))+Xnon0 # number of non-zero concentrations
                Ncb=((vrjb*C[:,ee])!=0).sum(1)         # number of non-zero concentrations
                for react in range(ns):
                    if Ncf[react]==Nvf[react] or Ncb[react]==Nvb[react]:    
                        config.Xi[react,ee]=(cunit/config.Keq[react]/config.LandaS[react,ee])*np.prod((C[:,ee]*config.LandaP[:,ee]/cunit)**Stoichiometry[react,:].transpose())      
                
                
                zigma=0
                cal=Stoichiometry.transpose()*config.Xi[:,ee]
                zigma=cal.sum(axis=1)
                U[:,ee]=C[:,ee]+zigma

        config.Uformer=np.copy(U)    
    elif ns==0:
        Xiformer=np.copy(config.Xi)
        Cformer=np.copy(C)
        config.Uformer=np.copy(C)
  
  
    
    C[:,0]=np.copy(config.Cinj)
    counter=0
    criteria=True
    sh_neg=1
    sh=0
    vol_former=np.copy(volfrac)
    neg_rme=np.ones((nMineral,n+1))
    endM=np.ones((nMineral,n+1))
    
    
    
    # ----------------------------------------------------------------------------------------
    # Initialize Variables and Begin Jacobian Calculation
    # ----------------------------------------------------------------------------------------

    while criteria==True:
        
        """
        Compute the full Jacobian matrix and the RHS vector for a system of equations.
    
        Parameters:
        - npr: Number of primary variables
        - n: Number of grid blocks
        - P: Pressure 
        - C: Concentration
        - config: Configuration object containing parameters
    
        Returns:
        - finalJacobian: The full Jacobian matrix
        - b: The RHS vector
        """
        
        'Jacobian Matrix--------------------------------------------------------------------'
        sh+=1
        for j in range(npr):
            # Initialize Jacobian matrix for concentration equations
            C_Jacobian=np.zeros((n,n))
            for k in range (npr):
                rr=time.time()
                
                # Compute the sub-Jacobian matrix for concentration
                C_Jacobian = cal_C_Jacobian(j,k,P,C,config.por0)
                
                # Build the main Jacobian by horizontally stacking submatrices
                if k==0:
                    Jacobian_Main=np.copy(C_Jacobian)
                elif k>0:
                    Jacobian_Main=np.hstack((Jacobian_Main,C_Jacobian))
                    
            # Append C-P Jacobian to the main Jacobian matrix
            Jacobian_Main=np.hstack((Jacobian_Main,cal_C_P_Jacobian(j,k,P,C,config.por0)))
            
            
            # Vertically stack the main Jacobian matrices for all primary variables
            if j==0:
                Jacobian=np.copy(Jacobian_Main)
            elif j>0:
                Jacobian=np.vstack((Jacobian,Jacobian_Main))
                
        # Append the last section of the Jacobian matrix for pressure equations
        lastSectionOfJacob=np.hstack((np.zeros((n,(npr)*(n))),cal_P_Jacobian(P,config.por0)))
        finalJacobMatrix=np.vstack((Jacobian,lastSectionOfJacob))


        # ----------------------------------------------------------------------------------------
        # Calculate the RHS Vector
        # ----------------------------------------------------------------------------------------
        
        # Total number of unknowns ( concentrations and pressure)
        nu = npr * n + n  # npr: number of primary variables
        
        # Initialize the RHS vector
        b = np.zeros((nu, 1))  # RHS vector to store residuals
        
        # Initialize the counter for indexing the RHS vector
        ss = -1
        
        # Compute residuals for the concentration equations and fill in the corresponding entries in b
        for j in range(npr):  # Loop over primary variables
            for i in range(1, n + 1):  # Loop over grid blocs
                ss += 1
                # Calculate the residual
                b[ss, 0] = -cal_C_Equation(j, i, P, C, config.por0)
        
        # Compute residuals for the pressure equations and fill in the corresponding entries in b
        for i in range(n):  
            ss += 1
            # Calculate the pressure residual for blocks and store it in the RHS vector
            b[ss, 0] = -cal_P_Equation(i, P, config.por0)
        
        # At this point, b contains the full RHS residual vector for the system

        
        'Calculate the Concentration and Pressure changes'
        dpc=np.linalg.solve(finalJacobMatrix,b)
        
        
        'Update Concentration'
        ss=0
        for j in range (npr):
            for i in range(1,n+1):
                Cnew[j,i]=C[j,i]+dpc[ss,0]
                ss+=1
             
        'Update Pressure     '
        for i in range (n):
            Pnew[0,i]=P[0,i]+dpc[ss,0]
            ss+=1
            
        nu=npr*n+n; # number of unknowns 
        fgnew=np.zeros((nu,1))
        ss=-1
        for j in range(npr):    
            for i in range (n):    
                ss+=1;
                fgnew[ss,0]=cal_C_Equation(j,i+1,Pnew,Cnew,config.por0)# 
                
        for i in range (n):
            ss+=1
            fgnew[ss,0]=cal_P_Equation(i,Pnew,config.por0)# %P
                
        counter+=1

        criteria=False
        if np.amax(fgnew)>0.0002:
            criteria=True
        
        if sh_neg==1 and criteria==False:
            sh_neg=2
            C_negetive=(Cnew-np.abs(Cnew))
            for ee in range(n+1):   # grid
                for com in range(C.shape[0]):   # component
                    if C_negetive[com][ee] !=0:
                        print('C[{}][{}] is negetive.'.format(com,ee))                        
                        neg_dt=True

        sh_vol=False    
        if counter>=30 or sum(np.abs(fgnew))>=10**9:
            counterE=1
            criteria=False
            sh_vol=True
            config.Xi=np.copy(Xiformer)
            C=np.copy(Cformer)
            P=np.copy(config.Pformer)
            config.por0=np.copy(por0f)
            
        if neg_dt==False and sh_vol==False: 
            C=np.copy(Cnew)
            P=np.copy(Pnew)
            '''-----------------------------------------------------------------------------------------------------------'''
            for ee in range (n+1):
                if ee==0:
                    config.Xi[:,0]=np.copy(config.Xinj)
                else:
                    Xee=config.Xi[:,ee]
                    Xnon0=[1 if wq!=0 else 0 for wq in Xee]
                    Ncf=(((vrjf*C[:,ee])!=0).sum(1))+Xnon0 # number of non-zero concentrations
                    Ncb=((vrjb*C[:,ee])!=0).sum(1)         # number of non-zero concentrations
                    for react in range(ns):
                        if Ncf[react]==Nvf[react] or Ncb[react]==Nvb[react]:    
                            config.Xi[react,ee]=(cunit/config.Keq[react]/config.LandaS[react,ee])*np.prod((C[:,ee]*config.LandaP[:,ee]/cunit)**Stoichiometry[react,:].transpose())      
            '''-----------------------------------------------------------------------------------------------------------'''

        else:
            config.Xi=np.copy(Xiformer)
            C=np.copy(Cformer)
            P=np.copy(config.Pformer)  
            config.por0=np.copy(por0f)


    'End of while loop and start of volfrac update   '
    vol_former=np.copy(volfrac)
    if criteria==False and neg_dt==False and sh_vol==False:
        for_rme_in_inlet=cal_C_Equation(0,0,P,C,config.por0)
        for ee in range(n+1):
            for com in range(nMineral):
                ggg=1
                volfrac[com,ee]=volfrac[com,ee]-ggg*config.dt*MV_m[com]*sum(vm[com,:]*config.rme[:,ee])                    
                if volfrac[com,ee]<critVOL:
                    neg_dt==True
                    volfrac=np.copy(vol_former)
        if neg_dt==True:
            volfrac=np.copy(vol_former)
        elif neg_dt==False:
            for ee in range(n+1):
                for com in range(nMineral):
                    if volfrac[com,ee]<critVOL+Dvol:
                        volfrac[com,ee]=critVOL
    if neg_dt==False:
        config.por0f=np.copy(config.por0)
        for ee in range(n+1):
            config.por0[0,ee]=1-sum(volfrac[:,ee])/nMineral
        config.por0[0,0]=np.copy(por0[0,1])
    else:
        config.Xi=np.copy(Xiformer)
        C=np.copy(Cformer)
        P=np.copy(config.Pformer)  
        config.por0=np.copy(por0f)
    
    
    
            
    'Save the values'                        
    if neg_dt==False and counterE==0:
        eltime=np.copy(eltime+config.dt)
        
        PERM=np.zeros([1,n+1])
        for Nperm in range(n+1):
            PERM[0,Nperm]=cal_K(Nperm,config.por0)
        if eltime==config.dt:
            C_all=np.copy(C)
            Gama_P_all=np.copy(config.LandaP)
            Gama_S_all=np.copy(config.LandaS)
            volfrac_all=np.copy(volfrac)
            Asf_all=np.copy(Asf)
            Asb_all=np.copy(Asb)
            rme_all=np.copy(config.rme)
            rmb_all=np.copy(config.rmeb)
            rmf_all=np.copy(rmef)
            KK_all=np.copy(config.KK) 
            R_all=np.copy(config.R)
            Xi_all=np.copy(config.Xi)
            por0_all=np.copy(config.por0)
            P_all=np.copy(P)
            Perm_all=np.copy(PERM)
            Clast=np.copy(C[:,-1])
            Xilast=np.copy(config.Xi[:,-1])
            Gama_P_Last_all=np.copy(config.LandaP[:,-1])
            Gama_S_Last_all=np.copy(LandaS[:,-1])
        else:
            C_all=np.hstack((C_all,C))
            Gama_P_all=np.column_stack((Gama_P_all,config.LandaP))
            Gama_S_all=np.column_stack((Gama_S_all,config.LandaS))
            volfrac_all=np.hstack((volfrac_all,volfrac))  
            Asf_all=np.hstack((Asf_all,Asf)) 
            Asb_all=np.hstack((Asb_all,Asb)) 
            rme_all=np.hstack((rme_all,config.rme))
            rmf_all=np.hstack((rmf_all,config.rmef))
            rmb_all=np.hstack((rmb_all,config.rmeb))
            KK_all=np.hstack((KK_all,config.KK))
            R_all=np.hstack((R_all,config.R))
            Xi_all=np.hstack((Xi_all,config.Xi))
            por0_all=np.hstack((por0_all,config.por0))
            P_all=np.hstack((P_all,P))
            Perm_all=np.hstack((Perm_all,PERM))
            Clast=np.column_stack((Clast,C[:,-1]))
            Xilast=np.column_stack((Xilast,config.Xi[:,-1]))
            Gama_P_Last_all=np.column_stack((Gama_P_Last_all,config.LandaP[:,-1]))
            Gama_S_Last_all=np.column_stack((Gama_S_Last_all,config.LandaS[:,-1]))
            print('Q',config.Q*60)
        permP+=[[1000*config.Q*vis*L/config.Ax/(P[0,0]-P[0,-1])]]
           
        if eltime==config.dt:
            xPV_amount=config.Q*config.dt/config.Ax/L/config.Porosity_i
            xPV.append(xPV_amount)
        else:
            xPV_amount=xPV[-1]+config.Q*config.dt/config.Ax/L/config.Porosity_i
            xPV.append(xPV_amount)
            
        'Calculate the average values ---------------------------------------------------------------------------------'  
        if ith_point < N_Exp_Data-1:
            xPVexp=0;
            xPVsim=0;
            
            if xPVsim <  xPVexp:
                
                CQdt+=C[:,-1]*config.dt*config.Q 
                Qdt+=config.Q*config.dt
                Clast_ave[:,ith_point]=CQdt/Qdt #Ave of the Concentration
                
            else: # Selection of a new point 
                    
                ith_point+=1
                
                if ith_point < N_Exp_Data-1:
                    Qdt=0.0
                    CQdt=0.0
                    
                    CQdt+=C[:,-1]*config.dt*config.Q
                    Qdt+=config.Q*config.dt
                    Clast_ave=np.column_stack((Clast_ave,CQdt/Qdt))

            
        '''-----------------------------------------------------------------------------------------------------------'''
          
    

        
          
        inj_Pvol=int(xPV_amount)
        inj_Pvol_not_round=np.copy(xPV[-1])
        
        '''-----------------------------------------------------------------------------------------------------------'''
        if inj_Pvol_not_round>=2: #and  Step2>inj_Pvol_not_round:
            
            config.Q=0.00001
            ShutInTime=ShutInTime+config.dt;
            print('step2')
            config.uinj=config.Q/config.Ax#
            here_N=0
            if ShutInTime > TotalShutInTime:
                inj_Pvol_not_round=NPV+5;

        '''-----------------------------------------------------------------------------------------------------------'''

        
        if inj_Pvol>int(numpv):
            if numpv==0:
                Cpv=np.copy(C[:,-1].reshape((npr,1)))
                pHpv=np.copy(C[1,-1])
            else:
                Cpv=np.hstack((Cpv,C[:,-1].reshape((npr,1))))
                pHpv=np.hstack((pHpv,C[1,-1],))
            numpv+=1
        print('time=',np.round(eltime,1),'s ', 'iPV=',np.round(inj_Pvol_not_round, 3),', timestep=',timestep,'dt=',config.dt,'Second')

    if neg_dt==False and counterE==0:       
        with open('myfile.pkl','wb') as PandC:
            pickle.dump([KK_all,Gama_P_Last_all,Gama_S_Last_all,Clast_ave,Gama_P_all,Gama_S_all,rmb_all,rmf_all,Asb_all,Asf_all,volfrac_all,L,C_all,rme_all,R_all,Xi_all,por0_all,P_all,Perm_all,P,C,Clast,Xilast,xPV,n,eltime,cunit,dx,km,pvs,pvf,PERM,volfrac,config.por0,config.rme,permP],PandC)
    
    if C[0,-1]/1000>=10**-1:
        shPH=1

pHPV=-np.log10(pHpv/1000)

The_pH=-np.log10(Clast[0,:]*Gama_P_Last_all[0,:]/1000) 
The_Ca=np.copy(Clast[1,:])
Final_pH=np.copy(xPV)
Final_pH=np.row_stack((Final_pH,The_pH))

Final_Ca=np.copy(xPV)
Final_Ca=np.row_stack((Final_Ca,The_Ca))

if lll==0:
    with open('myfile_final1.pkl','wb') as PandC2:
        pickle.dump([KK_all,Gama_P_Last_all,Gama_S_Last_all,Clast_ave,Gama_P_all,Gama_S_all,rmb_all,rmf_all,Asb_all,Asf_all,volfrac_all,L,C_all,rme_all,R_all,Xi_all,por0_all,P_all,Perm_all,P,C,Clast,Xilast,xPV,C_all,n,eltime,cunit,dx,km,Cpv,pHPV,pvs,pvf,PERM,volfrac,config.por0,rme,permP],PandC2)


dict_all_data[i_kRate]           = [KK_all,Gama_P_Last_all,Gama_S_Last_all,Clast_ave,Gama_P_all,Gama_S_all,rmb_all,rmf_all,Asb_all,Asf_all,volfrac_all,L,C_all,rme_all,R_all,Xi_all,por0_all,P_all,Perm_all,P,C,Clast,Xilast,xPV,C_all,n,eltime,cunit,dx,km,Cpv,pHPV,pvs,pvf,PERM,volfrac,config.por0,rme,permP]
dict_all_data_necessary[i_kRate] = [KK_all,Gama_P_Last_all,Gama_S_Last_all,Clast_ave,Clast,Xilast,xPV,n,eltime,cunit,dx]

if lll==0:
    with open('dict_all_data.pkl','wb') as PC:
        pickle.dump([dict_all_data],PC)
        
    with open('dict_all_data_necessary.pkl','wb') as PC2:
        pickle.dump([dict_all_data_necessary],PC2)
        
    with open('dict_all_data_necessary_BU.pkl','wb') as PC2:
        pickle.dump([dict_all_data_necessary],PC2)
            
EndTime=time.time();
print('Run Time: ', (EndTime-StartTime)/60,'  minute')
    
