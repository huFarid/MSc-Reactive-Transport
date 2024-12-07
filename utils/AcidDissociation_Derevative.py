from utils.AcidDissociation_Function import AcidDissociation_Function

# Derevative of a equation related to Initial dissociation of acetic acid--------------------------------
def AcidDissociation_Derevative(x_molal_H, keq_acid, Macid):
    
    dx=0.00000001;
    xmin=x_molal_H-dx
    xplus=x_molal_H+dx
    
    prim=(AcidDissociation_Function(xplus, keq_acid, Macid)-AcidDissociation_Function(xmin, keq_acid, Macid))/2/dx;
    
    return(prim)  
