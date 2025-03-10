###############################################################################
#
# This code is part of the Eindhoven ElectroMagnetics Solver 
#  
#   Function name: Analytial_2D_TM
#
#   Description: This file contains the analytical solution for 2D
#       scattering by the cilinder.
#
#   Input: 
#       The input is a dictionary with the simulation parameters: simparams
#       Example:
#   simparams = {
#       'frequency' : 3e8, #frequency of wave
#       'epsilon_r' : 2.4, #relative permittivity of cilinder
#       'radius' : 1.0, # radius of cilinder (centered at x=y=0)
#       'modes' : 50, # the number of modes. Should be chosen high enough
#                       for an accurate result
#        'incident_angle' : np.pi/2, #angle of incident wave
#       'evaluation_points_x' : gridx, #2D list of x-coordinates on which 
#               the E and H fields are computed
#        'evaluation_points_y' : gridy # same but for y-coordinates
#   }
#
#
#   Output:
#       4 lists of the same shape as evaluation_points_x and evaluation_points_y
#       The outputs are: Ex,Ey,Hz,Hiz. The first three are the fields
#       with scattering. The second is solely the incident field,
#       without scattering.
#   
#   Documentation: 
#       None. See accompanying jupyter notebook for example usage.
#       A similar solution to this problem is found in: Albertus M. van de 
#       Water, “LEGO: Linear Embedding via Green’s Oper-ators”, PhD Thesis,
#       Eindhoven University of Technology, 2007] Appendix B1
#   
#   Author name(s): R. J. Dilz and Mohammad Shahid
#               Based on matlab code by Maarten van Rossem
#   
#   Date: October 28, 2021
#
# The above authors declare that the following code is free of plagiarism.
#
# Maintainer/Supervisor: Roeland J. Dilz (r.dilz@tue.nl)
#
# This code will be published under GPL v 3.0 https://www.gnu.org/licenses/gpl-3.0.en.html
#
###############################################################################
import numpy as np
import math 
from numpy import sqrt,exp,cos,sin
from scipy.special import jv, hankel2
import matplotlib.pyplot as plt

def Analytical_2D_TM(simparams):
    f = simparams['frequency']
        # f = frequency of plane wave
    a = simparams['radius']
        # a = radius of dielectric cylinder
    epsr = simparams['epsilon_r']
        # epsr = relative permittivity of cilinder
    phi_i = simparams['incident_angle']
        # phi_i = angle of incident wave, with respect to x-axis
    nmax = simparams['modes']
        # nmax = maximum order of besselfunctions to use
    x = simparams['evaluation_points_x']
        # The x coordinates on which the fields are evaluated
    y = simparams['evaluation_points_y']
        # The y coordinates on which the fields are evaluated
    if(len(x) != len(y)):
        raise("Error, different number of coordinates in x and y")


#------------------------- Parameter declaration-------------------------------
    eps = np.finfo(float).eps
    rho = np.sqrt(x**2+y**2) + ((x==0)==(y==0))*1e-3*eps # Avoid the singularity in the origin.
    phi = np.arctan2(y,x)
    omega = 2*math.pi*f
    mu0 = math.pi*4e-7
    epsilon0 = 8.854187812813e-12
    k0 = omega*math.sqrt(mu0*epsilon0)
    E0 = math.sqrt(mu0/epsilon0) # Amplitude of incident wave
    H0 = 1
    print('value of Eo', E0)
#---------------- Compute coefficients at boundary rho=a-----------------------
    B = np.zeros(2*nmax+1,dtype=np.complex_)
    A = B.copy()
    
    for i in range(0,2*nmax+1):
        n = i-nmax
    
        J1 = jv(n,a*k0)
        J1c = jv(n,a*k0*math.sqrt(epsr))
        H2 = hankel2(n,a*k0)
        
        if n==0:
            J1d = -jv(1,a*k0)
            J1cd = -jv(1,a*k0*math.sqrt(epsr))
            H2d = -hankel2(1,a*k0)
            
        else:
            J1d = jv(n-1,a*k0) - n/(k0*a)*jv(n,a*k0)
            J1cd = jv(n-1,a*k0*math.sqrt(epsr)) - n/(k0*math.sqrt(epsr)*a)\
                    *jv(n,a*k0*math.sqrt(epsr))
            H2d = hankel2(n-1,a*k0) - n/(k0*a)*hankel2(n,a*k0)
        
        A[i] = H0*(J1*H2d-J1d*H2)/(J1c*H2d-J1cd*H2/sqrt(epsr))
        B[i] = (A[i]*J1cd/sqrt(epsr) - H0*J1d)/H2d
        
    #print(A)
    #print(B)
#-------------------------- Compute Hphi and Hrho------------------------------
    Ephi  = np.zeros(len(x))
    Eiphi = np.zeros(len(x))
    Erho  = np.zeros(len(x))
    Eirho = np.zeros(len(x))
    
    for i in range(0,2*nmax+1):
        n = i-nmax
        if n == 0:
            Ephi = Ephi + 1j*k0/omega/epsilon0* ( A[nmax]/sqrt(epsr)*(rho<=a)* (-jv(1,rho*k0*sqrt(epsr))) \
                + (rho>a) *( H0*(-jv(1,rho*k0) ) + B[nmax]*( -hankel2(1,rho*k0) ) ) )*exp(1j*0*(phi-phi_i))
            Eiphi = Eiphi + 1j*k0/omega/epsilon0*(-H0*jv(1,rho*k0))*exp(1j*0*(phi-phi_i))
            
        else:

            Ephi = Ephi + 1j**(-n)*1j*k0/omega/epsilon0* ( A[i]/sqrt(epsr)*(rho<=a)* (jv(n-1,rho*k0*sqrt(epsr)) - n/(k0*sqrt(epsr)*rho)*jv(n,rho*k0*sqrt(epsr))) \
            + (rho>a) *( H0*(jv(n-1,rho*k0) - n/(k0*rho)*jv(n,rho*k0)) + B[i]*( hankel2(n-1,rho*k0) - n/(k0*rho)*hankel2(n,rho*k0) ) ) )*exp(1j*n*(phi-phi_i))
            Eiphi = Eiphi + 1j**(-n)*1j*k0/omega/epsilon0*H0*(jv(n-1,rho*k0) - n/(k0*rho)*jv(n,rho*k0))*exp(1j*n*(phi-phi_i))  
            
        Erho = Erho + 1j**(-n)*1./(omega*epsilon0*rho)*( A[i]/epsr*(rho<=a)*jv(n,rho*k0*sqrt(epsr)) \
            + (rho>a) *( H0*jv(n,rho*k0) + B[i]*hankel2(n,rho*k0)) )*exp(1j*n*(phi-phi_i))*n
        Eirho = Eirho + 1j**(-n)*1./(omega*epsilon0*rho) * H0*jv(n,rho*k0)*exp(1j*n*(phi-phi_i))*n
    #print("Ephi")
    #print(Ephi)
    #print("Erho")
    #print(Erho)
    Ex = cos(phi)*Erho-sin(phi)*Ephi
    Ey = sin(phi)*Erho+cos(phi)*Ephi
    Eix = cos(phi)*Eirho-sin(phi)*Eiphi
    Eiy = sin(phi)*Eirho+cos(phi)*Eiphi
#--------------------------Compute Hz & Hiz-----------------------------------
    Hz = np.zeros(len(rho))
    Hiz = Hz
    for i in range(1,2*nmax+1):
        n = i-1-nmax;
        Hz = Hz + 1j**(-n)* ( (rho<=a)*A[i-1]*jv(n,rho*k0*sqrt(epsr)) \
            + (rho>a)*(H0*jv(n,rho*k0) + B[i-1]*hankel2(n,rho*k0)) )*exp(1j*n*(phi-phi_i))
        Hiz = Hiz + 1j**(-n)*(H0*jv(n,rho*k0))*exp(1j*n*(phi-phi_i))
    return Ex,Ey,Hz,Hiz
