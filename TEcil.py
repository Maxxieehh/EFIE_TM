###############################################################################
#
# This code is part of the Eindhoven ElectroMagnetics Solver
#
#   Function name: Analytial_2D_TE
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
#       6 lists of the same shape as evaluation_points_x and evaluation_points_y
#       The outputs are: Hx,Hy,Ez,Hix,Hiy,Eiz. The first three are the fields
#       with scattering. The second three are solely the incident fields,
#       without scattering.
#
#   Documentation:
#       None. See accompanying jupyter notebook for example usage.
#       A similar solution to this problem is found in: Albertus M. van de
#       Water, “LEGO: Linear Embedding via Green’s Oper-ators”, PhD Thesis,
#       Eindhoven University of Technology, 2007] Appendix B1
#
#   Author name(s): Mohammad Shahid (with edits by R. J. Dilz)
#
#   Date: January 28, 2021
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
from scipy.special import jv, hankel2
import matplotlib.pyplot as plt

def Analytical_2D_TE(simparams):
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
    rho = np.sqrt(x**2+y**2) + ((x==0)==(y==0))*1e3*eps # Avoid the singularity in the origin.
    phi = np.arctan2(y,x)
    omega = 2*math.pi*f
    mu0 = math.pi*4e-7
    epsilon0 = 8.854187812813e-12
    k0 = omega*math.sqrt(mu0*epsilon0)
    E0 = math.sqrt(mu0/epsilon0) # Amplitude of incident wave
    #print('value of Eo', E0)
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

        A[i] = (E0*((J1d*H2)-(J1*H2d)))/(((math.sqrt(epsr))*J1cd*H2)-(J1c*H2d))
        B[i] = ((A[i]*J1c)-(E0*J1))/H2

#-------------------------- Compute Hphi and Hrho------------------------------
    Hphi  = np.zeros(len(x))
    Hiphi = np.zeros(len(x))
    Hrho  = np.zeros(len(x))
    Hirho = np.zeros(len(x))

    for i in range(0,2*nmax+1):
        n = i-nmax
        if n == 0:
            Hphi = Hphi - 1j*k0/(omega*mu0)* ( math.sqrt(epsr)*A[nmax+1]*(rho<=a)* (-jv(1,rho*k0*math.sqrt(epsr))) \
                                        + (rho>a) *( E0*(-jv(1,rho*k0) ) + B[nmax+1]*( -hankel2(1,rho*k0) ) ) )*np.exp(1j*0*(phi-phi_i))
            Hiphi = Hiphi - 1j*k0/(omega*mu0)*(-E0*jv(1,rho*k0))*np.exp(1j*0*(phi-phi_i))

        else:

            Hphi = Hphi - 1j**(-n)*1j*k0/(omega*mu0)* ( math.sqrt(epsr)*A[i]*(rho<=a)* (jv(n-1,rho*k0*math.sqrt(epsr)) - n/(k0*math.sqrt(epsr)*rho)*jv(n,rho*k0*math.sqrt(epsr))) \
            + (rho>a) *( E0*(jv(n-1,rho*k0) - n/(k0*rho)*jv(n,rho*k0)) + B[i]*( hankel2(n-1,rho*k0) - n/(k0*rho)*hankel2(n,rho*k0) ) ) )*np.exp(1j*n*(phi-phi_i))

            Hiphi = Hiphi - 1j**(-n)*1j*k0/(omega*mu0)*E0*(jv(n-1,rho*k0) - n/(k0*rho)*jv(n,rho*k0))*np.exp(1j*n*(phi-phi_i))

        Hrho = Hrho - 1j**(-n)*1/(omega*mu0*rho)*( A[i]*(rho<=a)*jv(n,rho*k0*math.sqrt(epsr)) \
            + (rho>a) *( E0*jv(n,rho*k0) + B[i]*hankel2(n,rho*k0)) )*np.exp(1j*n*(phi-phi_i))*n
        Hirho = Hirho-1J**(-n)*1/(omega*epsilon0*rho)*E0*jv(n,rho*k0)*np.exp(1j*n*(phi-phi_i))*n

    Hx = np.cos(phi)*Hrho-np.sin(phi)*Hphi   #*double(rho>a);
    Hy = np.sin(phi)*Hrho+np.cos(phi)*Hphi       #*double(rho>a);

    #Hix = np.cos(phi)*Hirho-np.sin(phi)*Hiphi    #*double(rho>a);
    #%Hiy = np.sin(phi)*Hirho+np.cos(phi)*Hiphi    #*double(rho>a);

#--------------------------Compute Hz & Hiz------------------------------------
    Ez = np.zeros(len(rho))
    Eiz = Ez
    for i in range(0,2*nmax+1):
        n = i-nmax
        Ez = Ez + 1j**(-n)* ( (rho<=a)*A[i]*jv(n,rho*k0*math.sqrt(epsr)) \
        + (rho>a)*(E0*jv(n,rho*k0) + B[i]*hankel2(n,rho*k0)) )*np.exp(1j*n*(phi-phi_i))
        Eiz = Eiz + (1j)**(-n)*E0*jv(n,rho*k0)*np.exp(1j*n*(phi-phi_i))
#--------------------------Compute Hz & Hiz------------------------------------
    Ez = np.zeros(len(rho))
    Eiz = Ez
    for i in range(0,2*nmax+1):
        n = i-nmax
        Ez = Ez + 1j**(-n)* ( (rho<=a)*A[i]*jv(n,rho*k0*math.sqrt(epsr)) \
        + (rho>a)*(E0*jv(n,rho*k0) + B[i]*hankel2(n,rho*k0)) )*np.exp(1j*n*(phi-phi_i))
        Eiz = Eiz + (1j)**(-n)*E0*jv(n,rho*k0)*np.exp(1j*n*(phi-phi_i))
    return Hx, Hy, Ez, Eiz
