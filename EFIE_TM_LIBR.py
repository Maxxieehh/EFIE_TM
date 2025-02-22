###############################################################################
#
# This code is part of the Eindhoven ElectroMagnetics Solver
#
#   Function name: EFIE_TE
#
#   Description: This file contains all the functions used to calculate the
#                incident, scattered and total electric fields,
#                But the EFIE_TE function calculates the surface current density
#                on an arbitrary shape.
#
#   Input:  coordinates: the boundary points that describe the arbitrary shape,
#                        make sure there are plenty of sample points and
#                        if the shape is closed, do so to here in the input.
#           wavelength: wavelength of the incident plane wave.
#           angle: angle of the incident plane wave .
#
#   Output: The output is a tuple with multiple items.
#           Jz: the surface current density, has the length of coordinates-1.
#           Ein: the incident electric field sampled at the coordinates provided.
#           Zm: the impedance matrix used to calculate Jz.
#
#   Function name: Etot
#
#   Description: This file contains all the functions used to calculate the
#                incident, scattered and total electric fields,
#                But the Etot function calculates the total electric field
#                caused by the incident and scattered fields due to the shape present.
#
#   Input:  Jz: the surface current density, can be found using EFIE_TE
#           R: array including x,y coordinates to sample the total electric field at.
#           coordinates: the boundary points that describe the arbitrary shape,
#                        make sure there are plenty of sample points and
#                        if the shape is closed, do so to here in the input.
#           wavelength: wavelength of the incident plane wave.
#           angle: angle of the incident plane wave .
#
#   Output: Etot: array with the complex values of the total electric field
#                 sampled at the positions provided in R.
#
#   Documentation: See documentation in the files provided.
#
#   Original Author name(s):    Justin Geerarts
#                               Rowin Ansems
#                               Tudor Popa
#
#   Adapor Author name(s):      Dorien Duyndam
#                               Daniel Joaquim Ho
#                               Charalambos Kypridemou
#                               Max van Wijk
#
#   Original Date:      29-03-2021
#   Adaptation Date:    21-02-2025
#
# The above authors declare that the following code is free of plagiarism.
#
# Maintainer/Supervisor: Roeland J. Dilz (r.dilz@tue.nl)
#
# This code will be published under GPL v 3.0 https://www.gnu.org/licenses/gpl-3.0.en.html
###############################################################################
import numpy as np
import scipy as sp
from numpy import sin ,cos, pi
from scipy import integrate
from scipy.special import hankel2
    
# THIS IS A TEST

##-----------Functions used for the Electric field-------------------------
def DiscritizeEin(coordinates,wavelength,angle):
    # Create an array for the number of segments
    # the code does not close the contour, do so manually
    M = len(coordinates)-1
    Ein = np.zeros(M,dtype=np.complex128)
    # values of Ein are complex so matrix needs to be able to handle complex values
    # Overwrite each datapoint with the actual value of the Ein field
    for m in np.arange(M):
        # Sample the E field for varying coordinates, based on the testing Function
        # The loop goes over segments between 2 coordinates inside the array
        # Integrate.quad cannot deal with complex numbers (holds as of 22/03/2021)
        segment = Coordinates_to_segment(coordinates,m)
        EReal = lambda tau: np.real(Efield_in(Pulsebase(tau, segment),wavelength,angle))
        EImag = lambda tau: np.imag(Efield_in(Pulsebase(tau, segment),wavelength,angle))
        IntReal = integrate.quad(EReal,0,1)[0]
        IntImag = integrate.quad(EImag,0,1)[0]
        # Correct for the test function used, see documentation
        # [0], since integrate.quad outputs result[0] and upper bound of error[1]
        dst = np.linalg.norm(np.subtract(segment[1],segment[0]))
        # multiplication with length of segment due to normalization
        Ein[m] =  dst*(IntReal+ 1j*IntImag)
    return Ein

def Coordinates_to_segment(coordinates,set):
    # Take 2 positions in coordinates and return these to define the segment
    if set <len(coordinates)-1:
        # Link the index set of coordinates to the next index
        index = np.array([set,set+1])
    else:
        # If the value of set is invalid, raise an error
        raise ValueError("The requested segment is not found in the boundary points")
    return coordinates[index]

def Coordinates_to_Nodes(coordinates, set):
    # Create a list of nodes used to create the basis and test functions
    if np.array_equal(coordinates[0], coordinates[-1]):
        # Start and end of boundary_points are the same --> closed surface
        if set == len(coordinates) - 1:
            index = np.array([set-1, set, 0])
        elif set == 0:
            index = np.array([-1, set, set+1])
        else:
            index = np.array([set-1, set, set+1]) 
    else:
        # Start and end of boundary_points are NOT the same --> open surface
        if (set > 0) and (set < len(coordinates) - 1):
            # We neglect the 1st and last node, as these do not have two neighbours
            index = np.array([set-1, set, set+1])
        else:
            raise ValueError("The requested node does not have a left/right neighbor")
    return coordinates[index]

def Pulsebase(tau,segment):
    # Used for the test and basis function, creates a linear connection between the edges of the segment
    xvec=segment[:,0]
    yvec=segment[:,1]
    # define the beginning and ending positions
    x1, x2, y1, y2 = xvec[0], xvec[1], yvec[0], yvec[1]
    Xtau = tau*x2+(1-tau)*x1
    Ytau = tau*y2+(1-tau)*y1
    # Return the values (x,y) based on tau
    return [Xtau,Ytau]

    
def Efield_in(r,wavelength,angle):
    # Calculate the electric field value based on:
    # the x and y position (in r), wavelength and input angle
    mu0 = pi*4e-7
    epsilon0 = 8.854187812813e-12
    H0=1
    E0 =H0* np.sqrt(mu0/epsilon0)# Amplitude of incident wave
    # electric field is normalized to a magnetic field of 1
    x, y = r[0], r[1]
    # Assuming plane wave in losless material
    k0 = 2*pi/wavelength
    return E0*np.exp(1j*k0*(cos(angle)*x+sin(angle)*y))

##-----------Functions used for the Z matrix------------------------------
def Zmatrix(coordinates,wavelength):
    # Calculate the Z matrix used for the Method of Moments solution
    # Create a matrix for the number of coordinates that are given
    M = len(coordinates)-1 #python indexes from 0, unlike MATLAB
    Z = np.zeros((M,M),dtype=np.complex128) #Z can contain complex values
    for i in np.arange(M):
        for m in np.arange(M):
            # Correct for the test/basis function length used, see documentation
            segment_m = Coordinates_to_segment(coordinates,m)
            segment_i = Coordinates_to_segment(coordinates,i)
            dst_m = np.linalg.norm(np.subtract(segment_m[1],segment_m[0]))
            dst_i = np.linalg.norm(np.subtract(segment_i[1],segment_i[0]))
            dst = dst_m*dst_i
            if m==i:
                # The diagonal is singular
                Z[m,i] = dst*Z_mi_diag(coordinates,wavelength,m)
            elif i == m-1 or i==m+1:
                # One position is singular for adjacent segments
                Z[m,i] = dst*Z_mi_adj(coordinates,wavelength,m,i)
            else:
                # These indices are non-singular
                Z[m,i] = dst*Z_mi(coordinates,wavelength,m,i)
    return Z

def Z_mi(coordinates,wavelength,m,i):
    mu0 = 4*pi*10**-7
    c = 299792458
    omega = 2*pi*c/wavelength
    # Integrable Greens function, which is non-singular in this case
    GReal = lambda tau, prime: np.real(green(tau,prime,coordinates,wavelength,m,i))
    GImag = lambda tau, prime: np.imag(green(tau,prime,coordinates,wavelength,m,i))
    # Integrate.dblquad cannot deal with complex number (holds as of 22/03/2021)
    IntReal = integrate.dblquad(GReal, 0, 1, 0, 1)[0]
    IntImag = integrate.dblquad(GImag, 0, 1, 0, 1)[0]
    return -1j*omega*mu0*(IntReal +1j*IntImag)

def Z_mi_adj(coordinates,wavelength,m,i):
    # Special case, Greens function becomes singular on edge points
    mu0 = 4*pi*10**-7
    c = 299792458
    omega = 2*pi*c/wavelength
    # Use (double) Gauss-legendre quadrature method
    degree=30
    xh,wh=np.polynomial.legendre.leggauss(degree)
    xj=xh #positions for evaluation
    wj=wh #weights for evaluation
    ans=0
    for h in np.arange(degree):
        for j in np.arange(degree):
            ans+=wh[h]*wj[j]*green(0.5*(xh[h]+1),0.5*(xj[j]+1),coordinates,wavelength,m,i)
    return -1j*omega*mu0*ans/4

def green(tau,prime,coordinates,wavelength,m,i):
    # The Green's function for this specific case, represented using the Hankel Function
    # Note this function only works for non-singular arguments (m!=i)
    if m==i:
        raise ValueError("Singular case, use a different function!")
    k0 = 2*pi/wavelength
    # Obtain the arguments for the Greens function, based on the test/basis function
    Pm = Pulsebase(tau,Coordinates_to_segment(coordinates,m))
    Pi = Pulsebase(prime,Coordinates_to_segment(coordinates,i))
    dst = np.linalg.norm(np.subtract(Pm,Pi))
    return (-1j/4)*hankel2(0,k0*dst)

def Z_mi_diag(coordinates,wavelength,m):
    # Special case, Green's function becomes singular so requires special integration
    # Due to the change of basis cannot use the Green's function defined earlier, see documentation
    k0 = 2*pi/wavelength
    mu0 = 4*pi*10**-7
    c = 299792458
    omega = 2*pi*c/wavelength
    # Obtain the segment used for the integral
    segment = Coordinates_to_segment(coordinates,m)
    dst = np.linalg.norm(np.subtract(segment[1],segment[0]))
    # First define the variable for inner integral then outer
    # If prime is not defined here, integraton will not work
    QReal = lambda nu, prime: np.real(2*nu*hankel2(0,k0*nu**2*dst))
    QImag = lambda nu, prime: np.imag(2*nu*hankel2(0,k0*nu**2*dst))
    # For argument in boundary first give outer integral then inner boundary
    IntAReal = integrate.dblquad(QReal, 0, 1, 0, lambda prime: np.sqrt(prime))[0]
    IntAImag = integrate.dblquad(QImag, 0, 1, 0, lambda prime: np.sqrt(prime))[0]
    IntBReal = integrate.dblquad(QReal, 0, 1, 0, lambda prime: np.sqrt(1-prime))[0]
    IntBImag = integrate.dblquad(QImag, 0, 1, 0, lambda prime: np.sqrt(1-prime))[0]
    return (-1j*omega*mu0*-1j/4)*(IntAReal+1j*IntAImag+IntBReal+1j*IntBImag)

def EFIE_TE(coordinates,wavelength,angle):
    # The main algorithm
    Ein = DiscritizeEin(coordinates,wavelength,angle)
    Zm = Zmatrix(coordinates,wavelength)
    Jz = np.dot(np.linalg.inv(Zm),Ein)
    # Return all variables of interest as a tuple
    return Jz, Ein, Zm

##-------------------------E field calculation----------------------------
def Etot(Jz,R,coordinates,wavelength,angle):
    # Calculate the total field on given coordinates
    M = len(R)
    Etot = np.zeros(M,dtype=np.complex128)
    for i in np.arange(M):
            r = R[i]
            Esc = Escatter(Jz,r,coordinates,wavelength)
            Ein = Efield_in(r,wavelength,angle)
            Etot[i] = Ein+Esc
    return Etot

def Escatter(Jz,rho,coordinates,wavelength):
    # Calculate the Electric field scattered from the object on given coordinates
    mu0 = 4*pi*10**-7
    c = 299792458
    omega = 2*pi*c/wavelength
    G = np.zeros(len(Jz),dtype=np.complex128) # G can be complex so allocate complex matrix
    # Note length Jz = length coordinates-1
    for i in np.arange(len(Jz)):
        segment = Coordinates_to_segment(coordinates,i)
        GReal = lambda tau: np.real(greenEsc(rho,tau,segment,wavelength))
        GImag = lambda tau: np.imag(greenEsc(rho,tau,segment,wavelength))
        IntReal = integrate.quad(GReal, 0, 1)[0]
        IntImag = integrate.quad(GImag, 0, 1)[0]
        # Correct for the basis function used
        dst = np.linalg.norm(np.subtract(segment[1],segment[0]))
        G[i] = dst*(IntReal + 1j*IntImag)
    return (omega*mu0/4)*np.dot(Jz,G)

def greenEsc(r,tau,segment,wavelength):
    # Special case, only the basis function applied
    k0 = 2*pi/wavelength
    Pi = Pulsebase(tau,segment)
    return hankel2(0,k0*np.linalg.norm(np.subtract(r,Pi)))
