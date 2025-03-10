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
from np import sin ,cos, pi
from sp import integrate
# from sp.special import hankel2
from sp.special import kv
    
# THIS IS A TEST

#-----------Functions used for the Electric field-------------------------

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
        IntReal = integrate.quad(EReal,-1,1)[0]
        IntImag = integrate.quad(EImag,-1,1)[0]
        # Correct for the test function used, see documentation
        # [0], since integrate.quad outputs result[0] and upper bound of error[1]
        dst = segment_length(segment)
        # multiplication with length of segment due to normalization
        Ein[m] =  dst*(IntReal+ 1j*IntImag)
    return Ein

def Coordinates_to_segment(coordinates, set):
    # Take 2 positions in coordinates and return these to define the segment
    if set < len(coordinates)-1:
        # Link the index set of coordinates to the next index
        index = np.array([set, set+1])
    else:
        # If the value of set is invalid, raise an error
        raise ValueError("The requested segment is not found in the boundary points")
    return coordinates[index]

def Pulsebase(tau,segment):
    # Used for the test and basis function, creates a linear connection between the edges of the segment
    xvec=segment[:,0]
    yvec=segment[:,1]
    # define the beginning and ending positions for -1 <= tau <= 1
    x1, x2, y1, y2 = xvec[0], xvec[1], yvec[0], yvec[1]
    Xtau = (1/2)*((1 + tau)*x2 + (1 - tau)*x1)
    Ytau = (1/2)*((1 + tau)*y2 + (1 - tau)*y1)
    # Return the values (x,y) based on tau
    return [Xtau,Ytau]
    
def Efield_in(r, wavelength, angle):
    # Calculate the electric field value based on:
    # the x and y position (in r), wavelength and input angle
    mu0 = pi*4e-7
    epsilon0 = 8.854187812813e-12
    H0 = 1
    E0 = H0*np.sqrt(mu0/epsilon0)# Amplitude of incident wave
    # electric field is normalized to a magnetic field of 1
    x, y = r[0], r[1]
    # Assuming plane wave in losless material
    k0 = 2*pi/wavelength
    return E0*np.exp(1j*k0*(cos(angle)*x + sin(angle)*y))

def segment_length(segment):
    return np.linalg.norm(np.subtract(segment[1], segment[0]))

"""HAS YET TO BE LOOKED AT! DEFINITELY COMPLETELY WRONG!"""
##-----------Functions used for the Z matrix------------------------------
# def Zmatrix(coordinates,wavelength):
#     # Calculate the Z matrix used for the Method of Moments solution
#     # Create a matrix for the number of coordinates that are given
#     M = len(coordinates)-1 #python indexes from 0, unlike MATLAB
#     Z = np.zeros((M,M),dtype=np.complex128) #Z can contain complex values
#     for i in np.arange(M):
#         for m in np.arange(M):
#             # Correct for the test/basis function length used, see documentation
#             segment_m = Coordinates_to_segment(coordinates,m)
#             segment_i = Coordinates_to_segment(coordinates,i)
#             dst_m = np.linalg.norm(np.subtract(segment_m[1],segment_m[0]))
#             dst_i = np.linalg.norm(np.subtract(segment_i[1],segment_i[0]))
#             dst = dst_m*dst_i
#             if m==i:
#                 # The diagonal is singular
#                 Z[m,i] = dst*Z_mi_diag(coordinates,wavelength,m)
#             elif i == m-1 or i==m+1:
#                 # One position is singular for adjacent segments
#                 Z[m,i] = dst*Z_mi_adj(coordinates,wavelength,m,i)
#             else:
#                 # These indices are non-singular
#                 Z[m,i] = dst*Z_mi(coordinates,wavelength,m,i)
#     return Z

def Zmatrix(coordinates,wavelength):
    # Calculate the Z matrix used for the Method of Moments solution
    # Create a matrix for the number of coordinates that are given
    M = len(coordinates)-1 #python indexes from 0, unlike MATLAB
    Z = np.zeros((M,M),dtype=np.complex128) #Z can contain complex values
    for n in np.arange(M):
        for m in np.arange(M):
            # Correct for the test/basis function length used, see documentation
            segment_m = Coordinates_to_segment(coordinates, m)
            segment_n = Coordinates_to_segment(coordinates, n)
            dst_m = segment_length(segment_m)
            dst_n = segment_length(segment_n)
            dst = dst_m*dst_n
            if (abs(coordinates[0] - coordinates[-1]) < 1e-8).all(): # Closed
                if m == n or (m == M-1 and n == 0) or (m == 0 and n == M-1):
                    # The diagonal is singular
                    # The diag function already includes the distance, so we dont need to take it into account here
                    Z[m, n] = Z_mn_diag(coordinates, wavelength, m, n)
                elif abs(m - n) <= 1 or (m == M-2 and n == 0) or (m == M-1 and n == 1) or (m == 0 and n == M-2) or (m == 1 and n == M-1):
                    # One position is singular for adjacent segments
                    Z[m, n] = 0
                else:
                    # These indices are non-singular
                    Z[m, n] = dst*(Z_mn_left(coordinates, wavelength, m, n) + Z_mn_right(coordinates, wavelength, m, n))
            else: # Open
                if m == n:
                    # The diagonal is singular
                    # The diag function already includes the distance, so we dont need to take it into account here
                    Z[m, n] = Z_mn_diag(coordinates, wavelength, m, n)
                elif abs(m - n) <= 1:
                    # One position is singular for adjacent segments
                    Z[m, n] = 0
                else:
                    # These indices are non-singular
                    Z[m, n] = dst*(Z_mn_left(coordinates, wavelength, m, n) + Z_mn_right(coordinates, wavelength, m, n))
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

"""WORK IN PROGRESS"""
# def Z_mn(coordinates, wavelength, m, n):
#     mu0 = 4*pi*10**-7
#     c = 299792458
#     omega = 2*pi*c/wavelength
#     # Integrable Greens function, which is non-singular in this case
#     """DO THE IF STATEMENT FOR OPEN/CLOSE CONDITION!!!"""
#     GReal_1_mm = lambda eta, ksi: np.real(green(eta, ksi, coordinates, wavelength, m, n)*Rooftop_falling(eta, m)*Rooftop_falling(ksi, n))
#     GReal_1_mp = lambda eta, ksi: np.real(green(eta, ksi, coordinates, wavelength, m, n)*Rooftop_falling(eta, m)*Rooftop_rising(ksi, n - 1))
#     GReal_1_pm = lambda eta, ksi: np.real(green(eta, ksi, coordinates, wavelength, m, n)*Rooftop_rising(eta, m - 1)*Rooftop_falling(ksi, n))
#     GReal_1_pp = lambda eta, ksi: np.real(green(eta, ksi, coordinates, wavelength, m, n)*Rooftop_rising(eta, m - 1)*Rooftop_rising(ksi, n - 1))
    
#     GImag_1_mm = lambda eta, ksi: np.imag(green(eta, ksi, coordinates, wavelength, m, n)*Rooftop_falling(eta, m)*Rooftop_falling(ksi, n))
#     GImag_1_mp = lambda eta, ksi: np.imag(green(eta, ksi, coordinates, wavelength, m, n)*Rooftop_falling(eta, m)*Rooftop_rising(ksi, n - 1))
#     GImag_1_pm = lambda eta, ksi: np.imag(green(eta, ksi, coordinates, wavelength, m, n)*Rooftop_rising(eta, m - 1)*Rooftop_falling(ksi, n))
#     GImag_1_pp = lambda eta, ksi: np.imag(green(eta, ksi, coordinates, wavelength, m, n)*Rooftop_rising(eta, m - 1)*Rooftop_rising(ksi, n - 1))
#     # Integrate.dblquad cannot deal with complex number (holds as of 22/03/2021)
#     IntReal_1_mm = integrate.dblquad(GReal_1_mm, -1, 1, -1, 1)[0]
#     IntReal_1_mp = integrate.dblquad(GReal_1_mp, -1, 1, -1, 1)[0]
#     IntReal_1_pm = integrate.dblquad(GReal_1_pm, -1, 1, -1, 1)[0]
#     IntReal_1_pp = integrate.dblquad(GReal_1_pp, -1, 1, -1, 1)[0]
    
#     IntImag_1_mm = integrate.dblquad(GImag_1_mm, -1, 1, -1, 1)[0]
#     IntImag_1_mp = integrate.dblquad(GImag_1_mp, -1, 1, -1, 1)[0]
#     IntImag_1_pm = integrate.dblquad(GImag_1_pm, -1, 1, -1, 1)[0]
#     IntImag_1_pp = integrate.dblquad(GImag_1_pp, -1, 1, -1, 1)[0]
    
#     ans  = (IntReal_1_mm +1j*IntImag_1_mm) + (IntReal_1_mp +1j*IntImag_1_mp) + (IntReal_1_pm +1j*IntImag_1_pm) + (IntReal_1_pp +1j*IntImag_1_pp)
#     return omega*mu0/4*(ans)

# def Z_mi_adj(coordinates,wavelength,m,i):
#     # Special case, Greens function becomes singular on edge points
#     mu0 = 4*pi*10**-7
#     c = 299792458
#     omega = 2*pi*c/wavelength
#     # Use (double) Gauss-legendre quadrature method
#     degree=30
#     xh,wh=np.polynomial.legendre.leggauss(degree)
#     xj=xh #positions for evaluation
#     wj=wh #weights for evaluation
#     ans=0
#     for h in np.arange(degree):
#         for j in np.arange(degree):
#             ans+=wh[h]*wj[j]*green(0.5*(xh[h]+1),0.5*(xj[j]+1),coordinates,wavelength,m,i)
#     return -1j*omega*mu0*ans/4

# def Z_mi_diag(coordinates,wavelength,m):
#     # Special case, Green's function becomes singular so requires special integration
#     # Due to the change of basis cannot use the Green's function defined earlier, see documentation
#     k0 = 2*pi/wavelength
#     mu0 = 4*pi*10**-7
#     c = 299792458
#     omega = 2*pi*c/wavelength
#     # Obtain the segment used for the integral
#     segment = Coordinates_to_segment(coordinates,m)
#     dst = np.linalg.norm(np.subtract(segment[1],segment[0]))
#     # First define the variable for inner integral then outer
#     # If prime is not defined here, integraton will not work
#     QReal = lambda nu, prime: np.real(2*nu*hankel2(0,k0*nu**2*dst))
#     QImag = lambda nu, prime: np.imag(2*nu*hankel2(0,k0*nu**2*dst))
#     # For argument in boundary first give outer integral then inner boundary
#     IntAReal = integrate.dblquad(QReal, 0, 1, 0, lambda prime: np.sqrt(prime))[0]
#     IntAImag = integrate.dblquad(QImag, 0, 1, 0, lambda prime: np.sqrt(prime))[0]
#     IntBReal = integrate.dblquad(QReal, 0, 1, 0, lambda prime: np.sqrt(1-prime))[0]
#     IntBImag = integrate.dblquad(QImag, 0, 1, 0, lambda prime: np.sqrt(1-prime))[0]
#     return (-1j*omega*mu0*-1j/4)*(IntAReal+1j*IntAImag+IntBReal+1j*IntBImag)

def Z_mn_left(coordinates, wavelength, m, n):
    # Calculate the non-singular components of the Z matrix
    mu0 = 4 * pi * 10**-7
    c = 299792458
    omega = 2 * np.pi * c / wavelength
    
    # The indices are defined differently for closed and open surfaces (in the code only),
    # so we have to take that into account by using a -2 instead of a -1 in the rising function
    if (abs(coordinates[0] - coordinates[-1]) < 1e-8).all(): # Closed
        def G(eta, ksi, mode, m, n):
            # Intermediate function to calculate Green's function together with
            # the test & basis function to make the code more readable
            
            # Here, mm, mp, pm, and pp are the 4 overlapping options of triangle
            # functions. with m=minus and p=plus
            
            # In this function, m == 1 & n == 1 can't happen together -> non-singular!
            if m == 0:
                basis_funcs = {
                    # m - 2 instead of m - 1, as the first coord == last coord
                    "mm": (Rooftop_falling(eta, m), Rooftop_falling(ksi, n)),
                    "mp": (Rooftop_falling(eta, m), Rooftop_rising(ksi, n - 1)),
                    "pm": (Rooftop_rising(eta, m - 2), Rooftop_falling(ksi, n)),
                    "pp": (Rooftop_rising(eta, m - 2), Rooftop_rising(ksi, n - 1))
                }
            elif n == 0:
              basis_funcs = {
                  # n - 2 instead of n - 1, as the first coord == last coord
                  "mm": (Rooftop_falling(eta, m), Rooftop_falling(ksi, n)),
                  "mp": (Rooftop_falling(eta, m), Rooftop_rising(ksi, n - 2)),
                  "pm": (Rooftop_rising(eta, m - 1), Rooftop_falling(ksi, n)),
                  "pp": (Rooftop_rising(eta, m - 1), Rooftop_rising(ksi, n - 2))
              }  
            else:
                basis_funcs = {
                    "mm": (Rooftop_falling(eta, m), Rooftop_falling(ksi, n)),
                    "mp": (Rooftop_falling(eta, m), Rooftop_rising(ksi, n - 1)),
                    "pm": (Rooftop_rising(eta, m - 1), Rooftop_falling(ksi, n)),
                    "pp": (Rooftop_rising(eta, m - 1), Rooftop_rising(ksi, n - 1))
                }
            # Define the test/basis function depending on what mode we need
            tf_eta, bf_ksi = basis_funcs[mode]
            return green(eta, ksi, coordinates, wavelength, m, n) * tf_eta * bf_ksi
    else: # Open
        def G(eta, ksi, mode, m, n):
            # TODO
            return 0
    
    modes = ["mm", "mp", "pm", "pp"]
    integrals = {"real": {}, "imag": {}}
    for mode in modes:
        G_real = lambda eta, ksi: np.real(G(eta, ksi, mode, m, n))
        G_imag = lambda eta, ksi: np.imag(G(eta, ksi, mode, m, n))

        integrals["real"][mode] = integrate.dblquad(G_real, -1, 1, -1, 1)[0]
        integrals["imag"][mode] = integrate.dblquad(G_imag, -1, 1, -1, 1)[0]
        
    return 1j*omega*mu0*sum( (integrals["real"][mode] + 1j*integrals["imag"][mode]) for mode in modes )

def Z_mn_right(coordinates, wavelength, m, n):
    epsilon0 = 8.854187812813e-12
    c = 299792458
    omega = 2*pi*c/wavelength
    
    # The indices are defined differently for closed and open surfaces (in the code only),
    # so we have to take that into account by using a -2 instead of a -1 in the rising function
    if (abs(coordinates[0] - coordinates[-1]) < 1e-8).all(): # Closed
        def G(eta, ksi, mode, m, n):
            # Intermediate function to calculate Green's function together with
            # the test & basis function to make the code more readable
            
            # Here, mm, mp, pm, and pp are the 4 overlapping options of triangle
            # functions. with m=minus and p=plus
            
            # In this function, m == 1 & n == 1 can't happen together -> non-singular!
            basis_funcs = {
                "mm": (-1, -1),
                "mp": (-1, 1),
                "pm": (1, -1),
                "pp": (1, 1)
            }
            # Define the test/basis function depending on what mode we need
            tf_eta, bf_ksi = basis_funcs[mode]
            return green(eta, ksi, coordinates, wavelength, m, n) * tf_eta/2 * bf_ksi/2
    else: # Open
        def G(eta, ksi, mode, m, n):
            # TODO
            return 0

    modes = ["mm", "mp", "pm", "pp"]
    integrals = {"real": {}, "imag": {}}
    for mode in modes:
        G_real = lambda eta, ksi: np.real(G(eta, ksi, mode, m, n))
        G_imag = lambda eta, ksi: np.imag(G(eta, ksi, mode, m, n))

        integrals["real"][mode] = integrate.dblquad(G_real, -1, 1, -1, 1)[0]
        integrals["imag"][mode] = integrate.dblquad(G_imag, -1, 1, -1, 1)[0]
        
    return (-1)/(1j*omega*epsilon0)*sum( (integrals["real"][mode] + 1j*integrals["imag"][mode]) for mode in modes )

def Z_mn_adj(coordinates, wavelength, m, n):
    mu0 = 4 * pi * 10**-7
    epsilon0 = 8.854187812813e-12
    c = 299792458
    omega = 2*pi*c/wavelength
    
    Is = Self_term_integral(coordinates, wavelength, m, n) 
    # Minus instead of plus for Is[1], as the etaksi term of f(eta, ksi) is now negative, same for the last Is[0] term
    return 1j*omega*mu0/(2*pi)*Is[0] - 1j*omega*mu0/(2*pi)*Is[1] + 1/(1j*omega*epsilon0)*(1/(2*pi))*(1/4)*Is[0]

def Z_mn_diag(coordinates, wavelength, m, n):
    mu0 = 4 * pi * 10**-7
    epsilon0 = 8.854187812813e-12
    c = 299792458
    omega = 2*pi*c/wavelength
    
    Is = Self_term_integral(coordinates, wavelength, m, n) 
    # Is returns [f(1), f(eta*ksi)], so for the left integral, we need both terms, while the right integral only needs 1
    return 1j*omega*mu0/(2*pi)*sum(Is) - 1/(1j*omega*epsilon0)*(1/(2*pi))*(1/4)*Is[0]
    
def Self_term_integral(coordinates, wavelength, m, n):
    # This function gives the solution to the self integral term defined as:
    # dm*dm * int_-1^1 int_-1^1 f(eta, ksi)*kv(gamma*d*|ksi - eta|)
    # with f(eta, ksi)= (1 + eta + ksi + eta*ksi) or (1 - eta - ksi + eta*ksi)
    # However, since the eta and ksi term fall away, both cases SHOULD BE the same
    k0 = 2*pi/wavelength
    
    segment = Coordinates_to_segment(coordinates, m)
    d = segment_length(segment)
    p = d*1j*k0 # Here, gamma is jk0, as that is how we defined the inside of the modified besself function (same as j*omega*sqrt(e0*mu0), as paper)
    
    Is_etaksi = 0
    for k in np.arange(20): # Should go to inf, but I dont see a difference between 10 and 30, so 20 it is.
        numerator_etaksi = 2 - k*(2*(1 + k)*(1 + 2*k)*(2 + k)*(Harmonic_number(k) - np.log(p)) + k*(7 + 4*k))
        denominator_etaksi = np.multiply(np.multiply((1 + k)**2, (1 + 2*k)**2, dtype=object), np.multiply((2 + k)**2, (Factorial(k))**2, dtype=object), dtype=object)
        Is_etaksi += np.divide(np.multiply(numerator_etaksi, p**(2*k), dtype=object), denominator_etaksi, dtype=object)
        
    Is_1 = 0
    for k in np.arange(20):
        numerator_1 = 3 + 4*k + 2*(1 + k)*(1 + 2*k)*(Harmonic_number(k) - np.log(p))
        denominator_1 = np.multiply(np.multiply((1 + k)**2, (1 + 2*k)**2, dtype=object), (Factorial(k))**2, dtype=object)    
        Is_1 += np.divide(np.multiply(numerator_1, p**(2*k), dtype=object), denominator_1, dtype=object)
        
    return np.array([2*d**2 * Is_1, 2*d**2 * Is_etaksi])

def Harmonic_number(k):
    gamma_e = 0.57721566490153286060 # Eulers constant
    if k == 0:
        return -gamma_e
    elif k > 0:
        ans = 0
        for n in np.arange(k)+1:
            ans += 1/n 
        return -gamma_e + ans

def Factorial(n):
    ans = 1
    for i in range(2, n + 1):
        ans *= i
    return ans

def Rooftop_rising(tau, i):
    # For the rising triangle (/\^+), the shape is given as |\, where tau is
    # either Xi or Eta depending on if we use it for test or basis functions, respecitvely
    return (1/2)*(1 + tau)

def Rooftop_falling(tau, i):
    # For the decending triangle (/\^-), the shape is given as /|, where tau is
    # either Xi or Eta depending on if we use it for test or basis functions, respecitvely
    return (1/2)*(1 - tau)

def green(eta, ksi, coordinates, wavelength, m, n):
    # The Green's function for this specific case, represented using the Hankel Function
    # Note this function only works for non-singular arguments (m!=n)
    # Here, Tm is the m-segment test function, and Bn is the n-segment basis function,
    # which are parameterized by eta and ksi respectively
    if m == n:
        raise ValueError("Singular case, use a different function!")
    k0 = 2*pi/wavelength
    # Obtain the arguments for the Greens function, based on the test/basis function
    # Shape should not matter in this case, as we are only intrested in the distance
    Tm = Pulsebase(eta, Coordinates_to_segment(coordinates, m))
    Bn = Pulsebase(ksi, Coordinates_to_segment(coordinates, n))
    dst = segment_length(np.array([Tm, Bn]))
    return 1/(2*pi)*kv(0, 1j*k0*dst)

def EFIE_TM(coordinates,wavelength,angle):
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

# def Escatter(Jz,rho,coordinates,wavelength):
#     # Calculate the Electric field scattered from the object on given coordinates
#     mu0 = 4*pi*10**-7
#     c = 299792458
#     omega = 2*pi*c/wavelength
#     G = np.zeros(len(Jz),dtype=np.complex128) # G can be complex so allocate complex matrix
#     # Note length Jz = length coordinates-1
#     for i in np.arange(len(Jz)):
#         segment = Coordinates_to_segment(coordinates,i)
#         GReal = lambda tau: np.real(greenEsc(rho,tau,segment,wavelength))
#         GImag = lambda tau: np.imag(greenEsc(rho,tau,segment,wavelength))
#         IntReal = integrate.quad(GReal, 0, 1)[0]
#         IntImag = integrate.quad(GImag, 0, 1)[0]
#         # Correct for the basis function used
#         dst = np.linalg.norm(np.subtract(segment[1],segment[0]))
#         G[i] = dst*(IntReal + 1j*IntImag)
#     return (omega*mu0/4)*np.dot(Jz,G)

def Escatter(Jz, rho, coordinates, wavelength):
    # Calculate the Electric field scattered from the object on given coordinates
    mu0 = 4*pi*10**-7
    epsilon0 = 8.854187812813e-12
    c = 299792458
    omega = 2*pi*c/wavelength
    G = np.zeros(len(Jz), dtype=np.complex128) # G can be complex so allocate complex matrix
    GTriangle = np.zeros(len(Jz), dtype=np.complex128)
    # Note length Jz = length coordinates-1
    for i in np.arange(len(Jz)):
        segment = Coordinates_to_segment(coordinates,i)
        GReal = lambda tau: np.real(greenEsc(rho,tau,segment,wavelength))
        GImag = lambda tau: np.imag(greenEsc(rho,tau,segment,wavelength))
        IntReal = integrate.quad(GReal, -1, 1)[0]
        IntImag = integrate.quad(GImag, -1, 1)[0]
        
        GRealTriangle = lambda tau: np.real(greenEscTriangle(rho,tau,segment,wavelength))
        GImagTriangle = lambda tau: np.imag(greenEscTriangle(rho,tau,segment,wavelength))
        IntRealTriangle = integrate.quad(GRealTriangle, -1, 1)[0]
        IntImagTriangle = integrate.quad(GImagTriangle, -1, 1)[0]
        # Correct for the basis function used
        dst = segment_length(segment)
        G[i] = dst*(IntReal + 1j*IntImag)
        GTriangle[i] = dst*(IntRealTriangle + 1j*IntImagTriangle)
    return (1j*omega*mu0/(2*pi))*np.dot(Jz,GTriangle) - (1/(1j*omega*epsilon0*2*pi))*np.dot(Jz,G)

def greenEsc(r,tau,segment,wavelength):
    # Special case, only the basis function applied
    k0 = 2*pi/wavelength
    Pn = Pulsebase(tau,segment)
    return kv(0, 1j*k0*np.linalg.norm(np.subtract(r,Pn)))

def greenEscTriangle(r,tau,segment,wavelength):
    # Special case, only the basis function applied
    k0 = 2*pi/wavelength
    Pn = Pulsebase(tau,segment)
    return Rooftop_falling(tau, 2)*kv(0, 1j*k0*np.linalg.norm(np.subtract(r,Pn))) + Rooftop_rising(tau, 2)*kv(0, 1j*k0*np.linalg.norm(np.subtract(r,Pn)))
