# ----------------------------------------------------------------------------#
# This code solves the Electric Field Integral Equation (EFIE) for a 2D countor
# for a TM incident electric field.
#
# Authors:
#   - Max van Wijk
#   - ...


# Imports used for the code
import numpy as np
from numpy import sin ,cos, pi, sqrt, exp
from scipy import integrate
from scipy.special import kv, psi, factorial

def createcircle(M,R):
    # generate circle with radius R with M samples
    Arg = np.linspace(0,2*pi,M)
    Data = np.zeros((M,2))
    for i in np.arange(M):
        Data[i,:] = np.asarray([R*cos(Arg[i]),R*sin(Arg[i])])
    return Data

## FUNCTION DEFINITIONS
def DiscritizeEin(coordinates, wavelength, angle):
    # Create two arrays for the number of segments
    # the code does not close the contour, do so manually
    M = len(coordinates)-1
    Ein_x = np.zeros(M, dtype=np.complex128)
    Ein_y = np.zeros(M, dtype=np.complex128)
    
    # values of Ein_x & Ein_y are complex so matrix needs to be able to handle complex values
    # Overwrite each datapoint with the actual value of the Ein field
    for m in np.arange(M):
        # Sample the E field for varying coordinates, based on the testing Function
        # The loop goes over segments between 2 coordinates inside the array
        # Integrate.quad cannot deal with complex numbers (holds as of 22/03/2021)
        seg = Coordinates_to_segment(coordinates, m)
        [tau_x, tau_y] = Tangent_vector_coefficients(seg)
        
        # Define real and imaginary components of X and Y separately
        # Tangent vectors added due to the crossproduct with nu (see documentation)
        EReal_x = lambda eta: tau_x*np.real(Efield_in(rho(eta, seg), wavelength, angle)[0])
        EImag_x = lambda eta: tau_x*np.imag(Efield_in(rho(eta, seg), wavelength, angle)[0])
        EReal_y = lambda eta: tau_y*np.real(Efield_in(rho(eta, seg), wavelength, angle)[1])
        EImag_y = lambda eta: tau_y*np.imag(Efield_in(rho(eta, seg), wavelength, angle)[1])
        
        # Perform the integration
        IntReal_x = integrate.quad(EReal_x, -1, 1)[0]
        IntImag_x = integrate.quad(EImag_x, -1, 1)[0]
        IntReal_y = integrate.quad(EReal_y, -1, 1)[0]
        IntImag_y = integrate.quad(EImag_y, -1, 1)[0]
        
        # Normalize by segment length
        dst = segment_length(seg)
        Ein_x[m] = dst*(IntReal_x + 1j*IntImag_x)
        Ein_y[m] = dst*(IntReal_y + 1j*IntImag_y)
    
    return Ein_x, Ein_y

def Efield_in(r, wavelength, angle):
    # Calculate the electric field value based on:
    # the x and y position (in r), wavelength and input angle
    mu0 = pi*4e-7
    epsilon0 = 8.854187812813e-12
    H0 = 1
    E0 = H0*sqrt(mu0/epsilon0)
    
    x, y = r[0], r[1]
    k0 = 2*pi/wavelength
    
    # Compute Ex and Ey separately
    Ex = E0*cos(angle)*exp(1j*k0*(cos(angle)*x + sin(angle)*y))
    Ey = E0*sin(angle)*exp(1j*k0*(cos(angle)*x + sin(angle)*y))
    
    return Ex, Ey
        
        

# def DiscritizeEin(coordinates,wavelength,angle):
#     # Create an array for the number of segments
#     # the code does not close the contour, do so manually
#     M = len(coordinates)-1
#     Ein = np.zeros(M,dtype=np.complex128)
#     # values of Ein are complex so matrix needs to be able to handle complex values
#     # Overwrite each datapoint with the actual value of the Ein field
#     for m in np.arange(M):
#         # Sample the E field for varying coordinates, based on the testing Function
#         # The loop goes over segments between 2 coordinates inside the array
#         # Integrate.quad cannot deal with complex numbers (holds as of 22/03/2021)
#         segment = Coordinates_to_segment(coordinates,m)
#         [nu_x, nu_y] = Normal_vector_coefficients(segment)
#         EReal = lambda eta: np.real(Efield_in(rho(eta, segment),wavelength,angle))
#         EImag = lambda eta: np.imag(Efield_in(rho(eta, segment),wavelength,angle))
#         IntReal = integrate.quad(EReal,-1,1)[0]
#         IntImag = integrate.quad(EImag,-1,1)[0]
#         # Correct for the test function used, see documentation
#         # [0], since integrate.quad outputs result[0] and upper bound of error[1]
#         dst = segment_length(segment)
#         # multiplication with length of segment due to normalization
#         Ein[m] =  dst*(IntReal+ 1j*IntImag)
#     return Ein

# def Efield_in(r,wavelength,angle):
#     # Calculate the electric field value based on:
#     # the x and y position (in r), wavelength and input angle
#     mu0 = pi*4e-7
#     epsilon0 = 8.854187812813e-12
#     H0 = 1
#     E0 = H0*sqrt(mu0/epsilon0) # Amplitude of incident wave
#     # Electric field is normalized to a magnetic field of 1
#     x, y = r[0], r[1]
#     # Assuming plane wave in losless material
#     k0 = 2*pi/wavelength
#     return E0*exp(1j*k0*(cos(angle)*x + sin(angle)*y))

def closed(coordinates):
    return (abs(coordinates[0] - coordinates[-1]) < 1e-8).all()

def Coordinates_to_segment(coordinates, set):
    # Take 2 positions in coordinates and return these to define the segment
    if set < len(coordinates) - 1:
        # Link the index set of coordinates to the next index
        index = np.array([set, set + 1])
    elif set == len(coordinates) - 1:
        index = np.array([set, 1])
    else:
        # If the value of set is invalid, raise an error
        raise ValueError("The requested segment is not found in the boundary points")
    return coordinates[index]

def Segment_center(segment):
    return (segment[0] + segment[1]) / 2

def segment_length(segment):
    return np.linalg.norm(np.subtract(segment[1], segment[0]))

def Rooftop_rising(tau):
    # For the rising triangle (/\^+), the shape is given as |\, where tau is
    # either Xi or Eta depending on if we use it for test or basis functions, respecitvely
    return (1/2)*(1 + tau)

def Rooftop_falling(tau):
    # For the decending triangle (/\^-), the shape is given as /|, where tau is
    # either Xi or Eta depending on if we use it for test or basis functions, respecitvely
    return (1/2)*(1 - tau)

def Tangent_vector(segment):
    x0, y0 = segment[0]
    x1, y1 = segment[1]
    dx, dy = x1 - x0, y1 - y0
    length = segment_length(segment)
    
    # Compute unit tangent (dy, dx) and normalize
    tau = np.array([dx / length, dy / length])
    
    return tau

def Tangent_vector_coefficients(segment):
    midpoint = Segment_center(segment)
    tangent = Tangent_vector(segment)
    
    x1, x2 = midpoint[0], midpoint[0] + tangent[0]
    y1, y2 = midpoint[1], midpoint[1] + tangent[1]
    
    taux, tauy = x2 - x1, y2 - y1
    
    return [taux, tauy]

def Normal_vector(segment):
    x0, y0 = segment[0]
    x1, y1 = segment[1]
    dx, dy = x1 - x0, y1 - y0
    length = segment_length(segment)
    
    # Compute unit normal (dy, -dx) and normalize
    nu = np.array([dy / length, -dx / length])      
    
    return nu

def Normal_vector_coefficients(segment):
    midpoint = Segment_center(segment)
    normal = Normal_vector(segment)
    
    x1, x2 = midpoint[0], midpoint[0] + normal[0]
    y1, y2 = midpoint[1], midpoint[1] + normal[1]
    
    nux, nuy = x2 - x1, y2 - y1
    
    return [nux, nuy]

def rho(t, segment):
    # Used for the test and basis function, creates a linear connection between the edges of the segment
    xvec = segment[:,0]
    yvec = segment[:,1]
    # define the beginning and ending positions
    x1, x2, y1, y2 = xvec[0], xvec[1], yvec[0], yvec[1]
    Xtau = (1/2)*((1 + t)*x2 + (1 - t)*x1)
    Ytau = (1/2)*((1 + t)*y2 + (1 - t)*y1)
    # Return the values (x,y) based on tau
    return [Xtau,Ytau]

def Zmn_calculator(coordinates, wavelength):
    #M = len(coordinates)-1
    Zmn, ZmnLeft, ZmnRight = Zmn_calculator_2x2matrix_method(coordinates, wavelength)
    Zmn_diagonal_terms = Zmn_diag_calculator(coordinates, wavelength)
    
    np.fill_diagonal(Zmn, Zmn_diagonal_terms.flatten())
    
    return Zmn

# def Zmn_calculator_single_element_method(coordinates, wavelength):
#     M = len(coordinates)-1
#     k0 = 2*pi/wavelength
#     mu0 = 4*pi*10**-7
#     epsilon0 = 8.854187812813e-12
#     c = 299792458
#     omega = 2*pi*c/wavelength
    
#     Zmn_left = np.zeros((M,M), dtype=np.complex128)
#     Zmn_right = np.zeros((M,M), dtype=np.complex128)
    
#     Zmn_left_PP = np.zeros((M,M), dtype=np.complex128)
#     Zmn_left_MM = np.zeros((M,M), dtype=np.complex128)
#     Zmn_left_PM = np.zeros((M,M), dtype=np.complex128)
#     Zmn_left_MP = np.zeros((M,M), dtype=np.complex128)
    
#     Zmn_right_PP = np.zeros((M,M), dtype=np.complex128)
#     Zmn_right_MM = np.zeros((M,M), dtype=np.complex128)
#     Zmn_right_PM = np.zeros((M,M), dtype=np.complex128)
#     Zmn_right_MP = np.zeros((M,M), dtype=np.complex128)
    
#     for n in np.arange(M):
#         print(n) 
#         for m in np.arange(M):
#             # Calculate the segment for both m & n
#             segm_P = Coordinates_to_segment(coordinates, m)
#             segn_P = Coordinates_to_segment(coordinates, n)
            
#             # Also calculate the previous segment, where we need to take note of
#             # the -1th segment, which shifts over one (in closed)
#             # TODO: ADD OPEN CASE ASWELL!!
#             if m == 0:
#                 segm_M = Coordinates_to_segment(coordinates, -2)
#             else:
#                 segm_M = Coordinates_to_segment(coordinates, m-1)
            
#             if n == 0:
#                 segn_M = Coordinates_to_segment(coordinates, -2)
#             else:
#                 segn_M = Coordinates_to_segment(coordinates, n-1)
            
#             dstm_P = segment_length(segm_P)/2
#             dstn_P = segment_length(segn_P)/2
            
#             dstm_M = segment_length(segm_M)/2
#             dstn_M = segment_length(segn_M)/2
            
#             # Create a function which calculates the norm inside of the Modified Bessel Function
#             dstrho = lambda eta, ksi, segm, segn: np.linalg.norm(np.subtract(rho(eta, segm), rho(ksi, segn)))
            
#             tauxm_P, tauym_P = Tangent_vector_coefficients(segm_P)
#             tauxn_P, tauyn_P = Tangent_vector_coefficients(segn_P)
            
#             tauxm_M, tauym_M = Tangent_vector_coefficients(segm_M)
#             tauxn_M, tauyn_M = Tangent_vector_coefficients(segn_M)
            
#             # Calculate the 4 integrants, both for the real and imaginary part
#             integrantPP_real = lambda eta, ksi: 0.5*(1 + eta)*0.5*(1 + ksi)*(1/(2*pi))*np.real(kv(0, 1j*k0*dstrho(eta, ksi, segm_P, segn_P)))
#             integrantPP_imag = lambda eta, ksi: 0.5*(1 + eta)*0.5*(1 + ksi)*(1/(2*pi))*np.imag(kv(0, 1j*k0*dstrho(eta, ksi, segm_P, segn_P)))
#             integrantMM_real = lambda eta, ksi: 0.5*(1 - eta)*0.5*(1 - ksi)*(1/(2*pi))*np.real(kv(0, 1j*k0*dstrho(eta, ksi, segm_M, segn_M)))
#             integrantMM_imag = lambda eta, ksi: 0.5*(1 - eta)*0.5*(1 - ksi)*(1/(2*pi))*np.imag(kv(0, 1j*k0*dstrho(eta, ksi, segm_M, segn_M)))
#             integrantPM_real = lambda eta, ksi: 0.5*(1 + eta)*0.5*(1 - ksi)*(1/(2*pi))*np.real(kv(0, 1j*k0*dstrho(eta, ksi, segm_P, segn_M)))
#             integrantPM_imag = lambda eta, ksi: 0.5*(1 + eta)*0.5*(1 - ksi)*(1/(2*pi))*np.imag(kv(0, 1j*k0*dstrho(eta, ksi, segm_P, segn_M)))
#             integrantMP_real = lambda eta, ksi: 0.5*(1 - eta)*0.5*(1 + ksi)*(1/(2*pi))*np.real(kv(0, 1j*k0*dstrho(eta, ksi, segm_M, segn_P)))
#             integrantMP_imag = lambda eta, ksi: 0.5*(1 - eta)*0.5*(1 + ksi)*(1/(2*pi))*np.imag(kv(0, 1j*k0*dstrho(eta, ksi, segm_M, segn_P)))
            
#             # Remove diagonal terms from the calculations
#             if(m == n or m == n+1 or (n == 0 and m == M-1) or n == m+1 or (m == 0 and n == M-1)):
#                 intPP_real = 0
#                 intPP_imag = 0
#                 intMM_real = 0
#                 intMM_imag = 0
#                 intPM_real = 0
#                 intPM_imag = 0
#                 intMP_real = 0
#                 intMP_imag = 0
#             else:
#                 intPP_real = integrate.dblquad(integrantPP_real, -1, 1, -1, 1)[0]
#                 intPP_imag = integrate.dblquad(integrantPP_imag, -1, 1, -1, 1)[0]
#                 intMM_real = integrate.dblquad(integrantMM_real, -1, 1, -1, 1)[0]
#                 intMM_imag = integrate.dblquad(integrantMM_imag, -1, 1, -1, 1)[0]
#                 intPM_real = integrate.dblquad(integrantPM_real, -1, 1, -1, 1)[0]
#                 intPM_imag = integrate.dblquad(integrantPM_imag, -1, 1, -1, 1)[0]
#                 intMP_real = integrate.dblquad(integrantMP_real, -1, 1, -1, 1)[0]
#                 intMP_imag = integrate.dblquad(integrantMP_imag, -1, 1, -1, 1)[0]
            
#             Zmn_left_PP[m,n]   = dstm_P*dstn_P*1j*omega*mu0*(tauxm_P*tauxn_P + tauym_P*tauyn_P)*(intPP_real + 1j*intPP_imag)
#             Zmn_left_MM[m,n]   = dstm_M*dstn_M*1j*omega*mu0*(tauxm_M*tauxn_M + tauym_M*tauyn_M)*(intMM_real + 1j*intMM_imag)
#             Zmn_left_PM[m,n]   = dstm_P*dstn_M*1j*omega*mu0*(tauxm_P*tauxn_M + tauym_P*tauyn_M)*(intPM_real + 1j*intPM_imag)
#             Zmn_left_MP[m,n]   = dstm_P*dstn_M*1j*omega*mu0*(tauxm_M*tauxn_P + tauym_M*tauyn_P)*(intMP_real + 1j*intMP_imag)
            
#             Zmn_left[m,n] = Zmn_left_PP[m,n] + Zmn_left_MM[m,n] + Zmn_left_PM[m,n] + Zmn_left_MP[m,n]
                
#             Zmn_right_PP[m,n]  = dstm_P*dstn_P*(tauxm_P**2 + tauym_P**2)/(2*pi*1j*omega*epsilon0)*(intPP_real + 1j*intPP_imag)
#             Zmn_right_MM[m,n]  = dstm_M*dstn_M*(tauxm_M**2 + tauym_M**2)/(2*pi*1j*omega*epsilon0)*(intMM_real + 1j*intMM_imag)
#             Zmn_right_PM[m,n]  = dstm_P*dstn_M*(tauxm_P**2 + tauym_M**2)/(2*pi*1j*omega*epsilon0)*(intPM_real + 1j*intPM_imag)
#             Zmn_right_MP[m,n]  = dstm_M*dstn_P*(tauxm_M**2 + tauym_P**2)/(2*pi*1j*omega*epsilon0)*(intMP_real + 1j*intMP_imag)
            
#             Zmn_right[m,n] = Zmn_right_PP[m,n] + Zmn_right_MM[m,n] + Zmn_right_PM[m,n] + Zmn_right_MP[m,n]
            
#     Zmn = Zmn_left + Zmn_right
    
#     return Zmn, Zmn_left, Zmn_right#, Zmn_left_PP, Zmn_left_MM, Zmn_left_PM, Zmn_left_MP, Zmn_right_PP, Zmn_right_MM, Zmn_right_PM, Zmn_right_MP

def Zmn_calculator_2x2matrix_method(coordinates, wavelength):
    M = len(coordinates)-1
    k0 = 2*pi/wavelength
    mu0 = 4*pi*10**-7
    epsilon0 = 8.854187812813e-12
    c = 299792458
    omega = 2*pi*c/wavelength
    
    Zmn_left = np.zeros((M,M), dtype=np.complex128)
    Zmn_right = np.zeros((M,M), dtype=np.complex128)
    
    for n in np.arange(M):
        print(n) 
        for m in np.arange(M):
            # Calculate the segment for both m & n
            segm_P = Coordinates_to_segment(coordinates, m)
            segn_P = Coordinates_to_segment(coordinates, n)
            
            # Also calculate the previous segment, where we need to take note of
            # the -1th segment, which shifts over one (in closed)
            # TODO: ADD OPEN CASE ASWELL!!
            if m == 0:
                segm_M = Coordinates_to_segment(coordinates, -2)
            else:
                segm_M = Coordinates_to_segment(coordinates, m-1)
            
            if n == 0:
                segn_M = Coordinates_to_segment(coordinates, -2)
            else:
                segn_M = Coordinates_to_segment(coordinates, n-1)
            
            dstm_P = segment_length(segm_P)/2
            dstn_P = segment_length(segn_P)/2
            
            dstm_M = segment_length(segm_M)/2
            dstn_M = segment_length(segn_M)/2
            
            # Create a function which calculates the norm inside of the Modified Bessel Function
            dstrho = lambda eta, ksi, segm, segn: np.linalg.norm(np.subtract(rho(eta, segm), rho(ksi, segn)))
            
            tauxm_P, tauym_P = Tangent_vector_coefficients(segm_P)
            tauxn_P, tauyn_P = Tangent_vector_coefficients(segn_P)
            
            tauxm_M, tauym_M = Tangent_vector_coefficients(segm_M)
            tauxn_M, tauyn_M = Tangent_vector_coefficients(segn_M)
            
            # Calculate the 4 integrants, both for the real and imaginary part
            integrantPP_real = lambda eta, ksi: 0.5*(1 + eta)*0.5*(1 + ksi)*(1/(2*pi))*np.real(kv(0, 1j*k0*dstrho(eta, ksi, segm_P, segn_P)))
            integrantPP_imag = lambda eta, ksi: 0.5*(1 + eta)*0.5*(1 + ksi)*(1/(2*pi))*np.imag(kv(0, 1j*k0*dstrho(eta, ksi, segm_P, segn_P)))
            integrantMM_real = lambda eta, ksi: 0.5*(1 - eta)*0.5*(1 - ksi)*(1/(2*pi))*np.real(kv(0, 1j*k0*dstrho(eta, ksi, segm_M, segn_M)))
            integrantMM_imag = lambda eta, ksi: 0.5*(1 - eta)*0.5*(1 - ksi)*(1/(2*pi))*np.imag(kv(0, 1j*k0*dstrho(eta, ksi, segm_M, segn_M)))
            integrantPM_real = lambda eta, ksi: 0.5*(1 + eta)*0.5*(1 - ksi)*(1/(2*pi))*np.real(kv(0, 1j*k0*dstrho(eta, ksi, segm_P, segn_M)))
            integrantPM_imag = lambda eta, ksi: 0.5*(1 + eta)*0.5*(1 - ksi)*(1/(2*pi))*np.imag(kv(0, 1j*k0*dstrho(eta, ksi, segm_P, segn_M)))
            integrantMP_real = lambda eta, ksi: 0.5*(1 - eta)*0.5*(1 + ksi)*(1/(2*pi))*np.real(kv(0, 1j*k0*dstrho(eta, ksi, segm_M, segn_P)))
            integrantMP_imag = lambda eta, ksi: 0.5*(1 - eta)*0.5*(1 + ksi)*(1/(2*pi))*np.imag(kv(0, 1j*k0*dstrho(eta, ksi, segm_M, segn_P)))
            
            integrantPP_real_R = lambda eta, ksi: 0.5*0.5*(1/(2*pi))*np.real(kv(0, 1j*k0*dstrho(eta, ksi, segm_P, segn_P)))
            integrantPP_imag_R = lambda eta, ksi: 0.5*0.5*(1/(2*pi))*np.imag(kv(0, 1j*k0*dstrho(eta, ksi, segm_P, segn_P)))
            integrantMM_real_R = lambda eta, ksi: -0.5*-0.5*(1/(2*pi))*np.real(kv(0, 1j*k0*dstrho(eta, ksi, segm_M, segn_M)))
            integrantMM_imag_R = lambda eta, ksi: -0.5*-0.5*(1/(2*pi))*np.imag(kv(0, 1j*k0*dstrho(eta, ksi, segm_M, segn_M)))
            integrantPM_real_R = lambda eta, ksi: 0.5*-0.5*(1/(2*pi))*np.real(kv(0, 1j*k0*dstrho(eta, ksi, segm_P, segn_M)))
            integrantPM_imag_R = lambda eta, ksi: 0.5*-0.5*(1/(2*pi))*np.imag(kv(0, 1j*k0*dstrho(eta, ksi, segm_P, segn_M)))
            integrantMP_real_R = lambda eta, ksi: -0.5*0.5*(1/(2*pi))*np.real(kv(0, 1j*k0*dstrho(eta, ksi, segm_M, segn_P)))
            integrantMP_imag_R = lambda eta, ksi: -0.5*0.5*(1/(2*pi))*np.imag(kv(0, 1j*k0*dstrho(eta, ksi, segm_M, segn_P)))
            
            # Remove diagonal terms from the calculations
            if(m == n or m-1 == n or (n == 0 and m == M-1) or n-1 == m or (m == 0 and n == M-1)):
                intPP_real, intPP_imag = 0, 0
                intMM_real, intMM_imag = 0, 0
                intPM_real, intPM_imag = 0, 0
                intMP_real, intMP_imag = 0, 0
                intPP_real_R, intPP_imag_R = 0, 0
                intMM_real_R, intMM_imag_R = 0, 0
                intPM_real_R, intPM_imag_R = 0, 0
                intMP_real_R, intMP_imag_R = 0, 0
            # elif(m-1 == n or (n == 0 and m == M-1)): # Super-diagonal
            #     intPP_real = 0
            #     intPP_imag = 0
            #     intMM_real = 0
            #     intMM_imag = 0
            #     intPM_real = 0
            #     intPM_imag = 0
            #     intMP_real = 0
            #     intMP_imag = 0
            # elif(n-1 == m or (m == 0 and n == M-1)): # Sub-diagonal
            #     intPP_real = 0
            #     intPP_imag = 0
            #     intMM_real = 0
            #     intMM_imag = 0
            #     intPM_real = 0
            #     intPM_imag = 0
            #     intMP_real = 0
            #     intMP_imag = 0
            else:
                intPP_real = integrate.dblquad(integrantPP_real, -1, 1, -1, 1)[0]
                intPP_imag = integrate.dblquad(integrantPP_imag, -1, 1, -1, 1)[0]
                intMM_real = integrate.dblquad(integrantMM_real, -1, 1, -1, 1)[0]
                intMM_imag = integrate.dblquad(integrantMM_imag, -1, 1, -1, 1)[0]
                intPM_real = integrate.dblquad(integrantPM_real, -1, 1, -1, 1)[0]
                intPM_imag = integrate.dblquad(integrantPM_imag, -1, 1, -1, 1)[0]
                intMP_real = integrate.dblquad(integrantMP_real, -1, 1, -1, 1)[0]
                intMP_imag = integrate.dblquad(integrantMP_imag, -1, 1, -1, 1)[0]
                intPP_real_R = integrate.dblquad(integrantPP_real_R, -1, 1, -1, 1)[0]
                intPP_imag_R = integrate.dblquad(integrantPP_imag_R, -1, 1, -1, 1)[0]
                intMM_real_R = integrate.dblquad(integrantMM_real_R, -1, 1, -1, 1)[0]
                intMM_imag_R = integrate.dblquad(integrantMM_imag_R, -1, 1, -1, 1)[0]
                intPM_real_R = integrate.dblquad(integrantPM_real_R, -1, 1, -1, 1)[0]
                intPM_imag_R = integrate.dblquad(integrantPM_imag_R, -1, 1, -1, 1)[0]
                intMP_real_R = integrate.dblquad(integrantMP_real_R, -1, 1, -1, 1)[0]
                intMP_imag_R = integrate.dblquad(integrantMP_imag_R, -1, 1, -1, 1)[0]
            
            Zmn_left[m,n]       += dstm_P*dstn_P*1j*omega*mu0*(tauxm_P*tauxn_P + tauym_P*tauyn_P)*(intPP_real + 1j*intPP_imag)
            Zmn_left[m-1,n-1]   += dstm_M*dstn_M*1j*omega*mu0*(tauxm_M*tauxn_M + tauym_M*tauyn_M)*(intMM_real + 1j*intMM_imag)
            Zmn_left[m,n-1]     += dstm_P*dstn_M*1j*omega*mu0*(tauxm_P*tauxn_M + tauym_P*tauyn_M)*(intPM_real + 1j*intPM_imag)
            Zmn_left[m-1,n]     += dstm_P*dstn_M*1j*omega*mu0*(tauxm_M*tauxn_P + tauym_M*tauyn_P)*(intMP_real + 1j*intMP_imag)
                
            Zmn_right[m,n]      += dstm_P*dstn_P*(tauxm_P**2 + tauym_P**2)/(2*pi*1j*omega*epsilon0)*(intPP_real_R + 1j*intPP_imag_R)
            Zmn_right[m-1,n-1]  += dstm_M*dstn_M*(tauxm_M**2 + tauym_M**2)/(2*pi*1j*omega*epsilon0)*(intMM_real_R + 1j*intMM_imag_R)
            Zmn_right[m,n-1]    += dstm_P*dstn_M*(tauxm_P**2 + tauym_M**2)/(2*pi*1j*omega*epsilon0)*(intPM_real_R + 1j*intPM_imag_R)
            Zmn_right[m-1,n]    += dstm_M*dstn_P*(tauxm_M**2 + tauym_P**2)/(2*pi*1j*omega*epsilon0)*(intMP_real_R + 1j*intMP_imag_R)
            
    Zmn = Zmn_left + Zmn_right
    
    return Zmn, Zmn_left, Zmn_right #, Zmn_left_PP, Zmn_left_MM, Zmn_left_PM, Zmn_left_MP, Zmn_right_PP, Zmn_right_MM, Zmn_right_PM, Zmn_right_MP

# def Zmn_right_integral_calculator(coordinates, wavelength):
#     # Same as the left calculator, but different prefactors
#     M = len(coordinates)-1
#     k0 = 2*pi/wavelength
#     epsilon0 = 8.854187812813e-12
#     c = 299792458
#     omega = 2*pi*c/wavelength
    
#     Zmn_right = np.zeros((M,M), dtype=np.complex128)
#     Ipp = np.zeros((M,M), dtype=np.complex128)
#     Imm = np.zeros((M,M), dtype=np.complex128)
#     Ipm = np.zeros((M,M), dtype=np.complex128)
#     Imp = np.zeros((M,M), dtype=np.complex128)
    
#     for n in np.arange(M):
#         for m in np.arange(M):
#             segm = Coordinates_to_segment(coordinates, m)
#             segn = Coordinates_to_segment(coordinates, n)
            
#             dstm = segment_length(segm)/2
#             dstn = segment_length(segn)/2
            
#             dstrho = lambda eta, ksi, segm, segn: np.linalg.norm(np.subtract(rho(eta, segm), rho(ksi, segn)))
            
#             tauxm, tauym = Tangent_vector_coefficients(segm)
#             tauxn, tauyn = Tangent_vector_coefficients(segn)
            
#             integrantPP_real = lambda eta, ksi: 0.5*0.5*(1/(2*pi))*np.real(kv(0, 1j*k0*dstrho(eta, ksi, segm, segn)))
#             integrantPP_imag = lambda eta, ksi: 0.5*0.5*(1/(2*pi))*np.imag(kv(0, 1j*k0*dstrho(eta, ksi, segm, segn)))
#             integrantMM_real = lambda eta, ksi: -0.5*0.5*(1/(2*pi))*np.real(kv(0, 1j*k0*dstrho(eta, ksi, segm, segn)))
#             integrantMM_imag = lambda eta, ksi: -0.5*0.5*(1/(2*pi))*np.imag(kv(0, 1j*k0*dstrho(eta, ksi, segm, segn)))
#             integrantPM_real = lambda eta, ksi: 0.5*-0.5*(1/(2*pi))*np.real(kv(0, 1j*k0*dstrho(eta, ksi, segm, segn)))
#             integrantPM_imag = lambda eta, ksi: 0.5*-0.5*(1/(2*pi))*np.imag(kv(0, 1j*k0*dstrho(eta, ksi, segm, segn)))
#             integrantMP_real = lambda eta, ksi: -0.5*-0.5*(1/(2*pi))*np.real(kv(0, 1j*k0*dstrho(eta, ksi, segm, segn)))
#             integrantMP_imag = lambda eta, ksi: -0.5*-0.5*(1/(2*pi))*np.imag(kv(0, 1j*k0*dstrho(eta, ksi, segm, segn)))
            
#             if closed(coordinates):
#                 if(m == n or (m == 0 and n == M-1) or (m == M-1 and n == 0)):
#                     Zmn_right[m,n] = 0
#                 else:
#                     intPP_real = integrate.dblquad(integrantPP_real, -1, 1, -1, 1)[0]
#                     intPP_imag = integrate.dblquad(integrantPP_imag, -1, 1, -1, 1)[0]
#                     intMM_real = integrate.dblquad(integrantMM_real, -1, 1, -1, 1)[0]
#                     intMM_imag = integrate.dblquad(integrantMM_imag, -1, 1, -1, 1)[0]
#                     intPM_real = integrate.dblquad(integrantPM_real, -1, 1, -1, 1)[0]
#                     intPM_imag = integrate.dblquad(integrantPM_imag, -1, 1, -1, 1)[0]
#                     intMP_real = integrate.dblquad(integrantMP_real, -1, 1, -1, 1)[0]
#                     intMP_imag = integrate.dblquad(integrantMP_imag, -1, 1, -1, 1)[0]
                    
#                     Ipp[m,n] = intPP_real + 1j*intPP_imag
#                     Imm[m,n] = intMM_real + 1j*intMM_imag
#                     Ipm[m,n] = intPM_real + 1j*intPM_imag
#                     Imp[m,n] = intMP_real + 1j*intMP_imag
#                     Zmn_right[m,n] = dstm*dstn*(tauxm**2 + tauym**2)/(2*pi*1j*omega*epsilon0)*(Ipp[m,n] + Imm[m,n] + Ipm[m,n] + Imp[m,n])
#             else:
#                 if(m == n):
#                     Zmn_right[m,n] = 0
#                 else:
#                     intPP_real = integrate.dblquad(integrantPP_real, -1, 1, -1, 1)[0]
#                     intPP_imag = integrate.dblquad(integrantPP_imag, -1, 1, -1, 1)[0]
#                     intMM_real = integrate.dblquad(integrantMM_real, -1, 1, -1, 1)[0]
#                     intMM_imag = integrate.dblquad(integrantMM_imag, -1, 1, -1, 1)[0]
#                     intPM_real = integrate.dblquad(integrantPM_real, -1, 1, -1, 1)[0]
#                     intPM_imag = integrate.dblquad(integrantPM_imag, -1, 1, -1, 1)[0]
#                     intMP_real = integrate.dblquad(integrantMP_real, -1, 1, -1, 1)[0]
#                     intMP_imag = integrate.dblquad(integrantMP_imag, -1, 1, -1, 1)[0]
                    
#                     Ipp[m,n] = intPP_real + 1j*intPP_imag
#                     Imm[m,n] = intMM_real + 1j*intMM_imag
#                     Ipm[m,n] = intPM_real + 1j*intPM_imag
#                     Imp[m,n] = intMP_real + 1j*intMP_imag
#                     Zmn_right[m,n] = dstm*dstn*(tauxm**2 + tauym**2)/(2*pi*1j*omega*epsilon0)*(Ipp[m,n] + Imm[m,n] + Ipm[m,n] + Imp[m,n])
#         print(n)
#     return Zmn_right

def Zmn_diag_calculator(coordinates, wavelength):
    # Calculate the diagonal of the Z matrix us the self term approximation
    M = len(coordinates)-1
    mu0 = 4 * pi * 10**-7
    epsilon0 = 8.854187812813e-12
    c = 299792458
    omega = 2*pi*c/wavelength
    
    Zdiag = np.zeros((M,1), dtype=np.complex128)
    
    for m in np.arange(M):
        segm = Coordinates_to_segment(coordinates, m)
        
        Is = Self_term_integral(coordinates, wavelength, m)
        
        tauxm, tauym = Tangent_vector_coefficients(segm)
        
        # Only uses the left integral, as the right intgeral reduces to 0 (2Is[0]-2Is[0]=0)
        Zdiag[m] = 1j*omega*mu0/(2*pi)*(tauxm*tauxm + tauym*tauym)*(1/4)*(4*Is[0])
        
    return Zdiag

def Self_term_integral(coordinates, wavelength, m):
    # Define the different elements of the self term integral Is (Is[eta*ksi] and Is[1])
    # using Appendix A3 as a reference for the equations
    k0 = 2*pi/wavelength
    if m == -1:
        segm = Coordinates_to_segment(coordinates, -2);
    else:
        segm = Coordinates_to_segment(coordinates, m)
        
    d = segment_length(segm)/2
    p = d*1j*k0
    
    # Order of calculations (upper bound for the summations), the paper defines
    # an order of 15 to be sufficient for its use case
    order = 15

    # Calculate the self-term integral for Is[eta*ksi]
    Is_etaksi = 0;
    for k in range(order+1):
        term = (2 - k * (2 * (1 + k) * (1 + 2 * k) * (2 + k) * (Harmonic_number(k) - np.log(p)) + k * (7 + 4 * k)))
        term /= (pow((1 + k), 2) * pow((1 + 2 * k), 2) * pow((2 + k), 2) * pow(factorial(k), 2))
        Is_etaksi += term
        # print(f"k={k}, m={m}: term_etaksi={Is_etaksi}")
    Is_etaksi *= 2*pow(d, 2)
    
    # Calculate the self-term integral for Is[1]
    Is1 = 0
    for k in range(order+1):
        term = (3 + 4 * k + 2 * (1 + k) * (1 + 2 * k) * (Harmonic_number(k) - np.log(p)))
        term /= (pow((1 + k), 2) * pow((1 + 2 * k), 2) * pow(factorial(k), 2))
        Is1 += term
    Is1 *= 2*pow(d, 2)
    
    return np.array([Is1, Is_etaksi])

def Harmonic_number(k):
    # Scipy inbuilt function to calculate the harmonic number phi(k)
    return psi(k + 1)

"""CHANGE THIS LATER ON"""
# TODO: NOT YET TESTED, JUST COPIED FOR NOW
def EFIE_TM(coordinates,wavelength,angle):
    # The main algorithm
    Ein_x, Ein_y = DiscritizeEin(coordinates,wavelength,angle)
    Zm = Zmn_calculator(coordinates,wavelength)
    
    # Solve for both the x and y incident field
    Jz_x = np.dot(np.linalg.inv(Zm),Ein_x)
    Jz_y = np.dot(np.linalg.inv(Zm), Ein_y)
    # Return all variables of interest as a tuple
    return Jz_x, Jz_y, Ein_x, Ein_y, Zm

def Etot(Jz_x, Jz_y, R, coordinates, wavelength, angle):
    # Calculate the total field on given coordinates
    M = len(R)
    Etot_x = np.zeros(M, dtype=np.complex128)
    Etot_y = np.zeros(M, dtype=np.complex128)
    for i in np.arange(M):
            r = R[i]
            Esc_x, Esc_y = Escatter(Jz_x, Jz_y, r, coordinates, wavelength)
            Ein_x, Ein_y = Efield_in(r, wavelength, angle)
            Etot_x[i] = Ein_x + Esc_x
            Etot_y[i] = Ein_y + Esc_y
    return Etot_x, Etot_y

def Escatter(Jz_x, Jz_y, rho,coordinates,wavelength):
    # Calculate the Electric field scattered from the object on given coordinates
    mu0 = 4*pi*10**-7
    epsilon0 = 8.854187812813e-12
    c = 299792458
    omega = 2*pi*c/wavelength
    
    # G can be complex so allocate complex matrix
    G_x = np.zeros(len(Jz_x), dtype=np.complex128) 
    G_y = np.zeros(len(Jz_y), dtype=np.complex128)
    # Note length Jz = length coordinates-1, so no -1 nessesary here
    for n in np.arange(len(Jz_x)):
        segment = Coordinates_to_segment(coordinates,n)
        
        GReal_x = lambda eta: (1/4)*(1 + eta)*(1 - eta)*np.real(greenEsc(rho,eta,segment,wavelength)[0])
        GImag_x = lambda eta: (1/4)*(1 + eta)*(1 - eta)*np.imag(greenEsc(rho,eta,segment,wavelength)[0])
        GReal_y = lambda eta: (1/4)*(1 + eta)*(1 - eta)*np.real(greenEsc(rho,eta,segment,wavelength)[1])
        GImag_y = lambda eta: (1/4)*(1 + eta)*(1 - eta)*np.imag(greenEsc(rho,eta,segment,wavelength)[1])
        
        IntReal_x = integrate.quad(GReal_x, -1, 1)[0]
        IntImag_x = integrate.quad(GImag_x, -1, 1)[0]
        IntReal_y = integrate.quad(GReal_y, -1, 1)[0]
        IntImag_y = integrate.quad(GImag_y, -1, 1)[0]
        
        # Correct for the basis function used
        dst = segment_length(segment)
        G_x[n] = dst*(IntReal_x + 1j*IntImag_x)
        G_y[n] = dst*(IntReal_y + 1j*IntImag_y)
        
    # Compute the scattered field for both the x and y components
    Esc_x = 1j*omega*mu0*np.dot(Jz_x, G_x) + 1/(1j*omega*epsilon0)*np.dot(Jz_x, G_x)
    Esc_y = 1j*omega*mu0*np.dot(Jz_y, G_y) + 1/(1j*omega*epsilon0)*np.dot(Jz_y, G_y)
    return Esc_x, Esc_y

def greenEsc(r,ksi,segment,wavelength):
    # Special case, only the basis function applied
    k0 = 2*pi/wavelength
    Pn = rho(ksi, segment)
    
    # Compute Greens function for x and y separately
    G = (1/(2*pi))*kv(0, 1j*k0*np.linalg.norm(np.subtract(r, Pn)))
    
    # Project onto x and y components
    dx = r[0] - Pn[0]
    dy = r[1] - Pn[1]
    r_norm = np.linalg.norm([dx, dy]) + 1e-10
    
    return G*(dx/r_norm), G*(dy/r_norm)

# UNCOMMENT THIS PART TO TEST FUNCTIONS SEPARATELY!
# ## INPUTS
# angle = 1.05*pi/2
# c = 299792458
# f = 150*10**6
# mu0 = 4*pi*10**-7
# wavelength = c/f

# # # Generate a closed circle of radius 1 with 30 data points
# M = 30
# R = 1
# Data = createcircle(M,R)

# # # # # ## MAIN
# Zmn, Zmn_left, Zmn_right = Zmn_calculator_2x2matrix_method(Data, wavelength)
# # # # Z_diag_test = Zmn_diag_calculator(Data, wavelength)
# # # # print(Z_diag_test)    




