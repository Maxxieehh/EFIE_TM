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
        if m == 0:
            seg_prev = Coordinates_to_segment(coordinates, -2)
        else:
            seg_prev = Coordinates_to_segment(coordinates, m-1)
        
        dst = segment_length(seg)
        dst_prev = segment_length(seg_prev)
        
        [tau_x, tau_y] = Tangent_vector_coefficients(seg)
        [tau_x_prev, tau_y_prev] = Tangent_vector_coefficients(seg_prev)
        
        # Define real and imaginary components of X and Y separately
        # Tangent vectors added due to the crossproduct with nu (see documentation)
        EReal_x = lambda eta: dst*RT_min(eta)*tau_x*np.real(Efield_in(rho(eta, seg), wavelength, angle)[0]) + dst_prev*RT_plus(eta)*tau_x_prev*np.real(Efield_in(rho(eta, seg_prev), wavelength, angle)[0])
        EImag_x = lambda eta: dst*RT_min(eta)*tau_x*np.imag(Efield_in(rho(eta, seg), wavelength, angle)[0]) + dst_prev*RT_plus(eta)*tau_x_prev*np.imag(Efield_in(rho(eta, seg_prev), wavelength, angle)[0])
        EReal_y = lambda eta: dst*RT_min(eta)*tau_y*np.real(Efield_in(rho(eta, seg), wavelength, angle)[1]) + dst_prev*RT_plus(eta)*tau_y_prev*np.real(Efield_in(rho(eta, seg_prev), wavelength, angle)[0])
        EImag_y = lambda eta: dst*RT_min(eta)*tau_y*np.imag(Efield_in(rho(eta, seg), wavelength, angle)[1]) + dst_prev*RT_plus(eta)*tau_y_prev*np.imag(Efield_in(rho(eta, seg_prev), wavelength, angle)[0])
        
        # Perform the integration
        IntReal_x = integrate.quad(EReal_x, -1, 1)[0]
        IntImag_x = integrate.quad(EImag_x, -1, 1)[0]
        IntReal_y = integrate.quad(EReal_y, -1, 1)[0]
        IntImag_y = integrate.quad(EImag_y, -1, 1)[0]
        
        Ein_x[m] = (IntReal_x + 1j*IntImag_x)
        Ein_y[m] = (IntReal_y + 1j*IntImag_y)
    
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
    Ex = -E0*sin(angle)*exp(1j*k0*(cos(angle)*x + sin(angle)*y))
    Ey = E0*cos(angle)*exp(1j*k0*(cos(angle)*x + sin(angle)*y))
    
    return Ex, Ey
        
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

def RT_plus(tau):
    # For the rising triangle (/\^+), the shape is given as |\, where tau is
    # either Xi or Eta depending on if we use it for test or basis functions, respecitvely
    return (1/2)*(1 + tau)

def RT_min(tau):
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
    M = len(coordinates)-1
    Zmn_left = Zmn_calculator_left(coordinates, wavelength)
    Zmn_right = Zmn_calculator_right(coordinates, wavelength)
    Zmn = Zmn_left + Zmn_right
    
    # Add the diagonal to the Z matrix
    Zmn_diag = Zmn_diag_calculator_V2(coordinates, wavelength)
    np.fill_diagonal(Zmn, Zmn_diag.flatten())
    
    # Add the super and sub diagonal to the matrix
    Zmn_adj = Zmn_adj_calculator(coordinates, wavelength)
    np.fill_diagonal(Zmn[1:], Zmn_adj.flatten()) # sub
    np.fill_diagonal(Zmn[:,1:], Zmn_adj.flatten()) # super
    Zmn[0,M-1] = Zmn[0,1]
    Zmn[M-1,0] = Zmn[0,1]
    
    # Uncomment this if you want to use the matrix provided by Mathematica
    # Zmn_mathematica = np.asarray([
    #       [13.5812 +17.2948j, 11.8056 +3.93983j, 7.40115 -3.57578j, 2.50505 -5.40834j, -0.9013-4.15369j, -2.04578-1.89573j, -1.45371-0.263049j, -0.265374+0.0946988j, 0.552411 -0.583402j, 0.599297 -1.66701j, -0.0167671-2.60958j, -0.952688-3.16288j, -1.87388-3.3495j, -2.56453-3.32857j, -2.92528-3.2682j, -2.92528-3.2682j, -2.56453-3.32857j, -1.87388-3.3495j, -0.952687-3.16288j, -0.0167671-2.60958j, 0.599296 -1.66701j, 0.552411 -0.583402j, -0.265374+0.0946989j, -1.45371-0.263049j, -2.04578-1.89573j, -0.9013-4.15369j, 2.50505 -5.40834j, 7.40115 -3.57578j, 11.8056 +3.93983j],
    #       [11.8056 +3.93983j, 13.5812 +17.2948j, 11.8056 +3.93983j, 7.40115 -3.57578j, 2.50505 -5.40834j, -0.9013-4.15369j, -2.04578-1.89573j, -1.45371-0.263049j, -0.265374+0.0946989j, 0.552411 -0.583402j, 0.599296 -1.66701j, -0.0167671-2.60958j, -0.952688-3.16288j, -1.87388-3.3495j, -2.56453-3.32857j, -2.92528-3.2682j, -2.92528-3.2682j, -2.56453-3.32857j, -1.87388-3.3495j, -0.952688-3.16288j, -0.0167671-2.60958j, 0.599297 -1.66701j, 0.552411 -0.583402j, -0.265374+0.0946988j, -1.45371-0.263049j, -2.04578-1.89573j, -0.9013-4.15369j, 2.50505 -5.40834j, 7.40115 -3.57578j],
    #       [7.40115 -3.57578j, 11.8056 +3.93983j, 13.5812 +17.2948j, 11.8056 +3.93983j, 7.40115 -3.57578j, 2.50505 -5.40834j, -0.9013-4.15369j, -2.04578-1.89573j, -1.45371-0.263049j, -0.265374+0.0946988j, 0.552411 -0.583402j, 0.599297 -1.66701j, -0.0167671-2.60958j, -0.952688-3.16288j, -1.87388-3.3495j, -2.56453-3.32857j, -2.92528-3.2682j, -2.92528-3.2682j, -2.56453-3.32857j, -1.87388-3.3495j, -0.952688-3.16288j, -0.0167671-2.60958j, 0.599296 -1.66701j, 0.552411 -0.583402j, -0.265374+0.0946989j, -1.45371-0.263049j, -2.04578-1.89573j, -0.9013-4.15369j, 2.50505 -5.40834j],
    #       [2.50505 -5.40834j, 7.40115 -3.57578j, 11.8056 +3.93983j, 13.5812 +17.2948j, 11.8056 +3.93983j, 7.40115 -3.57578j, 2.50505 -5.40834j, -0.9013-4.15369j, -2.04578-1.89574j, -1.45371-0.263049j, -0.265374+0.0946989j, 0.552411 -0.583402j, 0.599297 -1.66701j, -0.0167671-2.60958j, -0.952688-3.16288j, -1.87388-3.3495j, -2.56453-3.32857j, -2.92528-3.2682j, -2.92528-3.2682j, -2.56453-3.32857j, -1.87388-3.3495j, -0.952688-3.16288j, -0.0167671-2.60958j, 0.599297 -1.66701j, 0.552411 -0.583402j, -0.265374+0.0946988j, -1.45371-0.263049j, -2.04578-1.89573j, -0.9013-4.15369j],
    #       [-0.9013-4.15369j, 2.50505 -5.40834j, 7.40115 -3.57578j, 11.8056 +3.93983j, 13.5812 +17.2948j, 11.8056 +3.93983j, 7.40115 -3.57578j, 2.50505 -5.40834j, -0.9013-4.15369j, -2.04578-1.89573j, -1.45371-0.263049j, -0.265374+0.0946989j, 0.552411 -0.583402j, 0.599296 -1.66701j, -0.0167671-2.60958j, -0.952687-3.16288j, -1.87388-3.3495j, -2.56453-3.32857j, -2.92528-3.2682j, -2.92528-3.2682j, -2.56453-3.32857j, -1.87388-3.3495j, -0.952687-3.16288j, -0.0167671-2.60958j, 0.599297 -1.66701j, 0.552411 -0.583402j, -0.265374+0.0946989j, -1.45371-0.263049j, -2.04578-1.89573j],
    #       [-2.04578-1.89573j, -0.9013-4.15369j, 2.50505 -5.40834j, 7.40115 -3.57578j, 11.8056 +3.93983j, 13.5812 +17.2948j, 11.8056 +3.93983j, 7.40115 -3.57578j, 2.50505 -5.40834j, -0.9013-4.15369j, -2.04578-1.89573j, -1.45371-0.263049j, -0.265374+0.0946988j, 0.552411 -0.583402j, 0.599297 -1.66701j, -0.0167671-2.60958j, -0.952688-3.16288j, -1.87388-3.3495j, -2.56453-3.32857j, -2.92528-3.2682j, -2.92528-3.2682j, -2.56453-3.32857j, -1.87388-3.3495j, -0.952687-3.16288j, -0.0167671-2.60958j, 0.599297 -1.66701j, 0.552411 -0.583402j, -0.265374+0.0946988j, -1.45371-0.263049j],
    #       [-1.45371-0.263049j, -2.04578-1.89573j, -0.9013-4.15369j, 2.50505 -5.40834j, 7.40115 -3.57578j, 11.8056 +3.93983j, 13.5812 +17.2948j, 11.8056 +3.93983j, 7.40115 -3.57578j, 2.50505 -5.40834j, -0.9013-4.15369j, -2.04578-1.89573j, -1.45371-0.263049j, -0.265374+0.0946989j, 0.552411 -0.583402j, 0.599297 -1.66701j, -0.0167671-2.60958j, -0.952688-3.16288j, -1.87388-3.3495j, -2.56453-3.32857j, -2.92528-3.2682j, -2.92528-3.2682j, -2.56453-3.32857j, -1.87388-3.3495j, -0.952687-3.16288j, -0.0167671-2.60958j, 0.599296 -1.66701j, 0.552411 -0.583402j, -0.265374+0.0946989j],
    #       [-0.265374+0.0946988j, -1.45371-0.263049j, -2.04578-1.89573j, -0.9013-4.15369j, 2.50505 -5.40834j, 7.40115 -3.57578j, 11.8056 +3.93983j, 13.5812 +17.2948j, 11.8056 +3.93983j, 7.40115 -3.57578j, 2.50505 -5.40834j, -0.9013-4.15369j, -2.04578-1.89573j, -1.45371-0.263049j, -0.265374+0.0946989j, 0.552411 -0.583402j, 0.599296 -1.66701j, -0.0167671-2.60958j, -0.952687-3.16288j, -1.87388-3.3495j, -2.56453-3.32857j, -2.92528-3.2682j, -2.92528-3.2682j, -2.56453-3.32857j, -1.87388-3.3495j, -0.952688-3.16288j, -0.0167671-2.60958j, 0.599297 -1.66701j, 0.552411 -0.583402j],
    #       [0.552411 -0.583402j, -0.265374+0.0946989j, -1.45371-0.263049j, -2.04578-1.89574j, -0.9013-4.15369j, 2.50505 -5.40834j, 7.40115 -3.57578j, 11.8056 +3.93983j, 13.5812 +17.2948j, 11.8056 +3.93983j, 7.40115 -3.57578j, 2.50505 -5.40834j, -0.9013-4.15369j, -2.04578-1.89573j, -1.45371-0.263049j, -0.265374+0.0946988j, 0.552411 -0.583402j, 0.599297 -1.66701j, -0.0167671-2.60958j, -0.952688-3.16288j, -1.87388-3.3495j, -2.56453-3.32857j, -2.92528-3.2682j, -2.92528-3.2682j, -2.56453-3.32857j, -1.87388-3.3495j, -0.952688-3.16288j, -0.0167671-2.60958j, 0.599296 -1.66701j],
    #       [0.599297 -1.66701j, 0.552411 -0.583402j, -0.265374+0.0946988j, -1.45371-0.263049j, -2.04578-1.89573j, -0.9013-4.15369j, 2.50505 -5.40834j, 7.40115 -3.57578j, 11.8056 +3.93983j, 13.5812 +17.2948j, 11.8056 +3.93983j, 7.40115 -3.57578j, 2.50505 -5.40834j, -0.9013-4.15369j, -2.04578-1.89573j, -1.45371-0.263049j, -0.265374+0.094699j, 0.552411 -0.583402j, 0.599296 -1.66701j, -0.016767-2.60958j, -0.952688-3.16288j, -1.87388-3.3495j, -2.56453-3.32857j, -2.92528-3.2682j, -2.92528-3.2682j, -2.56453-3.32857j, -1.87388-3.3495j, -0.952688-3.16288j, -0.0167671-2.60958j],
    #       [-0.0167671-2.60958j, 0.599296 -1.66701j, 0.552411 -0.583402j, -0.265374+0.0946989j, -1.45371-0.263049j, -2.04578-1.89573j, -0.9013-4.15369j, 2.50505 -5.40834j, 7.40115 -3.57578j, 11.8056 +3.93983j, 13.5812 +17.2948j, 11.8056 +3.93983j, 7.40115 -3.57578j, 2.50505 -5.40834j, -0.9013-4.15369j, -2.04578-1.89573j, -1.45371-0.263049j, -0.265374+0.0946988j, 0.552411 -0.583402j, 0.599296 -1.66701j, -0.0167671-2.60958j, -0.952687-3.16288j, -1.87388-3.3495j, -2.56453-3.32857j, -2.92528-3.2682j, -2.92528-3.2682j, -2.56453-3.32857j, -1.87388-3.3495j, -0.952687-3.16288j],
    #       [-0.952688-3.16288j, -0.0167671-2.60958j, 0.599297 -1.66701j, 0.552411 -0.583402j, -0.265374+0.0946989j, -1.45371-0.263049j, -2.04578-1.89573j, -0.9013-4.15369j, 2.50505 -5.40834j, 7.40115 -3.57578j, 11.8056 +3.93983j, 13.5812 +17.2948j, 11.8056 +3.93983j, 7.40115 -3.57578j, 2.50505 -5.40834j, -0.9013-4.15369j, -2.04578-1.89573j, -1.45371-0.263049j, -0.265374+0.0946988j, 0.552411 -0.583402j, 0.599297 -1.66701j, -0.0167671-2.60958j, -0.952688-3.16288j, -1.87388-3.3495j, -2.56453-3.32857j, -2.92528-3.2682j, -2.92528-3.2682j, -2.56453-3.32857j, -1.87388-3.3495j],
    #       [-1.87388-3.3495j, -0.952688-3.16288j, -0.0167671-2.60958j, 0.599297 -1.66701j, 0.552411 -0.583402j, -0.265374+0.0946988j, -1.45371-0.263049j, -2.04578-1.89573j, -0.9013-4.15369j, 2.50505 -5.40834j, 7.40115 -3.57578j, 11.8056 +3.93983j, 13.5812 +17.2948j, 11.8056 +3.93983j, 7.40115 -3.57578j, 2.50505 -5.40834j, -0.9013-4.15369j, -2.04578-1.89573j, -1.45371-0.263049j, -0.265374+0.094699j, 0.552411 -0.583402j, 0.599296 -1.66701j, -0.0167671-2.60958j, -0.952688-3.16288j, -1.87388-3.3495j, -2.56453-3.32857j, -2.92528-3.2682j, -2.92528-3.2682j, -2.56453-3.32857j],
    #       [-2.56453-3.32857j, -1.87388-3.3495j, -0.952688-3.16288j, -0.0167671-2.60958j, 0.599296 -1.66701j, 0.552411 -0.583402j, -0.265374+0.0946989j, -1.45371-0.263049j, -2.04578-1.89573j, -0.9013-4.15369j, 2.50505 -5.40834j, 7.40115 -3.57578j, 11.8056 +3.93983j, 13.5812 +17.2948j, 11.8056 +3.93983j, 7.40115 -3.57578j, 2.50505 -5.40834j, -0.9013-4.15369j, -2.04578-1.89573j, -1.45371-0.263049j, -0.265374+0.0946988j, 0.552411 -0.583402j, 0.599297 -1.66701j, -0.0167671-2.60958j, -0.952687-3.16288j, -1.87388-3.3495j, -2.56453-3.32857j, -2.92528-3.2682j, -2.92528-3.2682j],
    #       [-2.92528-3.2682j, -2.56453-3.32857j, -1.87388-3.3495j, -0.952688-3.16288j, -0.0167671-2.60958j, 0.599297 -1.66701j, 0.552411 -0.583402j, -0.265374+0.0946989j, -1.45371-0.263049j, -2.04578-1.89573j, -0.9013-4.15369j, 2.50505 -5.40834j, 7.40115 -3.57578j, 11.8056 +3.93983j, 13.5812 +17.2948j, 11.8056 +3.93983j, 7.40115 -3.57578j, 2.50505 -5.40834j, -0.9013-4.15369j, -2.04578-1.89573j, -1.45371-0.263049j, -0.265374+0.0946989j, 0.552411 -0.583402j, 0.599297 -1.66701j, -0.0167671-2.60958j, -0.952688-3.16288j, -1.87388-3.3495j, -2.56453-3.32857j, -2.92528-3.2682j],
    #       [-2.92528-3.2682j, -2.92528-3.2682j, -2.56453-3.32857j, -1.87388-3.3495j, -0.952687-3.16288j, -0.0167671-2.60958j, 0.599297 -1.66701j, 0.552411 -0.583402j, -0.265374+0.0946988j, -1.45371-0.263049j, -2.04578-1.89573j, -0.9013-4.15369j, 2.50505 -5.40834j, 7.40115 -3.57578j, 11.8056 +3.93983j, 13.5812 +17.2948j, 11.8056 +3.93983j, 7.40115 -3.57578j, 2.50505 -5.40834j, -0.9013-4.15369j, -2.04578-1.89573j, -1.45371-0.263049j, -0.265374+0.0946989j, 0.552411 -0.583402j, 0.599296 -1.66701j, -0.0167671-2.60958j, -0.952688-3.16288j, -1.87388-3.3495j, -2.56453-3.32857j],
    #       [-2.56453-3.32857j, -2.92528-3.2682j, -2.92528-3.2682j, -2.56453-3.32857j, -1.87388-3.3495j, -0.952688-3.16288j, -0.0167671-2.60958j, 0.599296 -1.66701j, 0.552411 -0.583402j, -0.265374+0.094699j, -1.45371-0.263049j, -2.04578-1.89573j, -0.9013-4.15369j, 2.50505 -5.40834j, 7.40115 -3.57578j, 11.8056 +3.93983j, 13.5812 +17.2948j, 11.8056 +3.93983j, 7.40115 -3.57578j, 2.50505 -5.40834j, -0.9013-4.15369j, -2.04578-1.89573j, -1.45371-0.263049j, -0.265374+0.0946988j, 0.552411 -0.583402j, 0.599297 -1.66701j, -0.0167671-2.60958j, -0.952688-3.16288j, -1.87388-3.3495j],
    #       [-1.87388-3.3495j, -2.56453-3.32857j, -2.92528-3.2682j, -2.92528-3.2682j, -2.56453-3.32857j, -1.87388-3.3495j, -0.952688-3.16288j, -0.0167671-2.60958j, 0.599297 -1.66701j, 0.552411 -0.583402j, -0.265374+0.0946988j, -1.45371-0.263049j, -2.04578-1.89573j, -0.9013-4.15369j, 2.50505 -5.40834j, 7.40115 -3.57578j, 11.8056 +3.93983j, 13.5812 +17.2948j, 11.8056 +3.93983j, 7.40115 -3.57578j, 2.50505 -5.40834j, -0.9013-4.15369j, -2.04578-1.89573j, -1.45371-0.263049j, -0.265374+0.0946989j, 0.552411 -0.583402j, 0.599297 -1.66701j, -0.0167671-2.60958j, -0.952688-3.16288j],
    #       [-0.952687-3.16288j, -1.87388-3.3495j, -2.56453-3.32857j, -2.92528-3.2682j, -2.92528-3.2682j, -2.56453-3.32857j, -1.87388-3.3495j, -0.952687-3.16288j, -0.0167671-2.60958j, 0.599296 -1.66701j, 0.552411 -0.583402j, -0.265374+0.0946988j, -1.45371-0.263049j, -2.04578-1.89573j, -0.9013-4.15369j, 2.50505 -5.40834j, 7.40115 -3.57578j, 11.8056 +3.93983j, 13.5812 +17.2948j, 11.8056 +3.93983j, 7.40115 -3.57578j, 2.50505 -5.40834j, -0.9013-4.15369j, -2.04578-1.89573j, -1.45371-0.263049j, -0.265374+0.0946989j, 0.552411 -0.583402j, 0.599296 -1.66701j, -0.0167671-2.60958j],
    #       [-0.0167671-2.60958j, -0.952688-3.16288j, -1.87388-3.3495j, -2.56453-3.32857j, -2.92528-3.2682j, -2.92528-3.2682j, -2.56453-3.32857j, -1.87388-3.3495j, -0.952688-3.16288j, -0.016767-2.60958j, 0.599296 -1.66701j, 0.552411 -0.583402j, -0.265374+0.094699j, -1.45371-0.263049j, -2.04578-1.89573j, -0.9013-4.15369j, 2.50505 -5.40834j, 7.40115 -3.57578j, 11.8056 +3.93983j, 13.5812 +17.2948j, 11.8056 +3.93983j, 7.40115 -3.57578j, 2.50505 -5.40834j, -0.9013-4.15369j, -2.04578-1.89573j, -1.45371-0.263049j, -0.265374+0.0946988j, 0.552411 -0.583402j, 0.599297 -1.66701j],
    #       [0.599296 -1.66701j, -0.0167671-2.60958j, -0.952688-3.16288j, -1.87388-3.3495j, -2.56453-3.32857j, -2.92528-3.2682j, -2.92528-3.2682j, -2.56453-3.32857j, -1.87388-3.3495j, -0.952688-3.16288j, -0.0167671-2.60958j, 0.599297 -1.66701j, 0.552411 -0.583402j, -0.265374+0.0946988j, -1.45371-0.263049j, -2.04578-1.89573j, -0.9013-4.15369j, 2.50505 -5.40834j, 7.40115 -3.57578j, 11.8056 +3.93983j, 13.5812 +17.2948j, 11.8056 +3.93983j, 7.40115 -3.57578j, 2.50505 -5.40834j, -0.9013-4.15369j, -2.04578-1.89574j, -1.45371-0.263049j, -0.265374+0.0946989j, 0.552411 -0.583402j],
    #       [0.552411 -0.583402j, 0.599297 -1.66701j, -0.0167671-2.60958j, -0.952688-3.16288j, -1.87388-3.3495j, -2.56453-3.32857j, -2.92528-3.2682j, -2.92528-3.2682j, -2.56453-3.32857j, -1.87388-3.3495j, -0.952687-3.16288j, -0.0167671-2.60958j, 0.599296 -1.66701j, 0.552411 -0.583402j, -0.265374+0.0946989j, -1.45371-0.263049j, -2.04578-1.89573j, -0.9013-4.15369j, 2.50505 -5.40834j, 7.40115 -3.57578j, 11.8056 +3.93983j, 13.5812 +17.2948j, 11.8056 +3.93983j, 7.40115 -3.57578j, 2.50505 -5.40834j, -0.9013-4.15369j, -2.04578-1.89573j, -1.45371-0.263049j, -0.265374+0.0946988j],
    #       [-0.265374+0.0946989j, 0.552411 -0.583402j, 0.599296 -1.66701j, -0.0167671-2.60958j, -0.952687-3.16288j, -1.87388-3.3495j, -2.56453-3.32857j, -2.92528-3.2682j, -2.92528-3.2682j, -2.56453-3.32857j, -1.87388-3.3495j, -0.952688-3.16288j, -0.0167671-2.60958j, 0.599297 -1.66701j, 0.552411 -0.583402j, -0.265374+0.0946989j, -1.45371-0.263049j, -2.04578-1.89573j, -0.9013-4.15369j, 2.50505 -5.40834j, 7.40115 -3.57578j, 11.8056 +3.93983j, 13.5812 +17.2948j, 11.8056 +3.93983j, 7.40115 -3.57578j, 2.50505 -5.40834j, -0.9013-4.15369j, -2.04578-1.89573j, -1.45371-0.263049j],
    #       [-1.45371-0.263049j, -0.265374+0.0946988j, 0.552411 -0.583402j, 0.599297 -1.66701j, -0.0167671-2.60958j, -0.952687-3.16288j, -1.87388-3.3495j, -2.56453-3.32857j, -2.92528-3.2682j, -2.92528-3.2682j, -2.56453-3.32857j, -1.87388-3.3495j, -0.952688-3.16288j, -0.0167671-2.60958j, 0.599297 -1.66701j, 0.552411 -0.583402j, -0.265374+0.0946988j, -1.45371-0.263049j, -2.04578-1.89573j, -0.9013-4.15369j, 2.50505 -5.40834j, 7.40115 -3.57578j, 11.8056 +3.93983j, 13.5812 +17.2948j, 11.8056 +3.93983j, 7.40115 -3.57578j, 2.50505 -5.40834j, -0.9013-4.15369j, -2.04578-1.89573j],
    #       [-2.04578-1.89573j, -1.45371-0.263049j, -0.265374+0.0946989j, 0.552411 -0.583402j, 0.599297 -1.66701j, -0.0167671-2.60958j, -0.952687-3.16288j, -1.87388-3.3495j, -2.56453-3.32857j, -2.92528-3.2682j, -2.92528-3.2682j, -2.56453-3.32857j, -1.87388-3.3495j, -0.952687-3.16288j, -0.0167671-2.60958j, 0.599296 -1.66701j, 0.552411 -0.583402j, -0.265374+0.0946989j, -1.45371-0.263049j, -2.04578-1.89573j, -0.9013-4.15369j, 2.50505 -5.40834j, 7.40115 -3.57578j, 11.8056 +3.93983j, 13.5812 +17.2948j, 11.8056 +3.93983j, 7.40115 -3.57578j, 2.50505 -5.40834j, -0.9013-4.15369j],
    #       [-0.9013-4.15369j, -2.04578-1.89573j, -1.45371-0.263049j, -0.265374+0.0946988j, 0.552411 -0.583402j, 0.599297 -1.66701j, -0.0167671-2.60958j, -0.952688-3.16288j, -1.87388-3.3495j, -2.56453-3.32857j, -2.92528-3.2682j, -2.92528-3.2682j, -2.56453-3.32857j, -1.87388-3.3495j, -0.952688-3.16288j, -0.0167671-2.60958j, 0.599297 -1.66701j, 0.552411 -0.583402j, -0.265374+0.0946989j, -1.45371-0.263049j, -2.04578-1.89574j, -0.9013-4.15369j, 2.50505 -5.40834j, 7.40115 -3.57578j, 11.8056 +3.93983j, 13.5812 +17.2948j, 11.8056 +3.93983j, 7.40115 -3.57578j, 2.50505 -5.40834j],
    #       [2.50505 -5.40834j, -0.9013-4.15369j, -2.04578-1.89573j, -1.45371-0.263049j, -0.265374+0.0946989j, 0.552411 -0.583402j, 0.599296 -1.66701j, -0.0167671-2.60958j, -0.952688-3.16288j, -1.87388-3.3495j, -2.56453-3.32857j, -2.92528-3.2682j, -2.92528-3.2682j, -2.56453-3.32857j, -1.87388-3.3495j, -0.952688-3.16288j, -0.0167671-2.60958j, 0.599297 -1.66701j, 0.552411 -0.583402j, -0.265374+0.0946988j, -1.45371-0.263049j, -2.04578-1.89573j, -0.9013-4.15369j, 2.50505 -5.40834j, 7.40115 -3.57578j, 11.8056 +3.93983j, 13.5812 +17.2948j, 11.8056 +3.93983j, 7.40115 -3.57578j],
    #       [7.40115 -3.57578j, 2.50505 -5.40834j, -0.9013-4.15369j, -2.04578-1.89573j, -1.45371-0.263049j, -0.265374+0.0946988j, 0.552411 -0.583402j, 0.599297 -1.66701j, -0.0167671-2.60958j, -0.952688-3.16288j, -1.87388-3.3495j, -2.56453-3.32857j, -2.92528-3.2682j, -2.92528-3.2682j, -2.56453-3.32857j, -1.87388-3.3495j, -0.952688-3.16288j, -0.0167671-2.60958j, 0.599296 -1.66701j, 0.552411 -0.583402j, -0.265374+0.0946989j, -1.45371-0.263049j, -2.04578-1.89573j, -0.9013-4.15369j, 2.50505 -5.40834j, 7.40115 -3.57578j, 11.8056 +3.93983j, 13.5812 +17.2948j, 11.8056 +3.93983j],
    #       [11.8056 +3.93983j, 7.40115 -3.57578j, 2.50505 -5.40834j, -0.9013-4.15369j, -2.04578-1.89573j, -1.45371-0.263049j, -0.265374+0.0946989j, 0.552411 -0.583402j, 0.599296 -1.66701j, -0.0167671-2.60958j, -0.952687-3.16288j, -1.87388-3.3495j, -2.56453-3.32857j, -2.92528-3.2682j, -2.92528-3.2682j, -2.56453-3.32857j, -1.87388-3.3495j, -0.952688-3.16288j, -0.0167671-2.60958j, 0.599297 -1.66701j, 0.552411 -0.583402j, -0.265374+0.0946988j, -1.45371-0.263049j, -2.04578-1.89573j, -0.9013-4.15369j, 2.50505 -5.40834j, 7.40115 -3.57578j, 11.8056 +3.93983j, 13.5812 +17.2948j]
    #     ])

    return Zmn

def Zmn_calculator_left(coordinates, wavelength):
    M = len(coordinates)-1
    k0 = 2*pi/wavelength
    mu0 = 4*pi*10**(-7)
    c = 299792458
    omega = 2*pi*c/wavelength
    
    Zmn_left = np.zeros((M,M), dtype=np.complex128)
    
    for m in range(M):
        print(m) # Display what row we are currently on
        for n in range(M):
            # Focus on pairs of edges rather than pairs of triangles
            segment_m = Coordinates_to_segment(coordinates, m)
            if m == 0:
                segment_m_M = Coordinates_to_segment(coordinates, -2)
            else:
                segment_m_M = Coordinates_to_segment(coordinates, m-1)
            segment_n = Coordinates_to_segment(coordinates, n)
            if n == 0:
                segment_n_M = Coordinates_to_segment(coordinates, -2)
            else:
                segment_n_M = Coordinates_to_segment(coordinates, n-1)
            
            dst_m = segment_length(segment_m)/2
            dst_m_M = segment_length(segment_m_M)/2
            dst_n = segment_length(segment_n)/2
            dst_n_M = segment_length(segment_n_M)/2
            
            tau_m = Tangent_vector_coefficients(segment_m)
            tau_m_M = Tangent_vector_coefficients(segment_m_M)
            tau_n = Tangent_vector_coefficients(segment_n)
            tau_n_M = Tangent_vector_coefficients(segment_n_M)
            
            D = lambda eta, ksi, segm, segn: np.linalg.norm(np.subtract(rho(eta, segm), rho(ksi, segn)))
            green = lambda eta, ksi, segm, segn: 1/(2*pi)*kv(0, 1j*k0*D(eta, ksi, segm, segn))
            
            integrant_PP_real = lambda eta, ksi: RT_plus(eta)*RT_plus(ksi)*np.real(green(eta, ksi, segment_m_M, segment_n_M))
            integrant_PP_imag = lambda eta, ksi: RT_plus(eta)*RT_plus(ksi)*np.imag(green(eta, ksi, segment_m_M, segment_n_M))
            integrant_MM_real = lambda eta, ksi: RT_min(eta)*RT_min(ksi)*np.real(green(eta, ksi, segment_m, segment_n))
            integrant_MM_imag = lambda eta, ksi: RT_min(eta)*RT_min(ksi)*np.imag(green(eta, ksi, segment_m, segment_n))
            integrant_PM_real = lambda eta, ksi: RT_plus(eta)*RT_min(ksi)*np.real(green(eta, ksi, segment_m_M, segment_n))
            integrant_PM_imag = lambda eta, ksi: RT_plus(eta)*RT_min(ksi)*np.imag(green(eta, ksi, segment_m_M, segment_n))
            integrant_MP_real = lambda eta, ksi: RT_min(eta)*RT_plus(ksi)*np.real(green(eta, ksi, segment_m, segment_n_M))
            integrant_MP_imag = lambda eta, ksi: RT_min(eta)*RT_plus(ksi)*np.imag(green(eta, ksi, segment_m, segment_n_M))
            
            if(m == n or m-1 == n or (n == 0 and m == M-1) or n-1 == m or (m == 0 and n == M-1)):
                I_PP_real, I_PP_imag = 0, 0
                I_MM_real, I_MM_imag = 0, 0
                I_PM_real, I_PM_imag = 0, 0
                I_MP_real, I_MP_imag = 0, 0
            else:   
                I_PP_real = integrate.dblquad(integrant_PP_real, -1, 1, -1, 1)[0]
                I_PP_imag = integrate.dblquad(integrant_PP_imag, -1, 1, -1, 1)[0]
                I_MM_real = integrate.dblquad(integrant_MM_real, -1, 1, -1, 1)[0]
                I_MM_imag = integrate.dblquad(integrant_MM_imag, -1, 1, -1, 1)[0]
                I_PM_real = integrate.dblquad(integrant_PM_real, -1, 1, -1, 1)[0]
                I_PM_imag = integrate.dblquad(integrant_PM_imag, -1, 1, -1, 1)[0]
                I_MP_real = integrate.dblquad(integrant_MP_real, -1, 1, -1, 1)[0]
                I_MP_imag = integrate.dblquad(integrant_MP_imag, -1, 1, -1, 1)[0]
                    
            I_PP = dst_m_M*dst_n_M*1j*omega*mu0*(tau_m_M[0]*tau_n_M[0] + tau_m_M[1]*tau_n_M[1])*(I_PP_real + 1j*I_PP_imag)
            I_MM = dst_m*dst_n*1j*omega*mu0*(tau_m[0]*tau_n[0] + tau_m[1]*tau_n[1])*(I_MM_real + 1j*I_MM_imag)
            I_PM = dst_m_M*dst_n*1j*omega*mu0*(tau_m_M[0]*tau_n[0] + tau_m_M[1]*tau_n[1])*(I_PM_real + 1j*I_PM_imag)
            I_MP = dst_m*dst_n_M*1j*omega*mu0*(tau_m[0]*tau_n_M[0] + tau_m[1]*tau_n_M[1])*(I_MP_real + 1j*I_MP_imag)
            
            Zmn_left[m,n] = I_PP + I_MM + I_PM + I_MP
            
    return Zmn_left
                
def Zmn_calculator_right(coordinates, wavelength):
    M = len(coordinates)-1
    k0 = 2*pi/wavelength
    c = 299792458
    epsilon0 = 8.854187812813e-12
    omega = 2*pi*c/wavelength
    
    Zmn_right = np.zeros((M,M), dtype=np.complex128)
    
    for m in range(M):
        print(m) # Display what row we are currently on
        for n in range(M):
            # Focus on pairs of edges rather than pairs of triangles
            segment_m = Coordinates_to_segment(coordinates, m)
            if m == 0:
                segment_m_M = Coordinates_to_segment(coordinates, -2)
            else:
                segment_m_M = Coordinates_to_segment(coordinates, m-1)
            segment_n = Coordinates_to_segment(coordinates, n)
            if n == 0:
                segment_n_M = Coordinates_to_segment(coordinates, -2)
            else:
                segment_n_M = Coordinates_to_segment(coordinates, n-1)
            
            dst_m = segment_length(segment_m)/2
            dst_m_M = segment_length(segment_m_M)/2
            dst_n = segment_length(segment_n)/2
            dst_n_M = segment_length(segment_n_M)/2
            
            tau_m = Tangent_vector_coefficients(segment_m)
            tau_m_M = Tangent_vector_coefficients(segment_m_M)
            #tau_n = Tangent_vector_coefficients(segment_n)
            #tau_n_M = Tangent_vector_coefficients(segment_n_M)
            
            D = lambda eta, ksi, segm, segn: np.linalg.norm(np.subtract(rho(eta, segm), rho(ksi, segn)))
            green = lambda eta, ksi, segm, segn: 1/(2*pi)*kv(0, 1j*k0*D(eta, ksi, segm, segn))
            
            integrant_PP_real = lambda eta, ksi: 0.5*0.5*np.real(green(eta, ksi, segment_m_M, segment_n_M))
            integrant_PP_imag = lambda eta, ksi: 0.5*0.5*np.imag(green(eta, ksi, segment_m_M, segment_n_M))
            integrant_MM_real = lambda eta, ksi: -0.5*-0.5*np.real(green(eta, ksi, segment_m, segment_n))
            integrant_MM_imag = lambda eta, ksi: -0.5*-0.5*np.imag(green(eta, ksi, segment_m, segment_n))
            integrant_PM_real = lambda eta, ksi: 0.5*-0.5*np.real(green(eta, ksi, segment_m_M, segment_n))
            integrant_PM_imag = lambda eta, ksi: 0.5*-0.5*np.imag(green(eta, ksi, segment_m_M, segment_n))
            integrant_MP_real = lambda eta, ksi: -0.5*0.5*np.real(green(eta, ksi, segment_m, segment_n_M))
            integrant_MP_imag = lambda eta, ksi: -0.5*0.5*np.imag(green(eta, ksi, segment_m, segment_n_M))
            
            if(m == n or m-1 == n or (n == 0 and m == M-1) or n-1 == m or (m == 0 and n == M-1)):
                I_PP_real, I_PP_imag = 0, 0
                I_MM_real, I_MM_imag = 0, 0
                I_PM_real, I_PM_imag = 0, 0
                I_MP_real, I_MP_imag = 0, 0
            else:   
                I_PP_real = integrate.dblquad(integrant_PP_real, -1, 1, -1, 1)[0]
                I_PP_imag = integrate.dblquad(integrant_PP_imag, -1, 1, -1, 1)[0]
                I_MM_real = integrate.dblquad(integrant_MM_real, -1, 1, -1, 1)[0]
                I_MM_imag = integrate.dblquad(integrant_MM_imag, -1, 1, -1, 1)[0]
                I_PM_real = integrate.dblquad(integrant_PM_real, -1, 1, -1, 1)[0]
                I_PM_imag = integrate.dblquad(integrant_PM_imag, -1, 1, -1, 1)[0]
                I_MP_real = integrate.dblquad(integrant_MP_real, -1, 1, -1, 1)[0]
                I_MP_imag = integrate.dblquad(integrant_MP_imag, -1, 1, -1, 1)[0]
            
            # tau_m[0]**2 + tau_m[1]**2 is always the same (due to |tau_m|=1), so only
            # dependent on the distance between nodes
            I_PP = dst_m_M*dst_n_M/(1j*omega*epsilon0)*(I_PP_real + 1j*I_PP_imag)
            I_MM = dst_m*dst_n/(1j*omega*epsilon0)*(I_MM_real + 1j*I_MM_imag)
            I_PM = dst_m_M*dst_n/(1j*omega*epsilon0)*(I_PM_real + 1j*I_PM_imag)
            I_MP = dst_m*dst_n_M/(1j*omega*epsilon0)*(I_MP_real + 1j*I_MP_imag)
            
            Zmn_right[m,n] = I_PP + I_MM + I_PM + I_MP
            
    return Zmn_right

# def Zmn_calculator_2x2matrix_method(coordinates, wavelength):
#     # Function to calculate the general terms of the Z matrix (non-tridiagonal).
#     # It uses the 2x2 sub-matrix method as described in the documentation.
#     # This can be seen by the += term in the definition of the different positions
#     # in each loop.
    
#     M = len(coordinates)-1
#     k0 = 2*pi/wavelength
#     mu0 = 4*pi*10**-7
#     epsilon0 = 8.854187812813e-12
#     c = 299792458
#     omega = 2*pi*c/wavelength
#     gamma_e = 0.577215664901532860606512090082402431042159335
    
#     # Predefine the sizes of the output matrices
#     Zmn_left = np.zeros((M,M), dtype=np.complex128)
#     Zmn_right = np.zeros((M,M), dtype=np.complex128)
#     Zmn_diag = np.zeros((M,1), dtype=np.complex128)
    
#     for n in np.arange(M):
#         print(n) # Used as a check to see how far along we are
#         for m in np.arange(M):
#             # Calculate the segment for both m & n. For all following variables,
#             # _P refers to the m/n case (plus), and _M refers to the m-1/n-1 case (min)
#             segm_P = Coordinates_to_segment(coordinates, m)
#             segn_P = Coordinates_to_segment(coordinates, n)
            
#             # Also calculate the previous segment, where we need to take note of
#             # the -1th segment, which shifts over one (in closed)
#             # This is used for the calculation of the m-1 and n-1 terms
            
#             # TODO: ADD OPEN CASE ASWELL!!
#             if m == 0:
#                 segm_M = Coordinates_to_segment(coordinates, -2)
#             else:
#                 segm_M = Coordinates_to_segment(coordinates, m-1)
            
#             if n == 0:
#                 segn_M = Coordinates_to_segment(coordinates, -2)
#             else:
#                 segn_M = Coordinates_to_segment(coordinates, n-1)
#             # Define the segment lengths for both cases
#             dstm_P = segment_length(segm_P)/2
#             dstn_P = segment_length(segn_P)/2
            
#             dstm_M = segment_length(segm_M)/2
#             dstn_M = segment_length(segn_M)/2
            
#             # Create a function which calculates the norm inside of the Modified Bessel Function
#             dstrho = lambda eta, ksi, segm, segn: np.linalg.norm(np.subtract(rho(eta, segm), rho(ksi, segn)))
            
#             # Define the tangential unit vector components
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
            
#             # Do the same, but then for the second (right) double integral
#             integrantPP_real_R = lambda eta, ksi: 0.5*0.5*(1/(2*pi))*np.real(kv(0, 1j*k0*dstrho(eta, ksi, segm_P, segn_P)))
#             integrantPP_imag_R = lambda eta, ksi: 0.5*0.5*(1/(2*pi))*np.imag(kv(0, 1j*k0*dstrho(eta, ksi, segm_P, segn_P)))
#             integrantMM_real_R = lambda eta, ksi: -0.5*-0.5*(1/(2*pi))*np.real(kv(0, 1j*k0*dstrho(eta, ksi, segm_M, segn_M)))
#             integrantMM_imag_R = lambda eta, ksi: -0.5*-0.5*(1/(2*pi))*np.imag(kv(0, 1j*k0*dstrho(eta, ksi, segm_M, segn_M)))
#             integrantPM_real_R = lambda eta, ksi: 0.5*-0.5*(1/(2*pi))*np.real(kv(0, 1j*k0*dstrho(eta, ksi, segm_P, segn_M)))
#             integrantPM_imag_R = lambda eta, ksi: 0.5*-0.5*(1/(2*pi))*np.imag(kv(0, 1j*k0*dstrho(eta, ksi, segm_P, segn_M)))
#             integrantMP_real_R = lambda eta, ksi: -0.5*0.5*(1/(2*pi))*np.real(kv(0, 1j*k0*dstrho(eta, ksi, segm_M, segn_P)))
#             integrantMP_imag_R = lambda eta, ksi: -0.5*0.5*(1/(2*pi))*np.imag(kv(0, 1j*k0*dstrho(eta, ksi, segm_M, segn_P)))
            
#             # Remove (tri-)diagonal terms from the calculations, as these create
#             # singularities, which Python is not able to numerically calculate.
#             if(m == n):
#                 int_PP = 1/(8*pi)*(7 - 4*np.log(2) - 4*(np.log(1j*k0*dstm_P/2) + gamma_e))
#                 int_MM = 1/(8*pi)*(7 - 4*np.log(2) - 4*(np.log(1j*k0*dstm_M/2) + gamma_e))
#                 int_PM = 1/(8*pi)*(5 - 4*np.log(2) - 4*(np.log(1j*k0*((dstm_P + dstn_M)/2)/2) + gamma_e))
#                 int_MP = 1/(8*pi)*(5 - 4*np.log(2) - 4*(np.log(1j*k0*((dstm_M + dstn_P)/2)/2) + gamma_e))
                
#                 Zmn_left[m,n]       += (dstm_P*dstn_P)*1j*omega*mu0*(int_PP)
#                 Zmn_left[m-1,n-1]   += (dstm_M*dstn_M)*1j*omega*mu0*(int_MM)
#                 Zmn_left[m,n-1]     += (dstm_P*dstn_M)*1j*omega*mu0*(int_PM)
#                 Zmn_left[m-1,n]     += (dstm_M*dstn_P)*1j*omega*mu0*(int_MP)
#             elif(m-1 == n or (n == 0 and m == M-1) or n-1 == m or (m == 0 and n == M-1)):
#                 intPP_real, intPP_imag = 0, 0
#                 intMM_real, intMM_imag = 0, 0
#                 intPM_real, intPM_imag = 0, 0
#                 intMP_real, intMP_imag = 0, 0
#                 intPP_real_R, intPP_imag_R = 0, 0
#                 intMM_real_R, intMM_imag_R = 0, 0
#                 intPM_real_R, intPM_imag_R = 0, 0
#                 intMP_real_R, intMP_imag_R = 0, 0
#             else:
#                 intPP_real = integrate.dblquad(integrantPP_real, -1, 1, -1, 1)[0]
#                 intPP_imag = integrate.dblquad(integrantPP_imag, -1, 1, -1, 1)[0]
#                 intMM_real = integrate.dblquad(integrantMM_real, -1, 1, -1, 1)[0]
#                 intMM_imag = integrate.dblquad(integrantMM_imag, -1, 1, -1, 1)[0]
#                 intPM_real = integrate.dblquad(integrantPM_real, -1, 1, -1, 1)[0]
#                 intPM_imag = integrate.dblquad(integrantPM_imag, -1, 1, -1, 1)[0]
#                 intMP_real = integrate.dblquad(integrantMP_real, -1, 1, -1, 1)[0]
#                 intMP_imag = integrate.dblquad(integrantMP_imag, -1, 1, -1, 1)[0]
#                 intPP_real_R = integrate.dblquad(integrantPP_real_R, -1, 1, -1, 1)[0]
#                 intPP_imag_R = integrate.dblquad(integrantPP_imag_R, -1, 1, -1, 1)[0]
#                 intMM_real_R = integrate.dblquad(integrantMM_real_R, -1, 1, -1, 1)[0]
#                 intMM_imag_R = integrate.dblquad(integrantMM_imag_R, -1, 1, -1, 1)[0]
#                 intPM_real_R = integrate.dblquad(integrantPM_real_R, -1, 1, -1, 1)[0]
#                 intPM_imag_R = integrate.dblquad(integrantPM_imag_R, -1, 1, -1, 1)[0]
#                 intMP_real_R = integrate.dblquad(integrantMP_real_R, -1, 1, -1, 1)[0]
#                 intMP_imag_R = integrate.dblquad(integrantMP_imag_R, -1, 1, -1, 1)[0]

#                 # Add the prefactors to the calculated integral based on the location
#                 # of the element in the Z matrix for both the left and right integrals.
#                 Zmn_left[m,n]       += dstm_P*dstn_P*1j*omega*mu0*(tauxm_P*tauxn_P + tauym_P*tauyn_P)*(intPP_real + 1j*intPP_imag)
#                 Zmn_left[m-1,n-1]   += dstm_M*dstn_M*1j*omega*mu0*(tauxm_M*tauxn_M + tauym_M*tauyn_M)*(intMM_real + 1j*intMM_imag)
#                 Zmn_left[m,n-1]     += dstm_P*dstn_M*1j*omega*mu0*(tauxm_P*tauxn_M + tauym_P*tauyn_M)*(intPM_real + 1j*intPM_imag)
#                 Zmn_left[m-1,n]     += dstm_P*dstn_M*1j*omega*mu0*(tauxm_M*tauxn_P + tauym_M*tauyn_P)*(intMP_real + 1j*intMP_imag)
                    
#                 Zmn_right[m,n]      += dstm_P*dstn_P*(tauxm_P**2 + tauym_P**2)/(1j*omega*epsilon0)*(intPP_real_R + 1j*intPP_imag_R)
#                 Zmn_right[m-1,n-1]  += dstm_M*dstn_M*(tauxm_M**2 + tauym_M**2)/(1j*omega*epsilon0)*(intMM_real_R + 1j*intMM_imag_R)
#                 Zmn_right[m,n-1]    += dstm_P*dstn_M*(tauxm_P**2 + tauym_P**2)/(1j*omega*epsilon0)*(intPM_real_R + 1j*intPM_imag_R)
#                 Zmn_right[m-1,n]    += dstm_M*dstn_P*(tauxm_M**2 + tauym_M**2)/(1j*omega*epsilon0)*(intMP_real_R + 1j*intMP_imag_R)
    
#     # Add the integrals together, and return them as the final Z matrix
#     Zmn = Zmn_left + Zmn_right
    
#     return Zmn, Zmn_left, Zmn_right #, Zmn_left_PP, Zmn_left_MM, Zmn_left_PM, Zmn_left_MP, Zmn_right_PP, Zmn_right_MM, Zmn_right_PM, Zmn_right_MP

# def Zmn_adj_calculator(coordinates, wavelength):
#     M = len(coordinates)-1
#     k0 = 2*pi/wavelength
#     mu0 = 4*pi*10**-7
#     epsilon0 = 8.854187812813e-12
#     c = 299792458
#     omega = 2*pi*c/wavelength
#     gamma_e = 0.577215664901532860606512090082402431042159335
    
#     degree = 100
    
#     Zmn_adj = np.zeros((M,1), dtype=np.complex128)
    
#     for m in range(M):
#         print(m)
#         segment_m = Coordinates_to_segment(coordinates, m)
#         if m == 0:
#             segment_n = Coordinates_to_segment(coordinates, -2)
#         else:
#             segment_n = Coordinates_to_segment(coordinates, m-1)
        
#         dst_m = segment_length(segment_m)/2
#         dst_n = segment_length(segment_n)/2
        
#         tau_m = Tangent_vector_coefficients(segment_m)
#         tau_n = Tangent_vector_coefficients(segment_n)
        
#         D = lambda eta, ksi, segm, segn: np.linalg.norm(np.subtract(rho(eta, segm), rho(ksi, segn)))
#         green = lambda eta, ksi, segm, segn: 1/(2*pi)*kv(0, 1j*k0*D(eta, ksi, segm, segn))
        
#         # ADJECENT
#         #x, w = np.polynomial.legendre.leggauss(degree)
        
#         #integrant_PP = RT_plus(x)*RT_plus(x)*green(x, x, segment_m, segment_n)
#         #integrant_MM = RT_min(x)*RT_min(x)*green(x, x, segment_m, segment_n)
        
#         #integrant_PP_R = 0.25*green(x, x, segment_m, segment_n)
#         #integrant_MM_R = 0.25*green(x, x, segment_m, segment_n)
        
#         #I_PP = dst_m*dst_n*1j*omega*mu0*(tau_m[0]*tau_n[0] + tau_m[1]*tau_n[1])*(np.sum(w*w*integrant_PP))
#         #I_MM = dst_m*dst_n*1j*omega*mu0*(tau_m[0]*tau_n[0] + tau_m[1]*tau_n[1])*(np.sum(w*w*integrant_MM))
        
#         #I_PP_R = dst_m*dst_n/(1j*omega*epsilon0)*(np.sum(w*w*integrant_PP_R))
#         #I_MM_R = dst_m*dst_n/(1j*omega*epsilon0)*(np.sum(w*w*integrant_MM_R))
        
#         # SELF TERM
#         #I_PM = (dst_m**2)*1j*omega*mu0*(1/(8*pi)*(5 - 4*np.log(2) - 4*(np.log(1j*k0*dst_m/2) + gamma_e)))
#         #I_PM_R = (dst_m**2)*(1j*omega*epsilon0)*(-1/(8*pi)*(6 - 4*np.log(2) - 4*(np.log(1j*k0*dst_m/2) + gamma_e)))
        
#         # REGULAR
#         integrant_PM_real = lambda eta, ksi: RT_plus(eta)*RT_min(ksi)*np.real(green(eta, ksi, segment_m, segment_n))       
#         integrant_PM_imag = lambda eta, ksi: RT_plus(eta)*RT_min(ksi)*np.imag(green(eta, ksi, segment_m, segment_n))
#         integrant_MP_real = lambda eta, ksi: RT_min(eta)*RT_plus(ksi)*np.real(green(eta, ksi, segment_m, segment_n))       
#         integrant_MP_imag = lambda eta, ksi: RT_min(eta)*RT_plus(ksi)*np.imag(green(eta, ksi, segment_m, segment_n))
#         integrant_PP_real = lambda eta, ksi: RT_plus(eta)*RT_plus(ksi)*np.real(green(eta, ksi, segment_m, segment_n))       
#         integrant_PP_imag = lambda eta, ksi: RT_plus(eta)*RT_plus(ksi)*np.imag(green(eta, ksi, segment_m, segment_n))
#         integrant_MM_real = lambda eta, ksi: RT_plus(eta)*RT_plus(ksi)*np.real(green(eta, ksi, segment_m, segment_n))       
#         integrant_MM_imag = lambda eta, ksi: RT_plus(eta)*RT_plus(ksi)*np.imag(green(eta, ksi, segment_m, segment_n))
        
#         integrant_PM_R_real = lambda eta, ksi: 0.5*-0.5*np.real(green(eta, ksi, segment_m, segment_n))       
#         integrant_PM_R_imag = lambda eta, ksi: 0.5*-0.5*np.imag(green(eta, ksi, segment_m, segment_n))
#         integrant_MP_R_real = lambda eta, ksi: -0.5*0.5*np.real(green(eta, ksi, segment_m, segment_n))
#         integrant_MP_R_imag = lambda eta, ksi: -0.5*0.5*np.imag(green(eta, ksi, segment_m, segment_n)) 
#         integrant_PP_R_real = lambda eta, ksi: 0.5*0.5*np.real(green(eta, ksi, segment_m, segment_n))       
#         integrant_PP_R_imag = lambda eta, ksi: 0.5*0.5*np.imag(green(eta, ksi, segment_m, segment_n))
#         integrant_MM_R_real = lambda eta, ksi: -0.5*-0.5*np.real(green(eta, ksi, segment_m, segment_n))       
#         integrant_MM_R_imag = lambda eta, ksi: -0.5*-0.5*np.imag(green(eta, ksi, segment_m, segment_n))
        
#         I_PM_real = integrate.dblquad(integrant_PM_real, -1, 1, -1, 1)[0]
#         I_PM_imag = integrate.dblquad(integrant_PM_imag, -1, 1, -1, 1)[0]
#         I_MP_real = integrate.dblquad(integrant_MP_real, -1, 1, -1, 1)[0]
#         I_MP_imag = integrate.dblquad(integrant_MP_imag, -1, 1, -1, 1)[0]
#         I_PP_real = integrate.dblquad(integrant_PP_real, -1, 1, -1, 1)[0]
#         I_PP_imag = integrate.dblquad(integrant_PP_imag, -1, 1, -1, 1)[0]
#         I_MM_real = integrate.dblquad(integrant_MM_real, -1, 1, -1, 1)[0]
#         I_MM_imag = integrate.dblquad(integrant_MM_imag, -1, 1, -1, 1)[0]
        
#         I_PM_R_real = integrate.dblquad(integrant_PM_R_real, -1, 1, -1, 1)[0]
#         I_PM_R_imag = integrate.dblquad(integrant_PM_R_imag, -1, 1, -1, 1)[0]
#         I_MP_R_real = integrate.dblquad(integrant_MP_R_real, -1, 1, -1, 1)[0]
#         I_MP_R_imag = integrate.dblquad(integrant_MP_R_imag, -1, 1, -1, 1)[0]
#         I_PP_R_real = integrate.dblquad(integrant_PP_R_real, -1, 1, -1, 1)[0]
#         I_PP_R_imag = integrate.dblquad(integrant_PP_R_imag, -1, 1, -1, 1)[0]
#         I_MM_R_real = integrate.dblquad(integrant_MM_R_real, -1, 1, -1, 1)[0]
#         I_MM_R_imag = integrate.dblquad(integrant_MM_R_imag, -1, 1, -1, 1)[0]
        
#         I_PM = dst_m*dst_n*1j*omega*mu0*(tau_m[0]*tau_n[0] + tau_m[1]*tau_n[1])*(I_PM_real + 1j*I_PM_imag)
#         I_PM_R = dst_m*dst_n/(1j*omega*epsilon0)*(tau_m[0]*tau_m[0] + tau_m[1]*tau_m[1])*(I_PM_R_real + 1j*I_PM_R_imag)
#         I_MP = dst_m*dst_n*1j*omega*mu0*(tau_m[0]*tau_n[0] + tau_m[1]*tau_n[1])*(I_MP_real + 1j*I_MP_imag)
#         I_MP_R = dst_m*dst_n/(1j*omega*epsilon0)*(tau_m[0]*tau_m[0] + tau_m[1]*tau_m[1])*(I_MP_R_real + 1j*I_MP_R_imag)
#         I_PP = dst_m*dst_n*1j*omega*mu0*(tau_m[0]*tau_n[0] + tau_m[1]*tau_n[1])*(I_PP_real + 1j*I_PP_imag)
#         I_PP_R = dst_m*dst_n/(1j*omega*epsilon0)*(tau_m[0]*tau_m[0] + tau_m[1]*tau_m[1])*(I_PP_R_real + 1j*I_PP_R_imag)
#         I_MM = dst_m*dst_n*1j*omega*mu0*(tau_m[0]*tau_n[0] + tau_m[1]*tau_n[1])*(I_MM_real + 1j*I_MM_imag)
#         I_MM_R = dst_m*dst_n/(1j*omega*epsilon0)*(tau_m[0]*tau_m[0] + tau_m[1]*tau_m[1])*(I_MM_R_real + 1j*I_MM_R_imag)
        
#         # ADD THEM UP
#         Zmn_adj[m] = I_PP + I_MM + I_PM + I_MP + I_PP_R + I_MM_R + I_PM_R + I_MP_R 
    
#     return Zmn_adj

def Zmn_adj_calculator(coordinates, wavelength):
    M = len(coordinates)-1
    k0 = 2*pi/wavelength
    mu0 = 4*pi*10**-7
    epsilon0 = 8.854187812813e-12
    c = 299792458
    omega = 2*pi*c/wavelength
    gamma_e = 0.577215664901532860606512090082402431042159335
    
    degree = 50
    
    Zmn_adj = np.zeros((M,1), dtype=np.complex128)
    
    for m in range(M):
        print(m)
        segment_m = Coordinates_to_segment(coordinates, m)
        if m == 0:
            segment_m_M = Coordinates_to_segment(coordinates, -2)
        else:
            segment_m_M = Coordinates_to_segment(coordinates, m-1)
            
        if m == 0:
            segment_n = Coordinates_to_segment(coordinates, -2)
            segment_n_M = Coordinates_to_segment(coordinates, -3)
        else:
            segment_n = Coordinates_to_segment(coordinates, m-1)
            segment_n_M = Coordinates_to_segment(coordinates, m-2)
        
        dst_m = segment_length(segment_m)/2
        dst_m_M = segment_length(segment_m_M)/2
        dst_n = segment_length(segment_n)/2
        dst_n_M = segment_length(segment_n_M)/2
        
        tau_m = Tangent_vector_coefficients(segment_m)
        tau_m_M = Tangent_vector_coefficients(segment_m_M)
        tau_n = Tangent_vector_coefficients(segment_n)
        tau_n_M = Tangent_vector_coefficients(segment_n_M)
        
        D = lambda eta, ksi, segm, segn: np.linalg.norm(np.subtract(rho(eta, segm), rho(ksi, segn)))
        green = lambda eta, ksi, segm, segn: 1/(2*pi)*kv(0, 1j*k0*D(eta, ksi, segm, segn))
        
        # ADJECENT
        xh, wh = np.polynomial.legendre.leggauss(degree)
        xj = xh
        wj = wh
        integrant_PP, integrant_PP_R = 0, 0
        integrant_MM, integrant_MM_R = 0, 0
        for h in range(degree):
            for j in range(degree):
                integrant_PP += wh[h]*wj[j]*RT_plus(xh[h])*RT_plus(xj[j])*green(xh[h], xj[j], segment_m_M, segment_n_M)
                integrant_MM += wh[h]*wj[j]*RT_min(xh[h])*RT_min(xj[j])*green(xh[h], xj[j], segment_m, segment_n)
                integrant_PP_R += wh[h]*wj[j]*0.25*green(xh[h], xj[j], segment_m_M, segment_n_M)
                integrant_MM_R += wh[h]*wj[j]*0.25*green(xh[h], xj[j], segment_m, segment_n)
        
        I_PP = dst_m_M*dst_n_M*1j*omega*mu0*(tau_m_M[0]*tau_n_M[0] + tau_m_M[1]*tau_n_M[1])*integrant_PP
        I_MM = dst_m*dst_n*1j*omega*mu0*(tau_m[0]*tau_n[0] + tau_m[1]*tau_n[1])*integrant_PP_R
        
        I_PP_R = dst_m_M*dst_n_M/(1j*omega*epsilon0)*integrant_MM
        I_MM_R = dst_m*dst_n/(1j*omega*epsilon0)*integrant_MM_R
        
        # SELF TERM
        I_PM = (dst_n**2)*1j*omega*mu0*(1/(8*pi)*(5 - 4*np.log(2) - 4*(np.log(1j*k0*dst_n/2) + gamma_e)))
        I_PM_R = (dst_n**2)*(1j*omega*epsilon0)*(-1/(8*pi)*(6 - 4*np.log(2) - 4*(np.log(1j*k0*dst_n/2) + gamma_e)))
        
        # REGULAR
        integrant_MP_real = lambda eta, ksi: RT_min(eta)*RT_plus(ksi)*np.real(green(eta, ksi, segment_m, segment_n_M))       
        integrant_MP_imag = lambda eta, ksi: RT_min(eta)*RT_plus(ksi)*np.imag(green(eta, ksi, segment_m, segment_n_M))
        
        integrant_MP_R_real = lambda eta, ksi: -0.5*0.5*np.real(green(eta, ksi, segment_m, segment_n_M))
        integrant_MP_R_imag = lambda eta, ksi: -0.5*0.5*np.imag(green(eta, ksi, segment_m, segment_n_M)) 
        
        I_MP_real = integrate.dblquad(integrant_MP_real, -1, 1, -1, 1)[0]
        I_MP_imag = integrate.dblquad(integrant_MP_imag, -1, 1, -1, 1)[0]
        
        I_MP_R_real = integrate.dblquad(integrant_MP_R_real, -1, 1, -1, 1)[0]
        I_MP_R_imag = integrate.dblquad(integrant_MP_R_imag, -1, 1, -1, 1)[0]

        I_MP = dst_m*dst_n_M*1j*omega*mu0*(tau_m[0]*tau_n_M[0] + tau_m[1]*tau_n_M[1])*(I_MP_real + 1j*I_MP_imag)
        I_MP_R = dst_m*dst_n_M/(1j*omega*epsilon0)*(tau_m[0]*tau_m[0] + tau_m[1]*tau_m[1])*(I_MP_R_real + 1j*I_MP_R_imag)
        
        # ADD THEM UP
        Zmn_adj[m] = I_PP + I_MM + I_PM + I_MP + I_PP_R + I_MM_R + I_PM_R + I_MP_R 
    return Zmn_adj

def Zmn_diag_calculator_V2(coordinates, wavelength):
    M = len(coordinates)-1
    k0 = 2*pi/wavelength
    mu0 = 4*pi*10**-7
    epsilon0 = 8.854187812813e-12
    c = 299792458
    omega = 2*pi*c/wavelength
    gamma_e = 0.577215664901532860606512090082402431042159335
    
    Zdiag = np.zeros((M,1), dtype=np.complex128)
    
    for m in range(M):
        segment_m = Coordinates_to_segment(coordinates, m)
        dst = segment_length(segment_m)/2
        
        intPP = 1/(8*pi)*(7 - 4*np.log(2) - 4*(np.log(1j*k0*dst/2) + gamma_e))
        intMM = 1/(8*pi)*(7 - 4*np.log(2) - 4*(np.log(1j*k0*dst/2) + gamma_e))
        intPM = 1/(8*pi)*(5 - 4*np.log(2) - 4*(np.log(1j*k0*dst/2) + gamma_e))
        intMP = 1/(8*pi)*(5 - 4*np.log(2) - 4*(np.log(1j*k0*dst/2) + gamma_e))
        
        Zdiag[m]    += (dst**2)*1j*omega*mu0*(intPP)
        Zdiag[m-1]  += (dst**2)*1j*omega*mu0*(intMM)
        Zdiag[m]    += (dst**2)*1j*omega*mu0*(intPM)
        Zdiag[m-1]  += (dst**2)*1j*omega*mu0*(intMP)
    
    return Zdiag
    

def Zmn_diag_calculator(coordinates, wavelength):
    # Calculate the diagonal of the Z matrix us the self term approximation
    M = len(coordinates)-1
    mu0 = 4 * pi * 10**-7
    epsilon0 = 8.854187812813e-12
    c = 299792458
    omega = 2*pi*c/wavelength
    
    # Predefine the array to be zero
    Zdiag = np.zeros((M,1), dtype=np.complex128)
    
    for m in np.arange(M):
        segm = Coordinates_to_segment(coordinates, m)
        
        # Calculate the self-term, based on Appendix A3. This returns an array
        # of two variables (per m), with Is[0] -> Is[1] and Is[1] -> Is[eta*ksi] 
        Is = Self_term_integral(coordinates, wavelength, m)
        
        # Calculate the tangential unit vector components
        tauxm, tauym = Tangent_vector_coefficients(segm)
        
        # Add the prefactors to the calculated self integral, where we do not
        # use the distance, as that is taken into account in the self-term itself.
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
    # Scipy inbuilt function to calculate the harmonic number phi(k).
    # Start from k+1, as psi(0) = -inf, which we want to ignore.
    return psi(k + 1)

def EFIE_TM(coordinates,wavelength,angle):
    # The main algorithm
    Ein_x, Ein_y = DiscritizeEin(coordinates,wavelength,angle)
    Zm = Zmn_calculator(coordinates,wavelength)
    # Solve for both the x and y incident field
    Jz_x = np.dot(np.linalg.inv(Zm),Ein_x)
    Jz_y = np.dot(np.linalg.inv(Zm), Ein_y)
    #Jz = np.dot(np.linalg.inv(Zm), Ein_x + Ein_y)
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

def Escatter(Jz_x,Jz_y,rho,coordinates,wavelength):
    # Calculate the Electric field scattered from the object on given coordinates
    mu0 = 4*pi*10**-7
    c = 299792458
    omega = 2*pi*c/wavelength
    
    # G can be complex so allocate complex matrix
    G_P = np.zeros(len(Jz_x), dtype=np.complex128)
    G_M = np.zeros(len(Jz_x), dtype=np.complex128)
    GRight_P = np.zeros(len(Jz_x), dtype=np.complex128)
    GRight_M = np.zeros(len(Jz_x), dtype=np.complex128)
    Esc_x, Esc_y = np.zeros(len(Jz_x), dtype=np.complex128), np.zeros(len(Jz_x), dtype=np.complex128)
    # Note length Jz = length coordinates-1, so no -1 nessesary here
    for n in np.arange(len(Jz_x)):
        segment = Coordinates_to_segment(coordinates,n)
        
        # Calculate the real and imaginary part seperately
        GReal_P = lambda eta: (1/2)*(1 + eta)*np.real(greenEsc(rho,eta,segment,wavelength))
        GImag_P = lambda eta: (1/2)*(1 + eta)*np.imag(greenEsc(rho,eta,segment,wavelength))
        GReal_M = lambda eta: (1/2)*(1 - eta)*np.real(greenEsc(rho,eta,segment,wavelength))
        GImag_M = lambda eta: (1/2)*(1 - eta)*np.imag(greenEsc(rho,eta,segment,wavelength))
        
        GRealRight_P = lambda eta: (1/2)*np.real(greenEsc(rho,eta,segment,wavelength))
        GImagRight_P = lambda eta: (1/2)*np.imag(greenEsc(rho,eta,segment,wavelength))
        GRealRight_M = lambda eta: (-1/2)*np.real(greenEsc(rho,eta,segment,wavelength))
        GImagRight_M = lambda eta: (-1/2)*np.imag(greenEsc(rho,eta,segment,wavelength))

        IntReal_P = integrate.quad(GReal_P, -1, 1)[0]
        IntImag_P = integrate.quad(GImag_P, -1, 1)[0]
        IntReal_M = integrate.quad(GReal_M, -1, 1)[0]
        IntImag_M = integrate.quad(GImag_M, -1, 1)[0]
        
        IntRealRight_P = integrate.quad(GRealRight_P, -1, 1)[0]
        IntImagRight_P = integrate.quad(GImagRight_P, -1, 1)[0]
        IntRealRight_M = integrate.quad(GRealRight_M, -1, 1)[0]
        IntImagRight_M = integrate.quad(GImagRight_M, -1, 1)[0]
        
        # Correct for the basis function used
        dst = segment_length(segment)
        G_P[n] = dst*(IntReal_P + 1j*IntImag_P)
        G_M[n] = dst*(IntReal_M + 1j*IntImag_M)
        
        GRight_P[n] = dst*(IntRealRight_P + 1j*IntImagRight_P)
        GRight_M[n] = dst*(IntRealRight_M + 1j*IntImagRight_M)

    # Compute the scattered field for both the x and y components
    # Use np.roll to convert Jn+ into Jn- (Height associated with the negative triangle function)
    Esc_x = 1j*omega*mu0*(np.dot(G_P, np.roll(Jz_x,1)) + np.dot(G_M,Jz_x))
    Esc_y = 1j*omega*mu0*(np.dot(G_P, np.roll(Jz_y,1)) + np.dot(G_M,Jz_y))
    return Esc_x, Esc_y

def greenEsc(r,ksi,segment,wavelength):
    # Special case, only the basis function applied
    k0 = 2*pi/wavelength
    Pn = rho(ksi, segment)
    
    # Compute Greens function
    G = (1/(2*pi))*kv(0, 1j*k0*np.linalg.norm(np.subtract(r, Pn)))
    
    return G

# UNCOMMENT THIS PART TO TEST FUNCTIONS SEPARATELY!
## INPUTS
# angle = 1.05*pi/2
c = 299792458
f = 150*10**6
# mu0 = 4*pi*10**-7
wavelength = c/f

# # # # Generate a closed circle of radius 1 with 30 data points
M = 10
R = 1
Data = createcircle(M,R)
#Data = np.asarray([[4,1],[3.7,1],[3,1],[2.3,1],[1,1],[0.5,1],[0,1],[-0.5,1],[-1,1],[-2,1],[-3,1],[-3.5,1],[-5,1]])
#M = np.size(Data, 0)

# # # # # ## MAIN
#Zmn_matrix, Zmn_left_matrix, Zmn_right_matrix = Zmn_calculator_2x2matrix_method(Data, wavelength)
#Zmn_left = Zmn_calculator_left(Data, wavelength)
#Zmn_right = Zmn_calculator_right(Data, wavelength)
#Zmn = Zmn_left + Zmn_right
#Zmn = Zmn_calculator(Data, wavelength)
# #Z_diag_test = Zmn_diag_calculator(Data, wavelength)
Zmn_diag_V2 = Zmn_diag_calculator_V2(Data, wavelength)
Zmn_diag = Zmn_diag_calculator(Data, wavelength)
# # np.fill_diagonal(Zmn, Z_diag_test.flatten())
# # # # # print(Z_diag_test)  
#Zmn_adj = Zmn_adj_calculator(Data, wavelength)
#Zmn_matrix_method = Zmn_calculator_2x2matrix_method(Data, wavelength)





