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
    Zmn_diag = Zmn_diag_calculator(coordinates, wavelength)
    np.fill_diagonal(Zmn, Zmn_diag.flatten())
    
    # Add the super and sub diagonal to the matrix
    Zmn_adj = Zmn_adj_calculator(coordinates, wavelength)
    np.fill_diagonal(Zmn[1:], Zmn_adj.flatten()) # sub
    np.fill_diagonal(Zmn[:,1:], Zmn_adj.flatten()) # super
    Zmn[0,M-1] = Zmn[0,1]
    Zmn[M-1,0] = Zmn[0,1]
    
    # Matrix provided by Mathematica
    Zmn_ref = np.asarray([
          [13.1017 +13.3253j, 11.454 +5.15144j, 7.34782 -2.90897j, 2.73285 -5.00781j, -0.558095-4.03342j, -1.77252-2.01069j, -1.34711-0.482235j, -0.311666-0.0990299j, 0.429282 -0.684847j, 0.476103 -1.6766j, -0.0966428-2.56387j, -0.980797-3.10127j, -1.86173-3.29645j, -2.52852-3.2912j, -2.879-3.24149j, -2.879-3.24149j, -2.52852-3.2912j, -1.86173-3.29645j, -0.980797-3.10127j, -0.0966428-2.56387j, 0.476103 -1.6766j, 0.429282 -0.684847j, -0.311666-0.0990299j, -1.34711-0.482235j, -1.77252-2.01069j, -0.558095-4.03342j, 2.73285 -5.00781j, 7.34782 -2.90897j, 11.454 +5.15144j],
          [11.454 +5.15144j, 13.1017 +13.3253j, 11.454 +5.15144j, 7.34782 -2.90897j, 2.73285 -5.00781j, -0.558095-4.03342j, -1.77252-2.01069j, -1.34711-0.482235j, -0.311666-0.0990299j, 0.429282 -0.684847j, 0.476103 -1.6766j, -0.0966428-2.56387j, -0.980797-3.10127j, -1.86173-3.29645j, -2.52852-3.2912j, -2.879-3.24149j, -2.879-3.24149j, -2.52852-3.2912j, -1.86173-3.29645j, -0.980797-3.10127j, -0.0966428-2.56387j, 0.476103 -1.6766j, 0.429282 -0.684847j, -0.311666-0.0990299j, -1.34711-0.482235j, -1.77252-2.01069j, -0.558095-4.03342j, 2.73285 -5.00781j, 7.34782 -2.90897j],
          [7.34782 -2.90897j, 11.454 +5.15144j, 13.1017 +13.3253j, 11.454 +5.15144j, 7.34782 -2.90897j, 2.73285 -5.00781j, -0.558095-4.03342j, -1.77252-2.01069j, -1.34711-0.482235j, -0.311666-0.0990299j, 0.429282 -0.684847j, 0.476103 -1.6766j, -0.0966428-2.56387j, -0.980797-3.10127j, -1.86173-3.29645j, -2.52852-3.2912j, -2.879-3.24149j, -2.879-3.24149j, -2.52852-3.2912j, -1.86173-3.29645j, -0.980797-3.10127j, -0.0966428-2.56387j, 0.476103 -1.6766j, 0.429282 -0.684847j, -0.311666-0.0990298j, -1.34711-0.482235j, -1.77252-2.01069j, -0.558095-4.03342j, 2.73285 -5.00781j],
          [2.73285 -5.00781j, 7.34782 -2.90897j, 11.454 +5.15144j, 13.1017 +13.3253j, 11.454 +5.15144j, 7.34782 -2.90897j, 2.73285 -5.00781j, -0.558095-4.03342j, -1.77252-2.01069j, -1.34711-0.482235j, -0.311666-0.0990299j, 0.429282 -0.684847j, 0.476103 -1.6766j, -0.0966428-2.56387j, -0.980797-3.10127j, -1.86173-3.29645j, -2.52852-3.2912j, -2.879-3.24149j, -2.879-3.24149j, -2.52852-3.2912j, -1.86173-3.29645j, -0.980797-3.10127j, -0.0966428-2.56387j, 0.476103 -1.6766j, 0.429282 -0.684847j, -0.311666-0.0990299j, -1.34711-0.482235j, -1.77252-2.01069j, -0.558095-4.03342j],
          [-0.558095-4.03342j, 2.73285 -5.00781j, 7.34782 -2.90897j, 11.454 +5.15144j, 13.1017 +13.3253j, 11.454 +5.15144j, 7.34782 -2.90897j, 2.73285 -5.00781j, -0.558095-4.03342j, -1.77252-2.01069j, -1.34711-0.482235j, -0.311666-0.0990298j, 0.429282 -0.684847j, 0.476103 -1.6766j, -0.0966428-2.56387j, -0.980797-3.10127j, -1.86173-3.29645j, -2.52852-3.2912j, -2.879-3.24149j, -2.879-3.24149j, -2.52852-3.2912j, -1.86173-3.29645j, -0.980797-3.10127j, -0.0966428-2.56387j, 0.476103 -1.6766j, 0.429282 -0.684847j, -0.311666-0.0990299j, -1.34711-0.482235j, -1.77252-2.01069j],
          [-1.77252-2.01069j, -0.558095-4.03342j, 2.73285 -5.00781j, 7.34782 -2.90897j, 11.454 +5.15144j, 13.1017 +13.3253j, 11.454 +5.15144j, 7.34782 -2.90897j, 2.73285 -5.00781j, -0.558095-4.03342j, -1.77252-2.01069j, -1.34711-0.482235j, -0.311666-0.0990299j, 0.429282 -0.684847j, 0.476103 -1.6766j, -0.0966428-2.56387j, -0.980797-3.10127j, -1.86173-3.29645j, -2.52852-3.2912j, -2.879-3.24149j, -2.879-3.24149j, -2.52852-3.2912j, -1.86173-3.29645j, -0.980797-3.10127j, -0.0966427-2.56387j, 0.476103 -1.6766j, 0.429282 -0.684847j, -0.311666-0.0990298j, -1.34711-0.482235j],
          [-1.34711-0.482235j, -1.77252-2.01069j, -0.558095-4.03342j, 2.73285 -5.00781j, 7.34782 -2.90897j, 11.454 +5.15144j, 13.1017 +13.3253j, 11.454 +5.15144j, 7.34782 -2.90897j, 2.73285 -5.00781j, -0.558095-4.03342j, -1.77252-2.01069j, -1.34711-0.482235j, -0.311666-0.0990299j, 0.429282 -0.684847j, 0.476103 -1.6766j, -0.0966428-2.56387j, -0.980797-3.10127j, -1.86173-3.29645j, -2.52852-3.2912j, -2.879-3.24149j, -2.879-3.24149j, -2.52852-3.2912j, -1.86173-3.29645j, -0.980797-3.10127j, -0.0966428-2.56387j, 0.476103 -1.6766j, 0.429282 -0.684847j, -0.311666-0.0990299j],
          [-0.311666-0.0990299j, -1.34711-0.482235j, -1.77252-2.01069j, -0.558095-4.03342j, 2.73285 -5.00781j, 7.34782 -2.90897j, 11.454 +5.15144j, 13.1017 +13.3253j, 11.454 +5.15144j, 7.34782 -2.90897j, 2.73285 -5.00781j, -0.558095-4.03342j, -1.77252-2.01069j, -1.34711-0.482235j, -0.311666-0.0990299j, 0.429282 -0.684847j, 0.476103 -1.6766j, -0.0966428-2.56387j, -0.980797-3.10127j, -1.86173-3.29645j, -2.52852-3.2912j, -2.879-3.24149j, -2.879-3.24149j, -2.52852-3.2912j, -1.86173-3.29645j, -0.980797-3.10127j, -0.0966428-2.56387j, 0.476103 -1.6766j, 0.429282 -0.684847j],
          [0.429282 -0.684847j, -0.311666-0.0990299j, -1.34711-0.482235j, -1.77252-2.01069j, -0.558095-4.03342j, 2.73285 -5.00781j, 7.34782 -2.90897j, 11.454 +5.15144j, 13.1017 +13.3253j, 11.454 +5.15144j, 7.34782 -2.90897j, 2.73285 -5.00781j, -0.558095-4.03342j, -1.77252-2.01069j, -1.34711-0.482235j, -0.311666-0.0990299j, 0.429282 -0.684847j, 0.476103 -1.6766j, -0.0966428-2.56387j, -0.980797-3.10127j, -1.86173-3.29645j, -2.52852-3.2912j, -2.879-3.24149j, -2.879-3.24149j, -2.52852-3.2912j, -1.86173-3.29645j, -0.980797-3.10127j, -0.0966428-2.56387j, 0.476103 -1.6766j],
          [0.476103 -1.6766j, 0.429282 -0.684847j, -0.311666-0.0990299j, -1.34711-0.482235j, -1.77252-2.01069j, -0.558095-4.03342j, 2.73285 -5.00781j, 7.34782 -2.90897j, 11.454 +5.15144j, 13.1017 +13.3253j, 11.454 +5.15144j, 7.34782 -2.90897j, 2.73285 -5.00781j, -0.558095-4.03342j, -1.77252-2.01069j, -1.34711-0.482235j, -0.311666-0.0990299j, 0.429282 -0.684847j, 0.476103 -1.6766j, -0.0966428-2.56387j, -0.980797-3.10127j, -1.86173-3.29645j, -2.52852-3.2912j, -2.879-3.24149j, -2.879-3.24149j, -2.52852-3.2912j, -1.86173-3.29645j, -0.980797-3.10127j, -0.0966428-2.56387j],
          [-0.0966428-2.56387j, 0.476103 -1.6766j, 0.429282 -0.684847j, -0.311666-0.0990299j, -1.34711-0.482235j, -1.77252-2.01069j, -0.558095-4.03342j, 2.73285 -5.00781j, 7.34782 -2.90897j, 11.454 +5.15144j, 13.1017 +13.3253j, 11.454 +5.15144j, 7.34782 -2.90897j, 2.73285 -5.00781j, -0.558095-4.03342j, -1.77252-2.01069j, -1.34711-0.482235j, -0.311666-0.0990299j, 0.429282 -0.684847j, 0.476103 -1.6766j, -0.0966428-2.56387j, -0.980797-3.10127j, -1.86173-3.29645j, -2.52852-3.2912j, -2.879-3.24149j, -2.879-3.24149j, -2.52852-3.2912j, -1.86173-3.29645j, -0.980797-3.10127j],
          [-0.980797-3.10127j, -0.0966428-2.56387j, 0.476103 -1.6766j, 0.429282 -0.684847j, -0.311666-0.0990298j, -1.34711-0.482235j, -1.77252-2.01069j, -0.558095-4.03342j, 2.73285 -5.00781j, 7.34782 -2.90897j, 11.454 +5.15144j, 13.1017 +13.3253j, 11.454 +5.15144j, 7.34782 -2.90897j, 2.73285 -5.00781j, -0.558095-4.03342j, -1.77252-2.01069j, -1.34711-0.482235j, -0.311666-0.0990299j, 0.429282 -0.684847j, 0.476103 -1.6766j, -0.0966428-2.56387j, -0.980797-3.10127j, -1.86173-3.29645j, -2.52852-3.2912j, -2.879-3.24149j, -2.879-3.24149j, -2.52852-3.2912j, -1.86173-3.29645j],
          [-1.86173-3.29645j, -0.980797-3.10127j, -0.0966428-2.56387j, 0.476103 -1.6766j, 0.429282 -0.684847j, -0.311666-0.0990299j, -1.34711-0.482235j, -1.77252-2.01069j, -0.558095-4.03342j, 2.73285 -5.00781j, 7.34782 -2.90897j, 11.454 +5.15144j, 13.1017 +13.3253j, 11.454 +5.15144j, 7.34782 -2.90897j, 2.73285 -5.00781j, -0.558095-4.03342j, -1.77252-2.01069j, -1.34711-0.482235j, -0.311666-0.0990299j, 0.429282 -0.684847j, 0.476103 -1.6766j, -0.0966428-2.56387j, -0.980797-3.10127j, -1.86173-3.29645j, -2.52852-3.2912j, -2.879-3.24149j, -2.879-3.24149j, -2.52852-3.2912j],
          [-2.52852-3.2912j, -1.86173-3.29645j, -0.980797-3.10127j, -0.0966428-2.56387j, 0.476103 -1.6766j, 0.429282 -0.684847j, -0.311666-0.0990299j, -1.34711-0.482235j, -1.77252-2.01069j, -0.558095-4.03342j, 2.73285 -5.00781j, 7.34782 -2.90897j, 11.454 +5.15144j, 13.1017 +13.3253j, 11.454 +5.15144j, 7.34782 -2.90897j, 2.73285 -5.00781j, -0.558095-4.03342j, -1.77252-2.01069j, -1.34711-0.482235j, -0.311666-0.0990299j, 0.429282 -0.684847j, 0.476103 -1.6766j, -0.0966428-2.56387j, -0.980797-3.10127j, -1.86173-3.29645j, -2.52852-3.2912j, -2.879-3.24149j, -2.879-3.24149j],
          [-2.879-3.24149j, -2.52852-3.2912j, -1.86173-3.29645j, -0.980797-3.10127j, -0.0966428-2.56387j, 0.476103 -1.6766j, 0.429282 -0.684847j, -0.311666-0.0990299j, -1.34711-0.482235j, -1.77252-2.01069j, -0.558095-4.03342j, 2.73285 -5.00781j, 7.34782 -2.90897j, 11.454 +5.15144j, 13.1017 +13.3253j, 11.454 +5.15144j, 7.34782 -2.90897j, 2.73285 -5.00781j, -0.558095-4.03342j, -1.77252-2.01069j, -1.34711-0.482235j, -0.311666-0.0990299j, 0.429282 -0.684847j, 0.476103 -1.6766j, -0.0966428-2.56387j, -0.980797-3.10127j, -1.86173-3.29645j, -2.52852-3.2912j, -2.879-3.24149j],
          [-2.879-3.24149j, -2.879-3.24149j, -2.52852-3.2912j, -1.86173-3.29645j, -0.980797-3.10127j, -0.0966428-2.56387j, 0.476103 -1.6766j, 0.429282 -0.684847j, -0.311666-0.0990299j, -1.34711-0.482235j, -1.77252-2.01069j, -0.558095-4.03342j, 2.73285 -5.00781j, 7.34782 -2.90897j, 11.454 +5.15144j, 13.1017 +13.3253j, 11.454 +5.15144j, 7.34782 -2.90897j, 2.73285 -5.00781j, -0.558095-4.03342j, -1.77252-2.01069j, -1.34711-0.482235j, -0.311666-0.0990299j, 0.429282 -0.684847j, 0.476103 -1.6766j, -0.0966428-2.56387j, -0.980797-3.10127j, -1.86173-3.29645j, -2.52852-3.2912j],
          [-2.52852-3.2912j, -2.879-3.24149j, -2.879-3.24149j, -2.52852-3.2912j, -1.86173-3.29645j, -0.980797-3.10127j, -0.0966428-2.56387j, 0.476103 -1.6766j, 0.429282 -0.684847j, -0.311666-0.0990299j, -1.34711-0.482235j, -1.77252-2.01069j, -0.558095-4.03342j, 2.73285 -5.00781j, 7.34782 -2.90897j, 11.454 +5.15144j, 13.1017 +13.3253j, 11.454 +5.15144j, 7.34782 -2.90897j, 2.73285 -5.00781j, -0.558095-4.03342j, -1.77252-2.01069j, -1.34711-0.482235j, -0.311666-0.0990299j, 0.429282 -0.684847j, 0.476103 -1.6766j, -0.0966428-2.56387j, -0.980797-3.10127j, -1.86173-3.29645j],
          [-1.86173-3.29645j, -2.52852-3.2912j, -2.879-3.24149j, -2.879-3.24149j, -2.52852-3.2912j, -1.86173-3.29645j, -0.980797-3.10127j, -0.0966428-2.56387j, 0.476103 -1.6766j, 0.429282 -0.684847j, -0.311666-0.0990299j, -1.34711-0.482235j, -1.77252-2.01069j, -0.558095-4.03342j, 2.73285 -5.00781j, 7.34782 -2.90897j, 11.454 +5.15144j, 13.1017 +13.3253j, 11.454 +5.15144j, 7.34782 -2.90897j, 2.73285 -5.00781j, -0.558095-4.03342j, -1.77252-2.01069j, -1.34711-0.482235j, -0.311666-0.0990299j, 0.429282 -0.684847j, 0.476103 -1.6766j, -0.0966428-2.56387j, -0.980797-3.10127j],
          [-0.980797-3.10127j, -1.86173-3.29645j, -2.52852-3.2912j, -2.879-3.24149j, -2.879-3.24149j, -2.52852-3.2912j, -1.86173-3.29645j, -0.980797-3.10127j, -0.0966428-2.56387j, 0.476103 -1.6766j, 0.429282 -0.684847j, -0.311666-0.0990299j, -1.34711-0.482235j, -1.77252-2.01069j, -0.558095-4.03342j, 2.73285 -5.00781j, 7.34782 -2.90897j, 11.454 +5.15144j, 13.1017 +13.3253j, 11.454 +5.15144j, 7.34782 -2.90897j, 2.73285 -5.00781j, -0.558095-4.03342j, -1.77252-2.01069j, -1.34711-0.482235j, -0.311666-0.0990298j, 0.429282 -0.684847j, 0.476103 -1.6766j, -0.0966428-2.56387j],
          [-0.0966428-2.56387j, -0.980797-3.10127j, -1.86173-3.29645j, -2.52852-3.2912j, -2.879-3.24149j, -2.879-3.24149j, -2.52852-3.2912j, -1.86173-3.29645j, -0.980797-3.10127j, -0.0966428-2.56387j, 0.476103 -1.6766j, 0.429282 -0.684847j, -0.311666-0.0990299j, -1.34711-0.482235j, -1.77252-2.01069j, -0.558095-4.03342j, 2.73285 -5.00781j, 7.34782 -2.90897j, 11.454 +5.15144j, 13.1017 +13.3253j, 11.454 +5.15144j, 7.34782 -2.90897j, 2.73285 -5.00781j, -0.558095-4.03342j, -1.77252-2.01069j, -1.34711-0.482235j, -0.311666-0.0990299j, 0.429282 -0.684847j, 0.476103 -1.6766j],
          [0.476103 -1.6766j, -0.0966428-2.56387j, -0.980797-3.10127j, -1.86173-3.29645j, -2.52852-3.2912j, -2.879-3.24149j, -2.879-3.24149j, -2.52852-3.2912j, -1.86173-3.29645j, -0.980797-3.10127j, -0.0966428-2.56387j, 0.476103 -1.6766j, 0.429282 -0.684847j, -0.311666-0.0990299j, -1.34711-0.482235j, -1.77252-2.01069j, -0.558095-4.03342j, 2.73285 -5.00781j, 7.34782 -2.90897j, 11.454 +5.15144j, 13.1017 +13.3253j, 11.454 +5.15144j, 7.34782 -2.90897j, 2.73285 -5.00781j, -0.558095-4.03342j, -1.77252-2.01069j, -1.34711-0.482235j, -0.311666-0.0990299j, 0.429282 -0.684847j],
          [0.429282 -0.684847j, 0.476103 -1.6766j, -0.0966428-2.56387j, -0.980797-3.10127j, -1.86173-3.29645j, -2.52852-3.2912j, -2.879-3.24149j, -2.879-3.24149j, -2.52852-3.2912j, -1.86173-3.29645j, -0.980797-3.10127j, -0.0966428-2.56387j, 0.476103 -1.6766j, 0.429282 -0.684847j, -0.311666-0.0990299j, -1.34711-0.482235j, -1.77252-2.01069j, -0.558095-4.03342j, 2.73285 -5.00781j, 7.34782 -2.90897j, 11.454 +5.15144j, 13.1017 +13.3253j, 11.454 +5.15144j, 7.34782 -2.90897j, 2.73285 -5.00781j, -0.558095-4.03342j, -1.77252-2.01069j, -1.34711-0.482235j, -0.311666-0.0990299j],
          [-0.311666-0.0990299j, 0.429282 -0.684847j, 0.476103 -1.6766j, -0.0966428-2.56387j, -0.980797-3.10127j, -1.86173-3.29645j, -2.52852-3.2912j, -2.879-3.24149j, -2.879-3.24149j, -2.52852-3.2912j, -1.86173-3.29645j, -0.980797-3.10127j, -0.0966428-2.56387j, 0.476103 -1.6766j, 0.429282 -0.684847j, -0.311666-0.0990299j, -1.34711-0.482235j, -1.77252-2.01069j, -0.558095-4.03342j, 2.73285 -5.00781j, 7.34782 -2.90897j, 11.454 +5.15144j, 13.1017 +13.3253j, 11.454 +5.15144j, 7.34782 -2.90897j, 2.73285 -5.00781j, -0.558095-4.03342j, -1.77252-2.01069j, -1.34711-0.482235j],
          [-1.34711-0.482235j, -0.311666-0.0990299j, 0.429282 -0.684847j, 0.476103 -1.6766j, -0.0966428-2.56387j, -0.980797-3.10127j, -1.86173-3.29645j, -2.52852-3.2912j, -2.879-3.24149j, -2.879-3.24149j, -2.52852-3.2912j, -1.86173-3.29645j, -0.980797-3.10127j, -0.0966428-2.56387j, 0.476103 -1.6766j, 0.429282 -0.684847j, -0.311666-0.0990299j, -1.34711-0.482235j, -1.77252-2.01069j, -0.558095-4.03342j, 2.73285 -5.00781j, 7.34782 -2.90897j, 11.454 +5.15144j, 13.1017 +13.3253j, 11.454 +5.15144j, 7.34782 -2.90897j, 2.73285 -5.00781j, -0.558095-4.03342j, -1.77252-2.01069j],
          [-1.77252-2.01069j, -1.34711-0.482235j, -0.311666-0.0990298j, 0.429282 -0.684847j, 0.476103 -1.6766j, -0.0966427-2.56387j, -0.980797-3.10127j, -1.86173-3.29645j, -2.52852-3.2912j, -2.879-3.24149j, -2.879-3.24149j, -2.52852-3.2912j, -1.86173-3.29645j, -0.980797-3.10127j, -0.0966428-2.56387j, 0.476103 -1.6766j, 0.429282 -0.684847j, -0.311666-0.0990299j, -1.34711-0.482235j, -1.77252-2.01069j, -0.558095-4.03342j, 2.73285 -5.00781j, 7.34782 -2.90897j, 11.454 +5.15144j, 13.1017 +13.3253j, 11.454 +5.15144j, 7.34782 -2.90897j, 2.73285 -5.00781j, -0.558095-4.03342j],
          [-0.558095-4.03342j, -1.77252-2.01069j, -1.34711-0.482235j, -0.311666-0.0990299j, 0.429282 -0.684847j, 0.476103 -1.6766j, -0.0966428-2.56387j, -0.980797-3.10127j, -1.86173-3.29645j, -2.52852-3.2912j, -2.879-3.24149j, -2.879-3.24149j, -2.52852-3.2912j, -1.86173-3.29645j, -0.980797-3.10127j, -0.0966428-2.56387j, 0.476103 -1.6766j, 0.429282 -0.684847j, -0.311666-0.0990298j, -1.34711-0.482235j, -1.77252-2.01069j, -0.558095-4.03342j, 2.73285 -5.00781j, 7.34782 -2.90897j, 11.454 +5.15144j, 13.1017 +13.3253j, 11.454 +5.15144j, 7.34782 -2.90897j, 2.73285 -5.00781j],
          [2.73285 -5.00781j, -0.558095-4.03342j, -1.77252-2.01069j, -1.34711-0.482235j, -0.311666-0.0990299j, 0.429282 -0.684847j, 0.476103 -1.6766j, -0.0966428-2.56387j, -0.980797-3.10127j, -1.86173-3.29645j, -2.52852-3.2912j, -2.879-3.24149j, -2.879-3.24149j, -2.52852-3.2912j, -1.86173-3.29645j, -0.980797-3.10127j, -0.0966428-2.56387j, 0.476103 -1.6766j, 0.429282 -0.684847j, -0.311666-0.0990299j, -1.34711-0.482235j, -1.77252-2.01069j, -0.558095-4.03342j, 2.73285 -5.00781j, 7.34782 -2.90897j, 11.454 +5.15144j, 13.1017 +13.3253j, 11.454 +5.15144j, 7.34782 -2.90897j],
          [7.34782 -2.90897j, 2.73285 -5.00781j, -0.558095-4.03342j, -1.77252-2.01069j, -1.34711-0.482235j, -0.311666-0.0990298j, 0.429282 -0.684847j, 0.476103 -1.6766j, -0.0966428-2.56387j, -0.980797-3.10127j, -1.86173-3.29645j, -2.52852-3.2912j, -2.879-3.24149j, -2.879-3.24149j, -2.52852-3.2912j, -1.86173-3.29645j, -0.980797-3.10127j, -0.0966428-2.56387j, 0.476103 -1.6766j, 0.429282 -0.684847j, -0.311666-0.0990299j, -1.34711-0.482235j, -1.77252-2.01069j, -0.558095-4.03342j, 2.73285 -5.00781j, 7.34782 -2.90897j, 11.454 +5.15144j, 13.1017 +13.3253j, 11.454 +5.15144j],
          [11.454 +5.15144j, 7.34782 -2.90897j, 2.73285 -5.00781j, -0.558095-4.03342j, -1.77252-2.01069j, -1.34711-0.482235j, -0.311666-0.0990299j, 0.429282 -0.684847j, 0.476103 -1.6766j, -0.0966428-2.56387j, -0.980797-3.10127j, -1.86173-3.29645j, -2.52852-3.2912j, -2.879-3.24149j, -2.879-3.24149j, -2.52852-3.2912j, -1.86173-3.29645j, -0.980797-3.10127j, -0.0966428-2.56387j, 0.476103 -1.6766j, 0.429282 -0.684847j, -0.311666-0.0990299j, -1.34711-0.482235j, -1.77252-2.01069j, -0.558095-4.03342j, 2.73285 -5.00781j, 7.34782 -2.90897j, 11.454 +5.15144j, 13.1017 +13.3253j]
    ])


    return Zmn, Zmn_ref

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
            
            # Calculate the current and previous (_M) segments
            # Take care at the m/n == 0 segment
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
            
            # Use the predefined functions to calculate the length of each segment
            dst_m = segment_length(segment_m)/2
            dst_m_M = segment_length(segment_m_M)/2
            dst_n = segment_length(segment_n)/2
            dst_n_M = segment_length(segment_n_M)/2
            
            # Use the predefined funcions to calculate the tangential unit vector
            # components (x & y) of each segment
            tau_m = Tangent_vector_coefficients(segment_m)
            tau_m_M = Tangent_vector_coefficients(segment_m_M)
            tau_n = Tangent_vector_coefficients(segment_n)
            tau_n_M = Tangent_vector_coefficients(segment_n_M)
            
            # Define a function for the green's function used in the integration
            D = lambda eta, ksi, segm, segn: np.linalg.norm(np.subtract(rho(eta, segm), rho(ksi, segn)))
            green = lambda eta, ksi, segm, segn: 1/(2*pi)*kv(0, 1j*k0*D(eta, ksi, segm, segn))
            
            # Split the integration functions up in 8 functions, 4 for each combination
            # of test and basis function, each having a real and imaginary part, as
            # integrate.dblquad is unable to integrate over complex numbers
            integrant_PP_real = lambda eta, ksi: RT_plus(eta)*RT_plus(ksi)*np.real(green(eta, ksi, segment_m_M, segment_n_M))
            integrant_PP_imag = lambda eta, ksi: RT_plus(eta)*RT_plus(ksi)*np.imag(green(eta, ksi, segment_m_M, segment_n_M))
            integrant_MM_real = lambda eta, ksi: RT_min(eta)*RT_min(ksi)*np.real(green(eta, ksi, segment_m, segment_n))
            integrant_MM_imag = lambda eta, ksi: RT_min(eta)*RT_min(ksi)*np.imag(green(eta, ksi, segment_m, segment_n))
            integrant_PM_real = lambda eta, ksi: RT_plus(eta)*RT_min(ksi)*np.real(green(eta, ksi, segment_m_M, segment_n))
            integrant_PM_imag = lambda eta, ksi: RT_plus(eta)*RT_min(ksi)*np.imag(green(eta, ksi, segment_m_M, segment_n))
            integrant_MP_real = lambda eta, ksi: RT_min(eta)*RT_plus(ksi)*np.real(green(eta, ksi, segment_m, segment_n_M))
            integrant_MP_imag = lambda eta, ksi: RT_min(eta)*RT_plus(ksi)*np.imag(green(eta, ksi, segment_m, segment_n_M))
            
            # Integrate the functions only when they are not on the tri-diagonal
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
            
            # Add the prefactors to the defined integrals based on the documentation.
            # Take note that the positive basis/test function contributions correspond
            # to the previous (_M) segment
            I_PP = dst_m_M*dst_n_M*1j*omega*mu0*(tau_m_M[0]*tau_n_M[0] + tau_m_M[1]*tau_n_M[1])*(I_PP_real + 1j*I_PP_imag)
            I_MM = dst_m*dst_n*1j*omega*mu0*(tau_m[0]*tau_n[0] + tau_m[1]*tau_n[1])*(I_MM_real + 1j*I_MM_imag)
            I_PM = dst_m_M*dst_n*1j*omega*mu0*(tau_m_M[0]*tau_n[0] + tau_m_M[1]*tau_n[1])*(I_PM_real + 1j*I_PM_imag)
            I_MP = dst_m*dst_n_M*1j*omega*mu0*(tau_m[0]*tau_n_M[0] + tau_m[1]*tau_n_M[1])*(I_MP_real + 1j*I_MP_imag)
            
            Zmn_left[m,n] = I_PP + I_MM + I_PM + I_MP
            
    return Zmn_left
                
def Zmn_calculator_right(coordinates, wavelength):
    # Exact same methods as Zmn_calculator_left, however, now we use different
    # prefactors and basis/test functions, so the functions will not be explained
    # in detail.
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

def Zmn_adj_calculator(coordinates, wavelength):
    # This function divides the segments into 3 distinct cases: one point, full segment, and no overlap.
    # One point overlap uses Gauss Legendre
    # Full segment overlap uses the self-term definition
    # No overlap uses the general case
    #
    # Take note, this function only calculates the adjacent diagonal for n=m-1,
    # not m=n-1. This does not matter, as the Z-matrix is symmetrical
    M = len(coordinates)-1
    k0 = 2*pi/wavelength
    mu0 = 4*pi*10**-7
    epsilon0 = 8.854187812813e-12
    c = 299792458
    omega = 2*pi*c/wavelength
    gamma_e = 0.577215664901532860606512090082402431042159335
    
    degree = 50
    
    # Predefine the adjacent values
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
        elif m == 1:
            segment_n = Coordinates_to_segment(coordinates, m-1)
            segment_n_M = Coordinates_to_segment(coordinates, -2)
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
        
        # ONE POINT OVERLAP:
        # Use Gauss Legendre to determine weights and coordinates to be used in
        # the integration
        xh, wh = np.polynomial.legendre.leggauss(degree)
        xj = xh
        wj = wh
        integrant_PP, integrant_PP_R = 0, 0
        integrant_MM, integrant_MM_R = 0, 0
        for h in range(degree):
            for j in range(degree):
                integrant_PP    += wh[h]*wj[j]*RT_plus(xh[h])*RT_plus(xj[j])*green(xh[h], xj[j], segment_m_M, segment_n_M)
                integrant_MM    += wh[h]*wj[j]*RT_min(xh[h])*RT_min(xj[j])*green(xh[h], xj[j], segment_m, segment_n)
                integrant_PP_R  += wh[h]*wj[j]*0.25*green(xh[h], xj[j], segment_m_M, segment_n_M)
                integrant_MM_R  += wh[h]*wj[j]*0.25*green(xh[h], xj[j], segment_m, segment_n)
        
        I_PP = dst_m_M*dst_n_M*1j*omega*mu0*(tau_m_M[0]*tau_n_M[0] + tau_m_M[1]*tau_n_M[1])*integrant_PP
        I_MM = dst_m*dst_n*1j*omega*mu0*(tau_m[0]*tau_n[0] + tau_m[1]*tau_n[1])*integrant_PP_R
        
        I_PP_R = dst_m_M*dst_n_M/(1j*omega*epsilon0)*integrant_MM
        I_MM_R = dst_m*dst_n/(1j*omega*epsilon0)*integrant_MM_R
        
        # FULL SEGMENT OVERLAP
        # subscript g is the singular part, and subscript f_g is the numerically solvable part
        I_PM_g = (dst_m_M*dst_n)*1j*omega*mu0*(1/(8*pi)*(5 - 4*np.log(2) - 4*(np.log(1j*k0*dst_n/2) + gamma_e))) # 5 used because PMhas basis functions 0.25(1+eta)(1-xi)=0.25(1+eta+xi-etaxi), so -1
        I_PM_R_g = (dst_m_M*dst_n)*(1j*omega*epsilon0)*(-1/(8*pi)*(6 - 4*np.log(2) - 4*(np.log(1j*k0*dst_n/2) + gamma_e))) # -1/(8pi) used because basis functions 0.5*-0.5=-0.25 
        
        integrant_PM_real = lambda eta, ksi: RT_plus(eta)*RT_min(ksi)*np.real(integrand(eta, ksi, wavelength, segment_m_M, segment_m))
        integrant_PM_imag = lambda eta, ksi: RT_plus(eta)*RT_min(ksi)*np.imag(integrand(eta, ksi, wavelength, segment_m_M, segment_m))
        integrant_PM_R_real = lambda eta, ksi: 0.5*-0.5*np.real(integrand(eta, ksi, wavelength, segment_m_M, segment_m))
        integrant_PM_R_imag = lambda eta, ksi: 0.5*-0.5*np.imag(integrand(eta, ksi, wavelength, segment_m_M, segment_m))
        
        I_PM_real = integrate.dblquad(integrant_PM_real, -1, 1, -1, 1)[0]
        I_PM_imag = integrate.dblquad(integrant_PM_imag, -1, 1, -1, 1)[0]
        I_PM_R_real = integrate.dblquad(integrant_PM_R_real, -1, 1, -1, 1)[0]
        I_PM_R_imag = integrate.dblquad(integrant_PM_R_imag, -1, 1, -1, 1)[0]
        
        I_PM_f_g = dst_m_M*dst_n*1j*omega*mu0*(tau_m_M[0]*tau_n[0] + tau_m_M[1]*tau_n[1])*(I_PM_real + 1j*I_PM_imag)
        I_PM_R_f_g = dst_m_M*dst_n/(1j*omega*epsilon0)*(I_PM_R_real + 1j*I_PM_R_imag)
        
        I_PM, I_PM_R = I_PM_f_g + I_PM_g, I_PM_R_f_g + I_PM_R_g
        
        # NO SEGMENT OVERLAP
        # Uses the exact same functions as the general case
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

def ln_approx(x, N):
    # Calculate the approximation of ln(x) close to x=0
    u = 1 - x
    n = np.arange(1, N + 1)
    terms = (pow(u, n)) / n
    return -np.sum(terms)  # Sum the terms in a highly optimized way

def integrand(eta, ksi, wavelength, segment_m, segment_n):
    # This function is used for the diagonal term, where it returns the integrand f-g used for 
    # singeling out the singularity term
    k0 = 2*pi/wavelength
    gamma_e = 0.577215664901532860606512090082402431042159335
    D = np.linalg.norm(np.subtract(rho(eta, segment_m), rho(ksi, segment_n)))
    dst = segment_length(segment_m)/2
    
    order = 2
    
    if abs(eta - ksi) < 1e-10:
        # eta ~= xi, do taylor series then  
        #log_approx = lambda x: 2*(((x-1)/(x+1)) + (1/3)*((x-1)/(x+1))**3 + (1/5)*((x-1)/(x+1))**5 + (1/7)*((x-1)/(x+1))**7 + (1/9)*((x-1)/(x+1))**9 + (1/11)*((x-1)/(x+1))**11 + (1/13)*((x-1)/(x+1))**13)
        K0_approx = lambda x: -(ln_approx(x/2, order) + gamma_e)*(1 + (x**2)/4) + (x**2)/4
        
        return  1/(2*pi)*(K0_approx(1j*k0*D/2) - (-ln_approx(1j*k0*D/2, order) - gamma_e - ln_approx(abs(eta - ksi), order)))  
    else:
        return 1/(2*pi)*(kv(0, 1j*k0*dst*abs(eta-ksi)) - (-np.log(1j*k0*D/2) - gamma_e - np.log(abs(eta - ksi))))

def Zmn_diag_calculator(coordinates, wavelength):
    M = len(coordinates)-1
    k0 = 2*pi/wavelength
    mu0 = 4*pi*10**-7
    epsilon0 = 8.854187812813e-12
    c = 299792458
    omega = 2*pi*c/wavelength
    gamma_e = 0.577215664901532860606512090082402431042159335 # Euler's constant
    
    # Define the diagonal of the Z matrix as an Mx1 array
    Zdiag = np.zeros((M,1), dtype=np.complex128)
    f_g = np.zeros((M,1), dtype=np.complex128)
    g = np.zeros((M,1), dtype=np.complex128)
    
    for m in range(M):
        print(m)
        segment_m = Coordinates_to_segment(coordinates, m)
        if m == 0:
            segment_m_M = Coordinates_to_segment(coordinates, -2)
        else:
            segment_m_M = Coordinates_to_segment(coordinates, m-1)
            
        dst = segment_length(segment_m)/2
        dst_M = segment_length(segment_m_M)/2
        
        tau = Tangent_vector_coefficients(segment_m)
        tau_M = Tangent_vector_coefficients(segment_m_M)
        
        # Definitions of the 4 different cases as described in the documentation, where
        # +1 or -1 is added inside the brackets depending on the polarity of
        # the eta*ksi term of the basis & test function combination
        intPP = 1/(8*pi)*(7 - 4*np.log(2) - 4*(np.log(1j*k0*dst_M/2) + gamma_e))
        intMM = 1/(8*pi)*(7 - 4*np.log(2) - 4*(np.log(1j*k0*dst/2) + gamma_e))
        intPM = 1/(8*pi)*(5 - 4*np.log(2) - 4*(np.log(1j*k0*dst_M/2) + gamma_e))
        intMP = 1/(8*pi)*(5 - 4*np.log(2) - 4*(np.log(1j*k0*dst/2) + gamma_e))
        
        # Add the 4 terms up using their common prefactor
        g[m] = dst*dst_M*1j*omega*mu0*(intPP + intMM + intPM + intMP)
        
        # Calculate the f-g term of the Z-matrix, by using the general method of calculation, but now by 
        # subtracting the singularity from the K0 function to be able to calculate the value.
        
        # This procedure is similar to the general case, so it will not be explained again
        integrant_PP_real = lambda eta, ksi: RT_plus(eta)*RT_plus(ksi)*np.real(integrand(eta, ksi, wavelength, segment_m_M, segment_m_M))
        integrant_PP_imag = lambda eta, ksi: RT_plus(eta)*RT_plus(ksi)*np.imag(integrand(eta, ksi, wavelength, segment_m_M, segment_m_M))
        integrant_MM_real = lambda eta, ksi: RT_min(eta)*RT_min(ksi)*np.real(integrand(eta, ksi, wavelength, segment_m, segment_m))
        integrant_MM_imag = lambda eta, ksi: RT_min(eta)*RT_min(ksi)*np.imag(integrand(eta, ksi, wavelength, segment_m, segment_m))
        integrant_PM_real = lambda eta, ksi: RT_plus(eta)*RT_min(ksi)*np.real(integrand(eta, ksi, wavelength, segment_m_M, segment_m))
        integrant_PM_imag = lambda eta, ksi: RT_plus(eta)*RT_min(ksi)*np.imag(integrand(eta, ksi, wavelength, segment_m_M, segment_m))
        integrant_MP_real = lambda eta, ksi: RT_min(eta)*RT_plus(ksi)*np.real(integrand(eta, ksi, wavelength, segment_m, segment_m_M))
        integrant_MP_imag = lambda eta, ksi: RT_min(eta)*RT_plus(ksi)*np.imag(integrand(eta, ksi, wavelength, segment_m, segment_m_M))
        
        integrant_PP_R_real = lambda eta, ksi: 0.5*0.5*np.real(integrand(eta, ksi, wavelength, segment_m_M, segment_m_M))
        integrant_PP_R_imag = lambda eta, ksi: 0.5*0.5*np.imag(integrand(eta, ksi, wavelength, segment_m_M, segment_m_M))
        integrant_MM_R_real = lambda eta, ksi: -0.5*-0.5*np.real(integrand(eta, ksi, wavelength, segment_m, segment_m))
        integrant_MM_R_imag = lambda eta, ksi: -0.5*-0.5*np.imag(integrand(eta, ksi, wavelength, segment_m, segment_m))
        integrant_PM_R_real = lambda eta, ksi: 0.5*-0.5*np.real(integrand(eta, ksi, wavelength, segment_m_M, segment_m))
        integrant_PM_R_imag = lambda eta, ksi: 0.5*-0.5*np.imag(integrand(eta, ksi, wavelength, segment_m_M, segment_m))
        integrant_MP_R_real = lambda eta, ksi: -0.5*0.5*np.real(integrand(eta, ksi, wavelength, segment_m, segment_m_M))
        integrant_MP_R_imag = lambda eta, ksi: -0.5*0.5*np.imag(integrand(eta, ksi, wavelength, segment_m, segment_m_M))
        
        I_PP_real = integrate.dblquad(integrant_PP_real, -1, 1, -1, 1)[0]
        I_PP_imag = integrate.dblquad(integrant_PP_imag, -1, 1, -1, 1)[0]
        I_MM_real = integrate.dblquad(integrant_MM_real, -1, 1, -1, 1)[0]
        I_MM_imag = integrate.dblquad(integrant_MM_imag, -1, 1, -1, 1)[0]
        I_PM_real = integrate.dblquad(integrant_PM_real, -1, 1, -1, 1)[0]
        I_PM_imag = integrate.dblquad(integrant_PM_imag, -1, 1, -1, 1)[0]
        I_MP_real = integrate.dblquad(integrant_MP_real, -1, 1, -1, 1)[0]
        I_MP_imag = integrate.dblquad(integrant_MP_imag, -1, 1, -1, 1)[0]
        
        I_PP_R_real = integrate.dblquad(integrant_PP_R_real, -1, 1, -1, 1)[0]
        I_PP_R_imag = integrate.dblquad(integrant_PP_R_imag, -1, 1, -1, 1)[0]
        I_MM_R_real = integrate.dblquad(integrant_MM_R_real, -1, 1, -1, 1)[0]
        I_MM_R_imag = integrate.dblquad(integrant_MM_R_imag, -1, 1, -1, 1)[0]
        I_PM_R_real = integrate.dblquad(integrant_PM_R_real, -1, 1, -1, 1)[0]
        I_PM_R_imag = integrate.dblquad(integrant_PM_R_imag, -1, 1, -1, 1)[0]
        I_MP_R_real = integrate.dblquad(integrant_MP_R_real, -1, 1, -1, 1)[0]
        I_MP_R_imag = integrate.dblquad(integrant_MP_R_imag, -1, 1, -1, 1)[0]
        
        I_PP = dst_M*dst_M*1j*omega*mu0*(tau_M[0]*tau_M[0] + tau_M[1]*tau_M[1])*(I_PP_real + 1j*I_PP_imag)
        I_PP_R = dst_M*dst_M/(1j*omega*epsilon0)*(I_PP_R_real + 1j*I_PP_R_imag)
        
        I_MM = dst*dst*1j*omega*mu0*(tau[0]*tau[0] + tau[1]*tau[1])*(I_MM_real + 1j*I_MM_imag)
        I_MM_R = dst_M*dst_M/(1j*omega*epsilon0)*(I_MM_R_real + 1j*I_MM_R_imag)
        
        I_PM = dst_M*dst*1j*omega*mu0*(tau_M[0]*tau[0] + tau_M[1]*tau[1])*(I_PM_real + 1j*I_PM_imag)
        I_PM_R = dst_M*dst/(1j*omega*epsilon0)*(I_PM_R_real + 1j*I_PM_R_imag)
        
        I_MP = dst*dst_M*1j*omega*mu0*(tau[0]*tau_M[0] + tau[1]*tau_M[1])*(I_MP_real + 1j*I_MP_imag)
        I_MP_R = dst*dst_M/(1j*omega*epsilon0)*(I_MP_R_real + 1j*I_MP_R_imag)
        
        f_g[m] = I_PP + I_PP_R + I_MM + I_MM_R + I_PM + I_PM_R + I_MP + I_MP_R
        
        Zdiag[m] = f_g[m] + g[m]
    
    return Zdiag

def EFIE_TM(coordinates,wavelength,angle):
    # The main algorithm
    Ein_x, Ein_y = DiscritizeEin(coordinates,wavelength,angle)
    Zm, Zm_ref = Zmn_calculator(coordinates,wavelength)

    Jz = np.dot(np.linalg.inv(Zm), Ein_x + Ein_y)
    # Return all variables of interest as a tuple
    return Jz, Ein_x, Ein_y, Zm, Zm_ref

def Etot(Jz, x, y, coordinates, wavelength, angle):
    # Calculate the total field on given coordinates
    N = len(x)
    Etot_x, Etot_y, Ein_x, Ein_y, Esc_x, Esc_y = np.zeros((N,N),dtype=np.complex128),np.zeros((N,N),dtype=np.complex128),np.zeros((N,N),dtype=np.complex128),np.zeros((N,N),dtype=np.complex128),np.zeros((N,N),dtype=np.complex128),np.zeros((N,N),dtype=np.complex128)

    for i in np.arange(N):
        for j in np.arange(N):
            r = np.asarray([x[i],y[j]])
            Ein_x[i,j], Ein_y[i,j] = Efield_in(r, wavelength, angle)
            Esc_x[i,j], Esc_y[i,j] = Escatter(Jz, r, coordinates, wavelength)
            Etot_x[i,j], Etot_y[i,j] = Esc_x[i,j] + Ein_x[i,j], Esc_y[i,j] + Ein_y[i,j]
        print(i)
    return Ein_x, Ein_y, Esc_x, Esc_y, Etot_x, Etot_y

def Escatter(Jz,r,coordinates,wavelength):
    # Calculate the Electric field scattered from the object on given coordinates
    mu0 = 4*pi*10**-7
    epsilon0 = 8.854187812813e-12
    c = 299792458
    omega = 2*pi*c/wavelength
    
    # G can be complex so allocate complex matrix
    G_P_x = np.zeros(len(Jz), dtype=np.complex128)
    G_M_x = np.zeros(len(Jz), dtype=np.complex128)
    G_P_y = np.zeros(len(Jz), dtype=np.complex128)
    G_M_y = np.zeros(len(Jz), dtype=np.complex128)
    GRight_P_x = np.zeros(len(Jz), dtype=np.complex128)
    GRight_M_x = np.zeros(len(Jz), dtype=np.complex128)
    GRight_P_y = np.zeros(len(Jz), dtype=np.complex128)
    GRight_M_y = np.zeros(len(Jz), dtype=np.complex128)
    
    Esc_x, Esc_y = np.zeros(len(Jz), dtype=np.complex128), np.zeros(len(Jz), dtype=np.complex128)
    # Note length Jz = length coordinates-1, so no -1 nessesary here
    for n in np.arange(len(Jz)):
        segment_n = Coordinates_to_segment(coordinates, n)
        if n == 0:
            segment_n_M = Coordinates_to_segment(coordinates, -2)
        else:
            segment_n_M = Coordinates_to_segment(coordinates, n-1)

        # Calculate the real and imaginary part seperately
        GReal_P_x = lambda ksi: (1/2)*(1 + ksi)*np.real(greenEsc(r,ksi,segment_n,wavelength))*(rho(ksi,segment_n)[0])/np.linalg.norm(rho(ksi,segment_n))
        GImag_P_x = lambda ksi: (1/2)*(1 + ksi)*np.imag(greenEsc(r,ksi,segment_n,wavelength))*(rho(ksi,segment_n)[0])/np.linalg.norm(rho(ksi,segment_n))
        GReal_M_x = lambda ksi: (1/2)*(1 - ksi)*np.real(greenEsc(r,ksi,segment_n_M,wavelength))*(rho(ksi,segment_n_M)[0])/np.linalg.norm(rho(ksi,segment_n_M))
        GImag_M_x = lambda ksi: (1/2)*(1 - ksi)*np.imag(greenEsc(r,ksi,segment_n_M,wavelength))*(rho(ksi,segment_n_M)[0])/np.linalg.norm(rho(ksi,segment_n_M))
        GReal_P_y = lambda ksi: (1/2)*(1 + ksi)*np.real(greenEsc(r,ksi,segment_n,wavelength))*(rho(ksi,segment_n)[1])/np.linalg.norm(rho(ksi,segment_n))
        GImag_P_y = lambda ksi: (1/2)*(1 + ksi)*np.imag(greenEsc(r,ksi,segment_n,wavelength))*(rho(ksi,segment_n)[1])/np.linalg.norm(rho(ksi,segment_n))
        GReal_M_y = lambda ksi: (1/2)*(1 - ksi)*np.real(greenEsc(r,ksi,segment_n_M,wavelength))*(rho(ksi,segment_n_M)[1])/np.linalg.norm(rho(ksi,segment_n_M))
        GImag_M_y = lambda ksi: (1/2)*(1 - ksi)*np.imag(greenEsc(r,ksi,segment_n_M,wavelength))*(rho(ksi,segment_n)[1])/np.linalg.norm(rho(ksi,segment_n_M))
        
        GRealRight_P_x = lambda ksi: (1/2)*np.real(greenEsc_R(r,ksi,segment_n,wavelength))*(np.subtract(r,rho(ksi,segment_n))[0])/np.linalg.norm(np.subtract(r, rho(ksi, segment_n)))
        GImagRight_P_x = lambda ksi: (1/2)*np.imag(greenEsc_R(r,ksi,segment_n,wavelength))*(np.subtract(r,rho(ksi,segment_n))[0])/np.linalg.norm(np.subtract(r, rho(ksi, segment_n)))
        GRealRight_M_x = lambda ksi: (-1/2)*np.real(greenEsc_R(r,ksi,segment_n_M,wavelength))*(np.subtract(r,rho(ksi,segment_n_M))[0])/np.linalg.norm(np.subtract(r, rho(ksi, segment_n_M)))
        GImagRight_M_x = lambda ksi: (-1/2)*np.imag(greenEsc_R(r,ksi,segment_n_M,wavelength))*(np.subtract(r,rho(ksi,segment_n_M))[0])/np.linalg.norm(np.subtract(r, rho(ksi, segment_n_M)))
        GRealRight_P_y = lambda ksi: (1/2)*np.real(greenEsc_R(r,ksi,segment_n,wavelength))*(np.subtract(r,rho(ksi,segment_n))[1])/np.linalg.norm(np.subtract(r, rho(ksi, segment_n)))
        GImagRight_P_y = lambda ksi: (1/2)*np.imag(greenEsc_R(r,ksi,segment_n,wavelength))*(np.subtract(r,rho(ksi,segment_n))[1])/np.linalg.norm(np.subtract(r, rho(ksi, segment_n)))
        GRealRight_M_y = lambda ksi: (-1/2)*np.real(greenEsc_R(r,ksi,segment_n_M,wavelength))*(np.subtract(r,rho(ksi,segment_n_M))[1])/np.linalg.norm(np.subtract(r, rho(ksi, segment_n_M)))
        GImagRight_M_y = lambda ksi: (-1/2)*np.imag(greenEsc_R(r,ksi,segment_n_M,wavelength))*(np.subtract(r,rho(ksi,segment_n_M))[1])/np.linalg.norm(np.subtract(r, rho(ksi, segment_n_M)))

        IntReal_P_x = integrate.quad(GReal_P_x, -1, 1)[0]
        IntImag_P_x = integrate.quad(GImag_P_x, -1, 1)[0]
        IntReal_M_x = integrate.quad(GReal_M_x, -1, 1)[0]
        IntImag_M_x = integrate.quad(GImag_M_x, -1, 1)[0]
        IntReal_P_y = integrate.quad(GReal_P_y, -1, 1)[0]
        IntImag_P_y = integrate.quad(GImag_P_y, -1, 1)[0]
        IntReal_M_y = integrate.quad(GReal_M_y, -1, 1)[0]
        IntImag_M_y = integrate.quad(GImag_M_y, -1, 1)[0]
        
        IntRealRight_P_x = integrate.quad(GRealRight_P_x, -1, 1)[0]
        IntImagRight_P_x = integrate.quad(GImagRight_P_x, -1, 1)[0]
        IntRealRight_M_x = integrate.quad(GRealRight_M_x, -1, 1)[0]
        IntImagRight_M_x = integrate.quad(GImagRight_M_x, -1, 1)[0]
        IntRealRight_P_y = integrate.quad(GRealRight_P_y, -1, 1)[0]
        IntImagRight_P_y = integrate.quad(GImagRight_P_y, -1, 1)[0]
        IntRealRight_M_y = integrate.quad(GRealRight_M_y, -1, 1)[0]
        IntImagRight_M_y = integrate.quad(GImagRight_M_y, -1, 1)[0]
          
        # Correct for the basis function used
        dst_n = segment_length(segment_n)/2
        dst_n_M = segment_length(segment_n_M)/2
        
        G_P_x[n] = dst_n*(IntReal_P_x + 1j*IntImag_P_x)
        G_M_x[n] = dst_n_M*(IntReal_M_x + 1j*IntImag_M_x)
        G_P_y[n] = dst_n*(IntReal_P_y + 1j*IntImag_P_y)
        G_M_y[n] = dst_n_M*(IntReal_M_y + 1j*IntImag_M_y)
        
        GRight_P_x[n] = dst_n*(IntRealRight_P_x + 1j*IntImagRight_P_x)
        GRight_M_x[n] = dst_n_M*(IntRealRight_M_x + 1j*IntImagRight_M_x)
        GRight_P_y[n] = dst_n*(IntRealRight_P_y + 1j*IntImagRight_P_y)
        GRight_M_y[n] = dst_n_M*(IntRealRight_M_y + 1j*IntImagRight_M_y)

    # Compute the scattered field for both the x and y components
    # Use np.roll to convert Jn+ into Jn- (Height associated with the negative triangle function)
    Esc_x = 1j*omega*mu0*(np.dot(G_P_x, np.roll(Jz,1)) + np.dot(G_M_x,Jz)) - 1/(1j*omega*epsilon0)*(np.dot(GRight_P_x, np.roll(Jz,1)) + np.dot(GRight_M_x,Jz))
    Esc_y = 1j*omega*mu0*(np.dot(G_P_y, np.roll(Jz,1)) + np.dot(G_M_y,Jz)) - 1/(1j*omega*epsilon0)*(np.dot(GRight_P_y, np.roll(Jz,1)) + np.dot(GRight_M_y,Jz))
    # print(type(Esc_x))
    return Esc_x, Esc_y

def greenEsc(r,ksi,segment,wavelength):
    # Special case, only the basis function applied
    k0 = 2*pi/wavelength
    Pn = rho(ksi, segment)
    
    # Compute Greens function
    G = (1/(2*pi))*kv(0, 1j*k0*np.linalg.norm(np.subtract(r, Pn)))
    
    return G

def greenEsc_R(r,ksi,segment,wavelength):
    # Special case, only the basis function applied for right integral
    k0 = 2*pi/wavelength
    Pn = rho(ksi, segment)
    
    # Compute Greens function for right integral with gradient applied
    G = (-1j*k0/(2*pi))*kv(1, 1j*k0*np.linalg.norm(np.subtract(r, Pn)))
    return G

# UNCOMMENT THIS PART TO TEST FUNCTIONS SEPARATELY!
## INPUTS
# angle = 1.05*pi/2
#c = 299792458
#f = 150*10**6
# mu0 = 4*pi*10**-7
#wavelength = c/f

# # # # Generate a closed circle of radius 1 with 30 data points
#M = 30
#R = 1
#Data = createcircle(M,R)
#Data = np.asarray([[4,1],[3.7,1],[3,1],[2.3,1],[1,1],[0.5,1],[0,1],[-0.5,1],[-1,1],[-2,1],[-3,1],[-3.5,1],[-5,1]])
#M = np.size(Data, 0)

# # # # # ## MAIN
#Zmn_matrix, Zmn_left_matrix, Zmn_right_matrix = Zmn_calculator_2x2matrix_method(Data, wavelength)
#Zmn_left = Zmn_calculator_left(Data, wavelength)
#Zmn_right = Zmn_calculator_right(Data, wavelength)
#Zmn = Zmn_left + Zmn_right
#Zmn, Zmn_ref = Zmn_calculator(Data, wavelength)
# #Z_diag_test = Zmn_diag_calculator(Data, wavelength)
#Zmn_diag_V2 = Zmn_diag_calculator_V2(Data, wavelength)
#Zmn_diag = Zmn_diag_calculator(Data, wavelength)
# # np.fill_diagonal(Zmn, Z_diag_test.flatten())
# # # # # print(Z_diag_test)  
#Zmn_adj = Zmn_adj_calculator(Data, wavelength)
#Zmn_matrix_method = Zmn_calculator_2x2matrix_method(Data, wavelength)
#Zmn_diag = Zmn_diag_calculator(Data, wavelength)

# ERROR CALCULATIONS
#error_Zmatrix_percent = np.abs((Zmn - Zmn_ref)/Zmn_ref) * 100
#error_Zmatrix_absolute = np.abs(Zmn - Zmn_ref)





