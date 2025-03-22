import numpy as np
from numpy import sin ,cos, pi, sqrt, exp
from scipy import integrate
from scipy.special import kv, psi, factorial

"""UNCOMMENT THIS PART WHEN RUNNING THE CODE WITH THE GEN_FIGURES FILE"""
def createcircle(M,R):
    # generate circle with radius R with M samples
    Arg = np.linspace(0,2*pi,M)
    Data = np.zeros((M,2))
    for i in np.arange(M):
        Data[i,:] = np.asarray([R*cos(Arg[i]),R*sin(Arg[i])])
    return Data

angle=1.05*pi/2
c = 299792458
f = 150*10**6
mu0 = 4*pi*10**-7
wavelength = c/f

M = 30
R = 1
Data = createcircle(M,R)


def DiscritizeEin(coordinates,wavelength,angle):
    # Create an array for the number of segments
    # the code does not close the contour, do so manually
    M = len(coordinates)
    Ein = np.zeros(M,dtype=np.complex128)
    # values of Ein are complex so matrix needs to be able to handle complex values
    # Overwrite each datapoint with the actual value of the Ein field
    for m in np.arange(M):
        # Sample the E field for varying coordinates, based on the testing Function
        # The loop goes over segments between 2 coordinates inside the array
        # Integrate.quad cannot deal with complex numbers (holds as of 22/03/2021)
        segment = Coordinates_to_segment(coordinates,m)
        [nu_x, nu_y] = Normal_vector_coefficients(segment)
        EReal = lambda eta: np.real((nu_x*sin(angle)-nu_y*cos(angle))*Efield_in(rho(eta, segment),wavelength,angle))
        EImag = lambda eta: np.imag((nu_x*sin(angle)-nu_y*cos(angle))*Efield_in(rho(eta, segment),wavelength,angle))
        IntReal = integrate.quad(EReal,-1,1)[0]
        IntImag = integrate.quad(EImag,-1,1)[0]
        # Correct for the test function used, see documentation
        # [0], since integrate.quad outputs result[0] and upper bound of error[1]
        dst = segment_length(segment)
        # multiplication with length of segment due to normalization
        Ein[m] =  dst*(IntReal+ 1j*IntImag)
    return Ein

def closed(coordinates):
    return (abs(coordinates[0] - coordinates[-1]) < 1e-8).all()

def Efield_in(r,wavelength,angle):
    # Calculate the electric field value based on:
    # the x and y position (in r), wavelength and input angle
    mu0 = pi*4e-7
    epsilon0 = 8.854187812813e-12
    H0 = 1
    E0 = H0*sqrt(mu0/epsilon0) # Amplitude of incident wave
    # Electric field is normalized to a magnetic field of 1
    x, y = r[0], r[1]
    # Assuming plane wave in losless material
    k0 = 2*pi/wavelength
    return E0*exp(1j*k0*(cos(angle)*x + sin(angle)*y))

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
    
    # Compute unit normal (-dy, dx) and normalize
    nu = np.array([-dy / length, dx / length])

    # Ensure normal is outward by checking dot product with radial direction
    midpoint = Segment_center(segment)
    radial_direction = midpoint / np.linalg.norm(midpoint)
    
    if np.dot(nu, radial_direction) < 0:
        nu = -nu  # Flip normal if it's pointing inward       
    
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
    M = len(coordinates)
    Zmn = Zmn_left_integral_calculator(coordinates, wavelength) + Zmn_right_integral_calculator(coordinates, wavelength)
    Zmn_diagonal_terms = Zmn_diag_calculator(coordinates, wavelength)
    
    np.fill_diagonal(Zmn, Zmn_diagonal_terms.flatten())
    
    if closed(coordinates):
        # Set the bottom left and top right edge equal to the diagonal, due to the closed nature of the surface
        Zmn[0, M-1] = Zmn_diagonal_terms[0]
        Zmn[M-1, 0] = Zmn_diagonal_terms[0]
        
        # Do the same for the diagonal next to these outer edges, but set them to the adjecent diagonal terms
        Zmn[0, M-2] = Zmn[0, 1]
        Zmn[1, M-1] = Zmn[0, 1]
        Zmn[M-2, 0] = Zmn[0, 1]
        Zmn[M-1, 1] = Zmn[0, 1]
        
    return Zmn

def Zmn_left_integral_calculator(coordinates, wavelength):
    # Define constants
    M = len(coordinates)
    k0 = 2*pi/wavelength
    mu0 = 4*pi*10**-7
    c = 299792458
    omega = 2*pi*c/wavelength
    
    # Predefine matrix sizes
    Zmn_left = np.zeros((M,M), dtype=np.complex128)
    Ipp = np.zeros((M,M), dtype=np.complex128)
    Imm = np.zeros((M,M), dtype=np.complex128)
    Ipm = np.zeros((M,M), dtype=np.complex128)
    Imp = np.zeros((M,M), dtype=np.complex128)
    
    for n in np.arange(M):
        print(n)
        for m in np.arange(M):
            # Calculate the segment for both m & n
            segm = Coordinates_to_segment(coordinates, m)
            segn = Coordinates_to_segment(coordinates, n)
            
            dstm = segment_length(segm)/2
            dstn = segment_length(segn)/2
            
            # Create a function which calculates the norm inside of the Modified Bessel Function
            dstrho = lambda eta, ksi, segm, segn: np.linalg.norm(np.subtract(rho(eta, segm), rho(ksi, segn)))
            
            tauxm, tauym = Tangent_vector_coefficients(segm)
            tauxn, tauyn = Tangent_vector_coefficients(segn)
            
            # Calculate the 4 integrants, both for the real and imaginary part
            integrantPP_real = lambda eta, ksi: 0.5*(1 + eta)*0.5*(1 + ksi)*(1/(2*pi))*np.real(kv(0, 1j*k0*dstrho(eta, ksi, segm, segn)))
            integrantPP_imag = lambda eta, ksi: 0.5*(1 + eta)*0.5*(1 + ksi)*(1/(2*pi))*np.imag(kv(0, 1j*k0*dstrho(eta, ksi, segm, segn)))
            integrantMM_real = lambda eta, ksi: 0.5*(1 - eta)*0.5*(1 - ksi)*(1/(2*pi))*np.real(kv(0, 1j*k0*dstrho(eta, ksi, segm, segn)))
            integrantMM_imag = lambda eta, ksi: 0.5*(1 - eta)*0.5*(1 - ksi)*(1/(2*pi))*np.imag(kv(0, 1j*k0*dstrho(eta, ksi, segm, segn)))
            integrantPM_real = lambda eta, ksi: 0.5*(1 + eta)*0.5*(1 - ksi)*(1/(2*pi))*np.real(kv(0, 1j*k0*dstrho(eta, ksi, segm, segn)))
            integrantPM_imag = lambda eta, ksi: 0.5*(1 + eta)*0.5*(1 - ksi)*(1/(2*pi))*np.imag(kv(0, 1j*k0*dstrho(eta, ksi, segm, segn)))
            integrantMP_real = lambda eta, ksi: 0.5*(1 - eta)*0.5*(1 + ksi)*(1/(2*pi))*np.real(kv(0, 1j*k0*dstrho(eta, ksi, segm, segn)))
            integrantMP_imag = lambda eta, ksi: 0.5*(1 - eta)*0.5*(1 + ksi)*(1/(2*pi))*np.imag(kv(0, 1j*k0*dstrho(eta, ksi, segm, segn)))
            
            # Remove diagonal terms from the calculations
            if closed(coordinates):
                if(m == n or (m == 0 and n == M-1) or (m == M-1 and n == 0)):
                    Zmn_left[m,n] = 0;
                else:
                    intPP_real = integrate.dblquad(integrantPP_real, -1, 1, -1, 1)[0]
                    intPP_imag = integrate.dblquad(integrantPP_imag, -1, 1, -1, 1)[0]
                    intMM_real = integrate.dblquad(integrantMM_real, -1, 1, -1, 1)[0]
                    intMM_imag = integrate.dblquad(integrantMM_imag, -1, 1, -1, 1)[0]
                    intPM_real = integrate.dblquad(integrantPM_real, -1, 1, -1, 1)[0]
                    intPM_imag = integrate.dblquad(integrantPM_imag, -1, 1, -1, 1)[0]
                    intMP_real = integrate.dblquad(integrantMP_real, -1, 1, -1, 1)[0]
                    intMP_imag = integrate.dblquad(integrantMP_imag, -1, 1, -1, 1)[0]
                    
                    Ipp[m,n] = intPP_real + 1j*intPP_imag
                    Imm[m,n] = intMM_real + 1j*intMM_imag
                    Ipm[m,n] = intPM_real + 1j*intPM_imag
                    Imp[m,n] = intMP_real + 1j*intMP_imag
                    Zmn_left[m,n] = dstm*dstn*1j*omega*mu0*(tauxm*tauxn + tauym*tauyn)*(Ipp[m,n] + Imm[m,n] + Ipm[m,n] + Imp[m,n])
            else:
                if m == n:
                    Zmn_left[m,n] = 0;
                else:
                    intPP_real = integrate.dblquad(integrantPP_real, -1, 1, -1, 1)[0]
                    intPP_imag = integrate.dblquad(integrantPP_imag, -1, 1, -1, 1)[0]
                    intMM_real = integrate.dblquad(integrantMM_real, -1, 1, -1, 1)[0]
                    intMM_imag = integrate.dblquad(integrantMM_imag, -1, 1, -1, 1)[0]
                    intPM_real = integrate.dblquad(integrantPM_real, -1, 1, -1, 1)[0]
                    intPM_imag = integrate.dblquad(integrantPM_imag, -1, 1, -1, 1)[0]
                    intMP_real = integrate.dblquad(integrantMP_real, -1, 1, -1, 1)[0]
                    intMP_imag = integrate.dblquad(integrantMP_imag, -1, 1, -1, 1)[0]
                    
                    Ipp[m,n] = intPP_real + 1j*intPP_imag
                    Imm[m,n] = intMM_real + 1j*intMM_imag
                    Ipm[m,n] = intPM_real + 1j*intPM_imag
                    Imp[m,n] = intMP_real + 1j*intMP_imag
                    Zmn_left[m,n] = dstm*dstn*1j*omega*mu0*(tauxm*tauxn + tauym*tauyn)*(Ipp[m,n] + Imm[m,n] + Ipm[m,n] + Imp[m,n])
                    
    return Zmn_left

def Zmn_right_integral_calculator(coordinates, wavelength):
    # Same as the left calculator, but different prefactors
    M = len(coordinates)
    k0 = 2*pi/wavelength
    epsilon0 = 8.854187812813e-12
    c = 299792458
    omega = 2*pi*c/wavelength
    
    Zmn_right = np.zeros((M,M), dtype=np.complex128)
    Ipp = np.zeros((M,M), dtype=np.complex128)
    Imm = np.zeros((M,M), dtype=np.complex128)
    Ipm = np.zeros((M,M), dtype=np.complex128)
    Imp = np.zeros((M,M), dtype=np.complex128)
    
    for n in np.arange(M):
        print(n)
        for m in np.arange(M):
            segm = Coordinates_to_segment(coordinates, m)
            segn = Coordinates_to_segment(coordinates, n)
            
            dstm = segment_length(segm)/2
            dstn = segment_length(segn)/2
            
            dstrho = lambda eta, ksi, segm, segn: np.linalg.norm(np.subtract(rho(eta, segm), rho(ksi, segn)))
            
            tauxm, tauym = Tangent_vector_coefficients(segm)
            tauxn, tauyn = Tangent_vector_coefficients(segn)
            
            integrantPP_real = lambda eta, ksi: 0.5*0.5*(1/(2*pi))*np.real(kv(0, 1j*k0*dstrho(eta, ksi, segm, segn)))
            integrantPP_imag = lambda eta, ksi: 0.5*0.5*(1/(2*pi))*np.imag(kv(0, 1j*k0*dstrho(eta, ksi, segm, segn)))
            integrantMM_real = lambda eta, ksi: -0.5*0.5*(1/(2*pi))*np.real(kv(0, 1j*k0*dstrho(eta, ksi, segm, segn)))
            integrantMM_imag = lambda eta, ksi: -0.5*0.5*(1/(2*pi))*np.imag(kv(0, 1j*k0*dstrho(eta, ksi, segm, segn)))
            integrantPM_real = lambda eta, ksi: 0.5*-0.5*(1/(2*pi))*np.real(kv(0, 1j*k0*dstrho(eta, ksi, segm, segn)))
            integrantPM_imag = lambda eta, ksi: 0.5*-0.5*(1/(2*pi))*np.imag(kv(0, 1j*k0*dstrho(eta, ksi, segm, segn)))
            integrantMP_real = lambda eta, ksi: -0.5*-0.5*(1/(2*pi))*np.real(kv(0, 1j*k0*dstrho(eta, ksi, segm, segn)))
            integrantMP_imag = lambda eta, ksi: -0.5*-0.5*(1/(2*pi))*np.imag(kv(0, 1j*k0*dstrho(eta, ksi, segm, segn)))
            
            if closed(coordinates):
                if(m == n or (m == 0 and n == M-1) or (m == M-1 and n == 0)):
                    Zmn_right[m,n] = 0
                else:
                    intPP_real = integrate.dblquad(integrantPP_real, -1, 1, -1, 1)[0]
                    intPP_imag = integrate.dblquad(integrantPP_imag, -1, 1, -1, 1)[0]
                    intMM_real = integrate.dblquad(integrantMM_real, -1, 1, -1, 1)[0]
                    intMM_imag = integrate.dblquad(integrantMM_imag, -1, 1, -1, 1)[0]
                    intPM_real = integrate.dblquad(integrantPM_real, -1, 1, -1, 1)[0]
                    intPM_imag = integrate.dblquad(integrantPM_imag, -1, 1, -1, 1)[0]
                    intMP_real = integrate.dblquad(integrantMP_real, -1, 1, -1, 1)[0]
                    intMP_imag = integrate.dblquad(integrantMP_imag, -1, 1, -1, 1)[0]
                    
                    Ipp[m,n] = intPP_real + 1j*intPP_imag
                    Imm[m,n] = intMM_real + 1j*intMM_imag
                    Ipm[m,n] = intPM_real + 1j*intPM_imag
                    Imp[m,n] = intMP_real + 1j*intMP_imag
                    Zmn_right[m,n] = dstm*dstn*(tauxm**2 + tauym**2)/(2*pi*1j*omega*epsilon0)*(Ipp[m,n] + Imm[m,n] + Ipm[m,n] + Imp[m,n])
            else:
                if(m == n):
                    Zmn_right[m,n] = 0
                else:
                    intPP_real = integrate.dblquad(integrantPP_real, -1, 1, -1, 1)[0]
                    intPP_imag = integrate.dblquad(integrantPP_imag, -1, 1, -1, 1)[0]
                    intMM_real = integrate.dblquad(integrantMM_real, -1, 1, -1, 1)[0]
                    intMM_imag = integrate.dblquad(integrantMM_imag, -1, 1, -1, 1)[0]
                    intPM_real = integrate.dblquad(integrantPM_real, -1, 1, -1, 1)[0]
                    intPM_imag = integrate.dblquad(integrantPM_imag, -1, 1, -1, 1)[0]
                    intMP_real = integrate.dblquad(integrantMP_real, -1, 1, -1, 1)[0]
                    intMP_imag = integrate.dblquad(integrantMP_imag, -1, 1, -1, 1)[0]
                    
                    Ipp[m,n] = intPP_real + 1j*intPP_imag
                    Imm[m,n] = intMM_real + 1j*intMM_imag
                    Ipm[m,n] = intPM_real + 1j*intPM_imag
                    Imp[m,n] = intMP_real + 1j*intMP_imag
                    Zmn_right[m,n] = dstm*dstn*(tauxm**2 + tauym**2)/(2*pi*1j*omega*epsilon0)*(Ipp[m,n] + Imm[m,n] + Ipm[m,n] + Imp[m,n])
            
    return Zmn_right

def Zmn_diag_calculator(coordinates, wavelength):
    M = len(coordinates)
    mu0 = 4 * pi * 10**-7
    epsilon0 = 8.854187812813e-12
    c = 299792458
    omega = 2*pi*c/wavelength
    
    Zdiag = np.zeros((M,1), dtype=np.complex128)
    for m in np.arange(M):
        segm = Coordinates_to_segment(coordinates, m)
        Is = Self_term_integral(coordinates, wavelength, m) 
        tauxm, tauym = Tangent_vector_coefficients(segm)
        
        # TODO: ADD THE 2ND INTEGRAL ASWELL TO THE DIAGONAL, AS NOW, ONLY THE LEFT INTEGRAL IS USED
        Zdiag[m] = 1j*omega*mu0/(2*pi)*(tauxm*tauxm + tauym*tauym)*Is[0]
    return Zdiag

def Self_term_integral(coordinates, wavelength, m):
    # Define the different elements of the self term integral Is (Is[eta*ksi] and Is[1])
    # using Appendix A3 as a reference for the equations
    k0 = 2*pi/wavelength
    segm = Coordinates_to_segment(coordinates, m);
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
    Ein = DiscritizeEin(coordinates,wavelength,angle)
    Zm = Zmn_calculator(coordinates,wavelength)
    Jz = np.dot(np.linalg.inv(Zm),Ein)
    # Return all variables of interest as a tuple
    return Jz, Ein, Zm

def Etot(Jz, R, coordinates, wavelength, angle):
    # Calculate the total field on given coordinates
    M = len(R)
    Etot = np.zeros(M, dtype=np.complex128)
    for i in np.arange(M):
            r = R[i]
            Esc = Escatter(Jz, r, coordinates, wavelength)
            Ein = Efield_in(r, wavelength, angle)
            Etot[i] = Ein + Esc
    return Etot

def Escatter(Jz,rho,coordinates,wavelength):
    # Calculate the Electric field scattered from the object on given coordinates
    mu0 = 4*pi*10**-7
    c = 299792458
    omega = 2*pi*c/wavelength
    G = np.zeros(len(Jz),dtype=np.complex128) # G can be complex so allocate complex matrix
    # Note length Jz = length coordinates-1
    for n in np.arange(len(Jz)):
        segment = Coordinates_to_segment(coordinates,n)
        GReal = lambda eta: np.real(greenEsc(rho,eta,segment,wavelength))
        GImag = lambda eta: np.imag(greenEsc(rho,eta,segment,wavelength))
        IntReal = integrate.quad(GReal, -1, 1)[0]
        IntImag = integrate.quad(GImag, -1, 1)[0]
        # Correct for the basis function used
        dst = segment_length(segment)
        G[n] = dst*(IntReal + 1j*IntImag)
    return 1j*omega*mu0*np.dot(Jz,G)

def greenEsc(r,ksi,segment,wavelength):
    # Special case, only the basis function applied
    k0 = 2*pi/wavelength
    Pn = rho(ksi,segment)
    return (1/2*pi)*kv(0, 1j*k0*np.linalg.norm(np.subtract(r,Pn)))

#Zmn_left = Zmn_left_integral_calculator(Data, wavelength)
Zmn_right = Zmn_right_integral_calculator(Data, wavelength)
#Zmn_diag = Zmn_diag_calculator(Data, wavelength)
#Zmn = Zmn_calculator(Data, wavelength)

    




