import numpy as np
#import scipy as sp
import matplotlib.pyplot as plt
import EFIE_TM_LIBR_2 as tm
import TMcil
import time
#from csv import writer
from numpy import sin, cos, pi

# def append_list_as_row(file_name, list_of_elem):
#     # Open file in append mode
#     with open(file_name, 'a+', newline='') as write_obj:
#         # Create a writer object from csv module
#         csv_writer = writer(write_obj)
#         # Add contents of list as last row in the csv file
#         csv_writer.writerow(list_of_elem)

def addlinearsamples(coordinates,N):
    # add N samples in between edges of shape defined by
    # coordinates
    x=coordinates[:,0]
    y=coordinates[:,1]
    xn = []
    yn = []
    for i in np.arange(len(coordinates)-1):
            x_tmp = np.linspace(x[i],x[i+1],N,endpoint=False)
            y_tmp = np.linspace(y[i],y[i+1],N,endpoint=False)
            xn = np.append(xn,x_tmp)
            yn = np.append(yn,y_tmp)
    return np.vstack((xn,yn)).T

def createcircle(M,R):
    # generate circle with radius R with M samples
    Arg = np.linspace(0,2*pi,M)
    Data = np.zeros((M,2))
    for i in np.arange(M):
        Data[i,:] = np.asarray([R*cos(Arg[i]),R*sin(Arg[i])])
    return Data

def createparabola(L,M,F):
    x = np.linspace(-L,L,M)
    Data = np.zeros((M,2))
    for i in np.arange(M):
        Data[i,:] = np.asarray([x[i],abs(x[i]**2/(4*F))])
    return Data
##------------------------Actual input---------------------
#angle=1.05*pi/2
angle = 5*pi/4
c = 299792458
f = 150*10**6
mu0 = 4*pi*10**-7
wavelength = c/f

##------------------------Generate shape--------------------
# Use the circle
M = 30
R = 1
Data = createcircle(M,R)

#Data = np.asarray([[3,1],[2,1],[1,1],[0.5,1],[0,1],[-0.5,1],[-1,1],[-2,1],[-3,1]])
#M = np.size(Data, 0)


# # Use a arbitrary shape (here TUe logo)
# shape = np.asarray([[1,12],[1,14],[10,14],[10,7],[11,5],[14,4],[16,5],[17,7],[17,14],[19,14],[19,7],[22,14],[24,14],[22,9],[23.5,10.5],[25,11],[27.5,10.5],[29,9],[29,8],[24,8],[23,6.5],[24,5],[25.5,4.5],[27.5,5],[29,6],[30.5,6],[30,4],[28.5,3],[26,2.5],[24,3],[22,4],[21.5,5],[21,7],[19,3],[17,3],[14,2.5],[11,3],[9.5,5],[9,7],[9,12],[7,12],[7,2],[5,2],[5,12],[1,12]])/5
# # Add samples to satisfy samples per wavelength requirement
# Data = addlinearsamples(shape,4)
# # Convert to left handed array (because we use right to left coordinates)
# Data = np.flip(Data, 0)
# M = len(Data)

#shape = np.asarray([[1,1],[1,2],[2,2],[2,1],[1,1]])
#Data = addlinearsamples(shape,4)

#Data = createparabola(R, M, 1.5)

midpoints = np.zeros((M,2))
normals = np.zeros((M,2))
tangents = np.zeros((M,2))

if tm.closed(Data):
    length = M
else:
    length = M-1
        
for m in range(length): # M-1 for open and M for closed!!
    segment = tm.Coordinates_to_segment(Data, m)  # Ensure it loops correctly
    midpoints[m] = tm.Segment_center(segment)
    normals[m] = tm.Normal_vector(segment)
    tangents[m] = tm.Tangent_vector(segment)
    
plt.gca().set_aspect('equal', adjustable='box')
plt.plot(Data[:,0], Data[:,1], "-o") # Plot the data points of the shape
plt.quiver(midpoints[:,0], midpoints[:,1], normals[:,0], normals[:,1], angles='xy', scale_units='xy', scale=4, color='red')
plt.quiver(midpoints[:,0], midpoints[:,1], tangents[:,0], tangents[:,1], angles='xy', scale_units='xy', scale=4, color='green')

#plt.xlim(-1.5, 1.5)
#plt.ylim(-1.5, 1.5)

#plt.xlim(-3.5, 3.5)
#plt.ylim(0.5, 1.5)

##------------------------Run main code---------------------
start = time.time()
boundary_points = Data
output = tm.EFIE_TM(boundary_points,wavelength,angle)
Jz_x = output[0]
Jz_y = output[1]
Z = output[4]
end = time.time()
time_main = end-start

##-----------------Generate grid to plot E field------------
start = time.time()
N = 80
xmin = -4
xmax = 4
ymin = -1
ymax = 7
x, y = np.linspace(xmin,xmax,N), np.linspace(ymin,ymax,N)

Etot_x, Etot_y, Ein, Ein_x, Ein_y, Esc_x, Esc_y = np.zeros((N,N),dtype=np.complex128),np.zeros((N,N),dtype=np.complex128),np.zeros((N,N),dtype=np.complex128),np.zeros((N,N),dtype=np.complex128),np.zeros((N,N),dtype=np.complex128),np.zeros((N,N),dtype=np.complex128),np.zeros((N,N),dtype=np.complex128)
for i in np.arange(N):
    for j in np.arange(N):
        r = np.asarray([x[i],y[j]])
        Ein_x[i,j], Ein_y[i,j] = tm.Efield_in(r,wavelength,angle)
        Esc_x[i,j], Esc_y[i,j] = tm.Escatter(Jz_x, Jz_y ,r ,boundary_points ,wavelength)
        Etot_x[i,j], Etot_y[i,j] = Esc_x[i,j] + Ein_x[i,j], Esc_y[i,j] + Ein_y[i,j]
end = time.time()
time_grid = end-start

##---------------generate the analytical field------------------------------
gridx,gridy = np.meshgrid(x,y)

simparams = {
    'frequency' : f,
    'epsilon_r' : 10*10**9,
    'radius' : R,
    'modes' : 110,
    'incident_angle' : -angle-pi/2,
    'evaluation_points_x' : gridx,
    'evaluation_points_y' : gridy
}
# uncomment for circle analytical solutions!
Ex,Ey,Hz,Hiz = TMcil.Analytical_2D_TM(simparams)

##-----------------Error calculations over grid-----------------------
# calculate error w.r.t. circular analytical solution
#rel_error_grid = sp.linalg.norm(Eref-Etot,2)/sp.linalg.norm(Etot,2)
#row_contents = [M,N,rel_error_grid,time_main,time_grid]
#append_list_as_row('complexity.csv', row_contents)
##---------------------Obtain figures------------------------
plt.rcParams['figure.figsize'] = [10, 7]
plt.rcParams['figure.dpi'] = 100

plt.figure("$Z_{matrix}$")
plt.imshow(abs(Z),origin='lower',interpolation='none')
plt.title("$|Z|$")
plt.xlabel("x")
plt.ylabel("y")
plt.colorbar()
#plt.savefig("Zmatrix.png")

plt.figure("Ein_x")
plt.imshow(np.real(np.rot90(Ein_x,1)),interpolation='none',extent=[xmin,xmax,ymin,ymax])
plt.title("$|E_{in,x}|$")
plt.xlabel("x")
plt.ylabel("y")
plt.colorbar()

plt.figure("Ein_y")
plt.imshow(np.real(np.rot90(Ein_y,1)),interpolation='none',extent=[xmin,xmax,ymin,ymax])
plt.title("$|E_{in,y}|$")
plt.xlabel("x")
plt.ylabel("y")
plt.colorbar()

plt.figure("Etotx_model")
plt.imshow(abs(np.rot90(Etot_x,1)),interpolation='none',extent=[xmin,xmax,ymin,ymax])
plt.title("$|E_{tot,x}|$")
plt.xlabel("x")
plt.ylabel("y")
plt.colorbar()
plt.show()

plt.figure("Etoty_model")
plt.imshow(abs(np.rot90(Etot_y,1)),interpolation='none',extent=[xmin,xmax,ymin,ymax])
plt.title("$|E_{tot,y}|$")
plt.xlabel("x")
plt.ylabel("y")
plt.colorbar()
plt.show()
# overlay with shape for clearity
#Data2 = addlinearsamples(shape,20)
#plt.scatter(Data2[:,0],Data2[:,1])
#plt.savefig("Etot.png")

# plt.figure("Etot_ref")
# plt.imshow(abs(Hz),origin='lower',interpolation='none',extent=[xmin,xmax,ymin,ymax])
# plt.title("$|E_{ref}|$")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.colorbar()
#plt.savefig("Eref.png")

plt.figure("Ex_ref")
plt.imshow(abs(Ex),origin='lower',interpolation='none',extent=[xmin,xmax,ymin,ymax])
plt.title("$|E_{x}|$")
plt.xlabel("x")
plt.ylabel("y")
plt.colorbar()

plt.figure("Ey_ref")
plt.imshow(abs(Ey),origin='lower',interpolation='none',extent=[xmin,xmax,ymin,ymax])
plt.title("$|E_{y}|$")
plt.xlabel("x")
plt.ylabel("y")
plt.colorbar()

#plt.figure("delta")
#plt.imshow(abs(Eref-Etot),origin='lower',interpolation='none',extent=[xmin,xmax,ymin,ymax])
#plt.title("$|\Delta|$")
#plt.xlabel("x")
#plt.ylabel("y")
#plt.colorbar()
#plt.savefig("DeltaE.png")
