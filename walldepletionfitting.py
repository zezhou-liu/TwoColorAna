from scipy.optimize import root
import os
import module
import numpy as np
import matplotlib.pyplot as plt
import module
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D
import time

def distance_generation():
    filepath = "/home/zezhou/McGillResearch/2019Manuscript_Analysis/femsimulation/t4concentration/"
    os.chdir(filepath)

    # Change this to loop to process all data
    tmp = "/home/zezhou/McGillResearch/2019Manuscript_Analysis/femsimulation/t4concentration/sampling/ecc0995_sampling.txt"
    data = np.loadtxt(tmp, skiprows=2)  # 1st column: x axis; 2nd column: y axis. 3rd column: z axis
    # Read the coordinate lattice
    x = data[:, 0]
    y = data[:, 1]
    a, b = module.ellipse_para(ecc=0.995)
    wall = np.zeros_like(x)
    # Create the depletion z profile
    for i in range(len(x)):
        xtmp = x[i]
        ytmp = y[i]
        rdis = xtmp ** 2 / a ** 2 + ytmp ** 2 / b ** 2
        if rdis > 1:
            continue

        def depletion(r):
            # r: dist to boundary; x,y: current coordinate of the point of interest; a,b: ecc's parameter
            return xtmp ** 2*(b - r) ** 2 + ytmp ** 2 * (a - r) ** 2 - (a - r) ** 2*(b - r) ** 2

        sol = root(depletion, x0=1e-5, method='lm')
        wall[i] = sol.x
    np.savetxt('ecc0995_rdist.txt', wall)
    plt.plot(wall, 'r')
    plt.show()
    return
def wdf(x):
    ## Using the following local variables to avoid loading everytime ## ADD THESE TO THE MAIN CODE BLOCK BEFORE USING THIS FUNCTION #####
    # rdistpath = "/home/zezhou/McGillResearch/2019Manuscript_Analysis/femsimulation/t4concentration/rdist/ecc0_rdist.txt"
    # simpath = "/home/zezhou/McGillResearch/2019Manuscript_Analysis/femsimulation/t4concentration/sampling/ecc0_sampling.txt"
    # simdata = np.loadtxt(simpath, skiprows=2)
    # a, b = module.ellipse_para(ecc=0)
    #########################################################################################################
    #######################
    # Initial guess value #
    A = x[0]
    rc = x[1]
    B = x[2]
    #######################
    tgv=1e30
    # Set the outside probability equal to 0. Here I set t4 concentration to 1e30(tgv) to make sure the particle does not penetrate
    simdata[:, 2] = abs(B)*simdata[:, 2]/np.sum(simdata[:, 2])
    for i in range(len(simdata[:,0])):
        xtmp = simdata[i,0]
        ytmp = simdata[i,1]
        rdis = xtmp**2/a**2+ytmp**2/b**2
        if rdis >= 1.0:
            simdata[i,2]=tgv

    rdistdata = np.loadtxt(rdistpath)
    x_sim = np.array(simdata[:, 0]*100-simdata[0, 0]*100+0.04, dtype=int)
    y_sim = np.array(simdata[:, 1]*100-simdata[0, 1]*100-0.01, dtype=int)
    z_mesh = np.zeros([nx+1,ny+1])
    z_mesh[x_sim, y_sim] = A*np.exp(-rdistdata/np.abs(rc)) + simdata[:,2]
    z_mesh[-1,:] = tgv
    z_mesh[:,-1] = tgv
    prob = np.exp(-z_mesh)
    normf = max(np.sum(prob),1e-8)
    prob = np.transpose(prob/normf)
    # plt.imshow(prob)
    # plt.show()
    return prob
def feedthrough(prob, xs, ys, h, x, y):
    # Convert the size of simulation to real data size
    probout = np.zeros_like(h)
    # Edge to center
    xsm, ysm = np.meshgrid(xs, ys)
    for i in range(len(x)-1):
        xl = x[i]  # x left edge
        xr = x[i + 1]  # x right edge
        for j in range(len(y)-1):
            yb = y[j] # y bottom edge
            yt = y[j+1] # y top edge
            maskx = (xsm >= xl)*(xsm<xr)
            masky = (ysm >= yb) * (ysm < yt)
            mask = maskx * masky
            noft = max(np.sum(mask), 1)
            probout[j, i] = np.sum(prob[mask])/noft
    un = max(np.sum(probout),1e-6)
    return probout/un

datapath = "/home/zezhou/McGillResearch/2019Manuscript_Analysis/femsimulation/t4concentration/data/Ecc0h.txt"
xpath = "/home/zezhou/McGillResearch/2019Manuscript_Analysis/femsimulation/t4concentration/data/Ecc0x.txt"
ypath = "/home/zezhou/McGillResearch/2019Manuscript_Analysis/femsimulation/t4concentration/data/Ecc0y.txt"
rdistpath = "/home/zezhou/McGillResearch/2019Manuscript_Analysis/femsimulation/t4concentration/rdist/ecc0_rdist.txt"
simpath = "/home/zezhou/McGillResearch/2019Manuscript_Analysis/femsimulation/t4concentration/sampling/ecc0_sampling.txt"
ecc = 0.0 # Eccentricity
xsample = 1.0	 # simulation xlim
ysample = 1.0		 # simulation ylim
nx = 200 # nofp in x dir.
ny = 200 # nofp in y dir.
sol_ini = [1, 0.1, 1e3] # Initial guess of the solution
wdf_pltflag = 1 # if i want to see the check plt of simulation
scale = 6.25 # 6.25pixel/um

# Real data & mesh
h = np.loadtxt(datapath)
h = np.transpose(h/np.sum(h))
x = np.loadtxt(xpath)/scale # xedge from hist
y = np.loadtxt(ypath)/scale # yedge from

# Simulation data & mesh

simdata = np.loadtxt(simpath, skiprows=2)
a, b = module.ellipse_para(ecc=ecc)

xs = np.linspace(-xsample,xsample,nx+1)
ys = np.linspace(-ysample,ysample,ny+1)

# Loss function definition
def lossfunc(ini):
    # Loss function define by 2-norm integration. Can be modified.
    global wdf_pltflag
    prob = wdf(ini) # Input is initial guess. Sim data loading is manually loaded by changing the filepath in wdf.
    prob = feedthrough(prob, xs, ys, h, x, y)
    # Check plot of discrete sim result and data
    if wdf_pltflag == 1:
        plt.imshow(prob, cmap='inferno')
        plt.show()
        plt.imshow(h, cmap='inferno')
        plt.show()
        wdf_pltflag = 0
    err = np.sum((prob-h)**2)
    return err
# Fitting by scipy.optimize.minimize
def fitting():
    # Initial guess of the solution
    method = 'Nelder-Mead'
    tic = time.clock()
    sol = minimize(lossfunc, x0=sol_ini, method=method)
    toc = time.clock()
    print(sol)
    print(method+"Optimizing time:"+str(toc-tic))
    return

solu = [ 4.05124796e+03,  1.81111892e-02, -2.33623784e+03]
lossfunc(solu)

# prob = wdf(sol_ini)
# prob = feedthrough(prob, xs, ys, h, x, y)
# plt.imshow(prob, cmap='inferno')
# plt.show()
# plt.imshow(h, cmap='inferno')
# plt.show()