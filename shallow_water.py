'''
Shallow water code that reproduces the results in the Perez and Showman (2013) paper
To run and plot using e.g. 4 processes, with input arguments:
    $ mpiexec -n 4 python3 shallow_water.py --ratio 0.001 --tau_drag 100 --tau_rad 10
    $ mpiexec -n 4 python3 plot_sphere.py snapshots/*.h5
    $ mpiexec -n 4 python3 plot_2d_map.py snapshots/*.h5
'''

import dedalus.public as d3 
import numpy as np
import logging
import time
import os
#args imports 
import argparse
logger = logging.getLogger(__name__)
np.seterr(over='raise', divide='raise', invalid='raise')

#parsing to get input variables 
parser = argparse.ArgumentParser()
parser.add_argument("--ratio", type =float, required =True)
parser.add_argument("--tau_drag", type =float, required =True)
parser.add_argument("--tau_rad", type =float, required =True)
args = parser.parse_args()

# Simulation units 
meter = 1.0 # previous scaling divides by /6.37122e6 
second = 1.0
hour = 3600* second
day = 24*hour

# Params
Nphi =  256 #number of mode for phi, used to be 256
Ntheta = 128 #number of mode for Ntheta, used to be 128
dealias = 3/2 # controls the grid mapping, predetermined already derived

#global variables
a = 8.2e7*meter #radius of HJ
Omega = 3.2e-5 / second #angular velocity of HJ
nu = 1e5 * meter**2 / second / 32**2 #Hyperdiffusivity matched at ell = 32
g = 10 * meter / second**2
H = 400e3 * meter 

#input variables 
ratio = args.ratio #delta heq / H ratio
delta_heq = H*ratio #delta heq, can also be *1
tau_drag = args.tau_drag * day
tau_rad = args.tau_rad * day 
print(f"tau_drag={args.tau_drag}, tau_rad={args.tau_rad}, ratio={ratio}")

#timestep = 600 * second #600 (try 50) INPUT TIMESTEP (now adaptive)
stop_sim_time = 600*day #wall
dtype = np.float64

# Bases 
coords = d3.S2Coordinates('phi', 'theta')
dist = d3.Distributor(coords, dtype=dtype)
basis = d3.SphereBasis(coords, (Nphi, Ntheta), radius=a, dealias=dealias, dtype=dtype)

# Fields (variables!)
u = dist.VectorField(coords, name='u', bases=basis)
h = dist.Field(name='h', bases = basis)
heq = dist.Field(name = 'heq', bases = basis) 
Q = dist.Field(name='Q', bases=basis)
R = dist.VectorField(coords, name='R', bases=basis) 


# Substitutions 
zcross = lambda A: d3.MulCosine(d3.skew(A))

# Initial conditions: zero velocity and h
phi, theta = dist.local_grids(basis)
lat = np.pi / 2 - theta + 0*phi
u['g'][0][:] = 0 #no velocity initially in both dimensions
u['g'][1][:] = 0 
h['g'][:] = 0

# distribution for heq
lat_mask = (lat>= -np.pi/2) & (lat<= np.pi/2) #not nec
day_mask = (((phi >= 0) & (phi <= np.pi/2)) | ((phi >= 3*np.pi/2) & (phi <= 2*np.pi))) & lat_mask
heq['g'][:] = np.where(day_mask, H+delta_heq*np.cos(phi)*np.cos(lat), H)

#heating and advective terms 
Q = (heq-h-H)/tau_rad
R= -u*(Q+abs(Q))/2/(h+H)

# Problem 
problem = d3.IVP([u,h], namespace = locals())
problem.add_equation("dt(u) + nu*lap(lap(u)) + g*grad(h) + 2*Omega*zcross(u) = -u@grad(u) + R - u/tau_drag") #put linear on left
problem.add_equation("dt(h) + H*div(u) + nu*lap(lap(h)) = -div(h*u) + Q") 


# Solver 
solver = problem.build_solver(d3.RK222) #Runge kutta 
solver.stop_sim_time = stop_sim_time

# Create a descriptive folder name based on parameters
if args.ratio>=1:
    ratio_str = str(int(args.ratio)).replace('.', '')
else: 
    ratio_str = str(args.ratio).replace('.', '')

if args.tau_drag > 1000:
    tau_drag_str = 'inf'
elif args.tau_drag >=1:
    tau_drag_str = str(int(args.tau_drag)).replace('.', '')
else:
    tau_drag_str = str(args.tau_drag).replace('.', '')

if args.tau_rad >= 1:
    tau_rad_str = str(int(args.tau_rad)).replace('.', '')
else:
    tau_rad_str = str(args.tau_rad).replace('.', '')

snapshot_dir = f"snapshots/snapshots_{ratio_str}_{tau_drag_str}_{tau_rad_str}/"

#Analysis
snapshots = solver.evaluator.add_file_handler(snapshot_dir, sim_dt = 1*hour, max_writes =100) 
snapshots.add_task(h, name = 'height')
snapshots.add_task(-d3.div(d3.skew(u)), name = 'vorticity') #vertical vorticity
snapshots.add_task(g*(h), name = 'geopotential') 
snapshots.add_task(u, name = 'velocity')
snapshots.add_task(R, name='vertical transport') #check if this works

#adaptive time step (recently added)
CFL_safety = 0.3 # Courant-Friedrichs-Lewy condition, usually < 1
max_timestep = 600 * second #1e-4 chabge to whatever needed 
CFL =d3.CFL(solver, initial_dt = max_timestep, cadence =1, #frequency of updating the time step, ie. 1 is compute time step every one iteration
            safety = CFL_safety, threshold = 0.1, max_change = 2, #max fractional change, default is infinity
            min_change = 0.1, max_dt = max_timestep)
CFL.add_velocity(u)

#Flow properties, use to look at velocity 
flow = d3.GlobalFlowProperty(solver, cadence=10)
ephi = dist.VectorField(coords, bases=basis)
ephi['g'][0] = 1  # phi-direction
flow.add_property((u@ephi)**2, name = 'u_phi')

#variables to automatically stop running the loop when steady state is achieved
tol = 1e-16 #tolerance, can change 
h_prev = None #initialize
u_prev = None

#Main loop
start_time = time.time()
try:
    logger.info('Starting main loop')
    while solver.proceed:

        #timestep (adaptive)
        timestep = CFL.compute_timestep()
        solver.step(timestep)
        
        if (solver.iteration-1)%10==0:

            if h_prev is None and u_prev is None:
                h_prev = h['g'].copy()#store prev values 
                u_prev = u['g'].copy()
                continue

            #relative changes
            h_diff = np.linalg.norm(h['g'] - h_prev) / (np.linalg.norm(h_prev) + 1e-18)
            u_diff = max(np.linalg.norm(u['g'][0] - u_prev[0]) / (np.linalg.norm(u_prev[0]) + 1e-18), 
                         np.linalg.norm(u['g'][1] - u_prev[1]) / (np.linalg.norm(u_prev[1]) + 1e-18))

            #info bout the max u phi 
            max_u_phi = np.sqrt(flow.max('u_phi'))
            logger.info('Iteration=%i, Time=%e, dt=%e, h_diff=%.3e, u_diff=%.3e' %(solver.iteration, solver.sim_time, timestep, h_diff, u_diff)) #put in max(u_phi)=%f, max_u_time

            #breaking if the differences are smaller than the tolerance 
            if h_diff < tol and u_diff < tol:
                logger.info("Converged: fields no longer changing significantly. Stopping simulation.")
                break
            
            # update stored fields
            h_prev[:] = h['g']
            u_prev[:] = u['g']


except: 
    logger.error('Exceptionraised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()

end_time = time.time()
print('The total computational time elapsed is :',end_time - start_time)
