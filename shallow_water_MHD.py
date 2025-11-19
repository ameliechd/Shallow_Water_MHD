'''
Shallow water code that reproduces the results in the Perez and Showman (2013) paper
To run and plot using e.g. 4 processes, with input arguments:
    $ mpiexec -n 4 python3 shallow_water_MHD.py --ratio 0.001 --tau_drag 1 --tau_rad 1
    $ mpiexec -n 4 python3 plot_sphere.py snapMHD/*.h5
    $ mpiexec -n 4 python3 plot_2d_map.py snapMHD/*.h5
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

#Redoing this code but in 2D cartesian!!

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
Nx =  256 
Ny = 255 
dealias = 3/2 # controls the grid mapping, predetermined already derived

#global variables
a = 8.2e7*meter #radius of HJ
Omega = 3.2e-5 / second #angular velocity of HJ
nu = 1e5 * meter**2 / second / 32**2 #Hyperdiffusivity matched at ell = 32
g = 10 * meter / second**2
H = 400e3 * meter 
beta = 2*Omega / a #coriolis parameter, has units already
Leq = (np.sqrt(g*H)/beta)**(1/2) #equatorial Rossby deformation radius
Lm = Leq / 2 #latitudinal decay length of the magnetic field
nu_par = 10**8 * meter**2 / second #viscous diff
eta_par = 3e7 * meter**2 / second #magnetic diff

#input variables 
ratio = args.ratio #delta heq / H ratio
delta_heq = H*ratio #delta heq, can also be *1
tau_drag = args.tau_drag * day
tau_rad = args.tau_rad * day 
V_A = np.sqrt(g*H)/4 #Alfvén speed 
print(f"tau_drag={args.tau_drag}, tau_rad={args.tau_rad}, ratio={ratio}")

stop_sim_time = 600*day #wall
dtype = np.float64

# Bases 
coords = d3.CartesianCoordinates('x', 'y')
x_basis = d3.RealFourier(coords['x'], size =Nx, bounds =(-np.pi *a, np.pi*a), dealias= dealias) #periodic i x
y_basis = d3.Chebyshev(coords['y'], size =Ny, bounds = (-np.pi/2 * a, np.pi/ 2 *a), dealias= dealias) #bounded in y
dist = d3.Distributor(coords, dtype=dtype)
basis = (x_basis, y_basis)
ex, ey = coords.unit_vector_fields(dist)

# Fields (variables!)
u = dist.VectorField(coords, name='u', bases=basis)
h = dist.Field(name='h', bases = basis)
heq = dist.Field(name = 'heq', bases = basis) 
Q = dist.Field(name='Q', bases=basis)
R = dist.VectorField(coords, name='R', bases=basis) 
A = dist.Field(name ='A', bases=basis)
D_nu = dist.VectorField(coords, name ='D_nu', bases=basis)
D_eta = dist.Field(name='D_eta', bases=basis)

# Substitutions 
zcross = lambda A: d3.skew(A) #z cross A
grad = lambda f: d3.grad(f)
lap  = lambda f: d3.Laplacian(f)
grad_perp = lambda f: zcross(d3.grad(f))
dy = lambda A: d3.Differentiate(A, coords['y'])


# tau terms for 1st-order reduction
horiz_bases = (x_basis) # the base that is not Chebyshev, this case your xbasis only
"""
For every derivative (on the Chebyshev basis, y) of a field (vector or scalar)
you will need a corresponding tau term.
i.e is you see grad(u), you need a tau_u1, and you see lap(u), which is 2nd derivative,
    you will need another tau_u2
So, any field A with dy(A) != 0, you will need the tau terms (in this case, A, u, and h)
"""
tau_u1 = dist.VectorField(coords, name='tau_u1', bases=horiz_bases)
tau_u2 = dist.VectorField(coords, name='tau_u2', bases=horiz_bases)
tau_A1 = dist.Field(name='tau_A1', bases=horiz_bases)
tau_A2 = dist.Field(name='tau_A2', bases=horiz_bases)
tau_h1 = dist.Field(name='tau_h1', bases=horiz_bases)
tau_h2 = dist.Field(name='tau_h2', bases=horiz_bases)
"""
For any field A, and unit vector on Chebyshev basis (ey here)
you replace 
grad(A) --> grad(A) + ey*dy(tau_A1)
lap(A)  --> lap(A) + ey*dy^2(tau_A1) - ey*dy(tau_A2) = div(grad_A) - dy(tau_A2)
"""
lift_basis = y_basis.derivative_basis(1)
lift = lambda A, n: d3.Lift(A, lift_basis, n)
Grad_u = d3.grad(u) + ey*lift(tau_u1, -1)
Grad_A = d3.grad(A) + ey*lift(tau_A1, -1)
Grad_h = d3.grad(h) + ey*lift(tau_h1, -1) 
Div_u  = d3.trace(Grad_u)
Lap2_u = d3.Laplacian(d3.div(Grad_u)) - lift(tau_u2, -1)
Lap2_h = d3.Laplacian(d3.div(Grad_h)) - lift(tau_h2, -1)
Div_hu = Grad_h @ u + h * d3.div(u)
Lap_A  = d3.div(Grad_A) - lift(tau_A2, -1)

# Initial conditions: zero velocity and h
x, y = dist.local_grids(x_basis, y_basis)
u['g'][0][:] = 0 #no velocity initially in both dimensions
u['g'][1][:] = 0 
h['g'][:] = 0
A['g'][:] = -np.exp(1/2)*H*V_A*Lm*np.exp(-y**2/(2*Lm**2))#background vector potential (should turn on once steady state is reached)

# distribution for heq
lat_mask = (y/a>= -np.pi/2) & (y/a<= np.pi/2) #not nec
day_mask = (((x/a >= 0) & (x/a <= np.pi/2)) | ((x/a >= 3*np.pi/2) & (x/a <= 2*np.pi))) & lat_mask
heq['g'][:] = np.where(day_mask, H+delta_heq*np.cos(x/a)*np.cos(y/a), H)

#DEFINING heating, advective term, vicous diffusion and magnetic diffusion
Q = (heq-h-H)/tau_rad
R= -u*(Q+abs(Q))/2/(h+H)
#D_nu = ... ignore for now
B = (1/(h+H)) * (-zcross(grad(A))) #because curl(A zhat) = -zhat cross gradA

# Problem 
problem = d3.IVP([u,h,A, tau_u1, tau_u2, tau_A1, tau_A2, tau_h1, tau_h2], namespace = locals()) #

# #equations to solve 
# problem.add_equation("dt(u) + nu*lap(lap(u)) + g*grad(h) + 2*Omega*zcross(u) = -u@grad(u) + B@grad(B) + R - u/tau_drag") 
# problem.add_equation("dt(h) + H*div(u) + nu*lap(lap(h)) = -div(h*u) + Q") 
# problem.add_equation("dt(A) - eta_par*(lap(A))= eta_par*(-(1/(h+H))*grad(h)@grad(A)) -u@grad(A)") #added induction equation

problem.add_equation("dt(u) + nu*Lap2_u + g*Grad_h + 2*Omega*zcross(u) = -u@grad(u) + B@grad(B) + R - u/tau_drag") 
problem.add_equation("dt(h) + H*Div_u + nu*Lap2_h = -Div_hu + Q") 
problem.add_equation("dt(A) - eta_par*(Lap_A) = eta_par*(-(1/(h+H))*grad(h)@grad(A)) -u@grad(A)") 

# boundary conditions
problem.add_equation("ey@(u(y = np.pi/2*a)) = 0")
problem.add_equation("ey@(u(y = - np.pi/2*a)) = 0")

problem.add_equation("-ex@(grad_perp(u)(y = np.pi/2*a)) = 0") #taking d(u_1)/dy
problem.add_equation("-ex@(grad_perp(u)(y = - np.pi*a)) = 0")
# problem.add_equation("ex@(dy(u)(y = np.pi/2*a)) = 0") #taking d(u_1)/dy
# problem.add_equation("ex@(dy(u)(y = - np.pi*a)) = 0")
problem.add_equation("ex@(grad(A)(y = np.pi/2*a)) = 0")
problem.add_equation("ex@(grad(A)(y = - np.pi/2*a)) = 0")

# Solver 
solver = problem.build_solver(d3.RK222) #Runge kutta 
#solver.stop_sim_time = stop_sim_time
#solver.stop_wall_time = 1800  # seconds = 30 minutes

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

snapshot_dir = f"snapMHD/snapMHD_{ratio_str}_{tau_drag_str}_{tau_rad_str}/"

#Analysis
snapshots = solver.evaluator.add_file_handler(snapshot_dir, sim_dt = 1*hour, max_writes =10) #change sim time dt from 1h to 5min
snapshots.add_task(g*(h), name = 'geopotential') 
snapshots.add_task(u, name = 'velocity')
snapshots.add_task(R, name='vertical transport') #check if this works
snapshots.add_task(A, name = 'mag flux')
#snapshots.add_task((1/(h+H)) * (-zcross(grad(A))), name = 'mag field')

#adaptive time step (recently added)
CFL_safety = 0.3 # Courant-Friedrichs-Lewy condition, usually < 1
max_timestep = 600 * second #1e-4 chabge to whatever needed 
CFL =d3.CFL(solver, initial_dt = max_timestep, cadence =1, #frequency of updating the time step, ie. 1 is compute time step every one iteration
            safety = CFL_safety, threshold = 0.1, max_change = 2, #max fractional change, default is infinity
            min_change = 0.1, max_dt = max_timestep)
CFL.add_velocity(u)

#Flow properties, use to look at velocity 
flow = d3.GlobalFlowProperty(solver, cadence=10)
ex = dist.VectorField(coords, bases=basis)
ex['g'][0] = 1  # x-direction
flow.add_property((u@ex)**2, name = 'u_x')

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