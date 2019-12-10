# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 15:54:48 2019

@author: Alexander Campbell
"""
from math import sin, pi
from numpy import exp, linspace
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def solve_analytical(Cd=1.0, r=10, H=8000, v0=20e3, theta=45, rho0=3000):
    ## input variables ##
    #H = 8000 # atmospheric scale height
    #Cd = 1 # drag coefficient
    #r = 10 ## radius
    A = pi*r**2 # cross-sectional area 
    P0 = 1.2 # air density at zero altitude
    #v0 = 20e3 # initial velocity 
    #theta = 45 # trajectory angle from horizontal
    z0 = 100e3 # initial z-position
    x0 = 0
    
    # calc mass:
    rho = rho0 # density
    vol = (4/3)*pi*r**3 # volume
    m = rho*vol # mass
    
    # timestepping parameters:
    t_0 = 0 # initisal time
    t_f = 17 # final times
    dt = 0.1 # timestep
    state_0 = np.array([v0, z0, x0]) # initial array for solver
    z = linspace(0,100000,100) # z array
    t = np.arange(t_0, t_f, dt) # time array
    
    
    def f(t, state):
        v, z, x = state
        f = np.zeros_like(state)
        f[0] = -Cd * P0* np.exp(-z/H) * A * v**2 / (2 * m)
        f[1] = -v*np.sin(theta)
        f[2] = v*np.cos(theta)
        return f
    
    # analytical solution:
    M = -H*Cd*A*P0/(2*m*sin(theta)) # collecting constant terms
    V = v0*exp(M*exp(-z/H) - M*exp(-z0/H)) # analytical solution
    # numerical solution:
    ivp_sol = solve_ivp(f, [t_0, t_f], state_0, t_eval = t)
    
    return z, V, ivp_sol


## plotting ##
#def plotting(z, V, v0, ivp_sol):
#plt.plot(z,V,'b') # analytical
#plt.xlim([0,100000])
#plt.ylim([18000,v0])
#plt.title('analytical vs numerical solution')
#plt.xlabel('z (m)')
#plt.ylabel('velocity (m^2/s')
#plt.plot(ivp_sol.y[1,:], ivp_sol.y[0,:],'r.') # numerical
#plt.legend(['Analytical','Numerical'])