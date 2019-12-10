# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 18:43:02 2019

@author: Alexander Campbell
"""

import tkinter as tk
import matplotlib.pyplot as plt
# input variables:
fields = ('Cd', 'Ch', 'Q', 'Cl', 'alpha', 'Rp', 'g', 'H', 'rho0', 'radius', 'velocity', 'density', 'strength', 'angle')

def makeform(root, fields):
    init_vals = ("1.0", "0.1", "1e7", "1e-3", "0.3", "6371e3", "9.81", "8000", "1.2", "10", "20e3", "3e3", "3e3", "45")
    entries = {}
    for i, field in enumerate(fields,1):
        row = tk.Frame(root)
        lab = tk.Label(row, width=15, text=field+": ", anchor='w')
        ent = tk.Entry(row)
        ent.insert(0, init_vals[i-1])
        row.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        lab.pack(side=tk.LEFT)
        ent.pack(side=tk.RIGHT, expand=tk.YES, fill=tk.X)
        entries[field] = ent
    return entries

def ode_solve(entries):
    """ numerical solver with inputs defined by user in GUI"""
    # input values for solver:
    Cd = float(entries['Cd'].get())
    Ch = float(entries['Ch'].get())
    Q = float(entries['Q'].get())
    Cl = float(entries['Cl'].get())
    alpha = float(entries['alpha'].get())
    Rp = float(entries['Rp'].get())
    g = float(entries['g'].get())
    H = float(entries['H'].get())
    rho0 = float(entries['rho0'].get())
    
    radius = float(entries['radius'].get())
    velocity = float(entries['velocity'].get())
    density = float(entries['density'].get())
    strength = float(entries['strength'].get())
    angle = float(entries['angle'].get())

    # import numerical solver and output:
    import solver as sv
    x = sv.Planet(Cd=Cd, Ch=Ch, Q=Q, Cl=Cl, alpha=alpha, Rp=Rp, rho0=rho0, g=g, H=H)
    result, outcome = x.impact(radius, velocity, density, strength, angle)
    
    # calculate analytical solution:
    
    # print results to terminal:
    print('result')
    #print(result)
    print('  ')
    print('  ')
    print('outcome')
    print(outcome)
    
    plt.figure()
    plt.plot(result["dedz"], result["altitude"])
    #plt.plot((0,3.7e13),(27949, 27949))
    plt.xlabel('dE/dZ')
    plt.ylabel('altitude (m)')
    plt.show()

def plot_an(entries):
    import plot_analytical as pa
    import solver as sv
    
    # input values for solver:
    Cd = float(entries['Cd'].get())
    Ch = float(entries['Ch'].get())
    Q = float(entries['Q'].get())
    Cl = float(entries['Cl'].get())
    alpha = float(entries['alpha'].get())
    Rp = float(entries['Rp'].get())
    g = float(entries['g'].get())
    H = float(entries['H'].get())
    rho0 = float(entries['rho0'].get())  
    radius = float(entries['radius'].get())
    velocity = float(entries['velocity'].get())
    density = float(entries['density'].get())
    strength = float(entries['strength'].get())
    angle = float(entries['angle'].get())
    
    # simplifying assumptions required for comparison:
    Cd=1
    H=8000
    rho0=1.2
    alpha=0
    Cl=0
    Ch=0.1
    Rp=1e10
    
    x = sv.Planet(Cd=Cd, Ch=Ch, Q=Q, Cl=Cl, alpha=alpha, Rp=Rp, rho0=rho0, g=g, H=H)
    result, outcome = x.impact(radius, velocity, density, strength, angle)
    z, V, ivp_sol = pa.solve_analytical()
    plt.plot(z,V,'b-') # analytical
    plt.plot(ivp_sol.y[1,:], ivp_sol.y[0,:],'k.') # ivp solve
    plt.plot(result["altitude"][::10], result["velocity"][::10],'r.') # RK solver

    plt.xlim([0,100000])
    plt.ylim([16500,velocity+(0.01*velocity)])
    plt.title('analytical vs numerical solution')
    plt.xlabel('z (m)')
    plt.ylabel('velocity (m^2/s')
    plt.legend(['Analytical','ivp solve', 'Our solver'])
    plt.show()    

if __name__ == '__main__':
    root = tk.Tk()
    root.title('Asteroid Numerical Solver GUI')
    ents = makeform(root, fields)
    b1 = tk.Button(root, text='Analytical vs numerical',
           command=(lambda e=ents: plot_an(e)))
    b1.pack(side=tk.LEFT, padx=5, pady=5)
    b2 = tk.Button(root, text='ODE solver',
           command=(lambda e=ents: ode_solve(e)))
    b2.pack(side=tk.LEFT, padx=5, pady=5)

    root.mainloop()