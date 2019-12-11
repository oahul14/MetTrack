
from collections import OrderedDict
import pandas as pd

from pytest import fixture

# Use pytest fixtures to generate objects we know we'll reuse.
# This makes sure tests run quickly

@fixture(scope='module')
def armageddon():
    """Perform the module import"""
    import armageddon
    return armageddon

@fixture(scope='module')
def planet(armageddon):
    """Return a default planet with a constant atmosphere"""
    return armageddon.Planet(atmos_func='constant')

@fixture(scope='module')
def input_data():
    input_data = {'radius': 1.,
                  'velocity': 1.0e5,
                  'density': 3000.,
                  'strength': 1e32,
                  'angle': 30.0,
                  'init_altitude':100e3,
                  'dt': 0.05,
                  'radians': False 
                 }
    return input_data

@fixture(scope='module')
def result(planet, input_data):
    """Solve a default impact for the default planet"""

    result = planet.solve_atmospheric_entry(**input_data)

    return result

def test_import(armageddon):
    """Check package imports"""
    assert armageddon

def test_planet_signature(armageddon):
    """Check planet accepts specified inputs"""
    inputs = OrderedDict(atmos_func='constant',
                         atmos_filename=None,
                         Cd=1., Ch=0.1, Q=1e7, Cl=1e-3,
                         alpha=0.3, Rp=6371e3,
                         g=9.81, H=8000., rho0=1.2)

    # call by keyword
    planet = armageddon.Planet(**inputs)

    # call by position
    planet = armageddon.Planet(*inputs.values())

def test_attributes(planet):
    """Check planet has specified attributes."""
    for key in ('Cd', 'Ch', 'Q', 'Cl',
                'alpha', 'Rp', 'g', 'H', 'rho0'):
        assert hasattr(planet, key)

def test_solve_atmospheric_entry(result, input_data):
    """Check atmospheric entry solve. 

    Currently only the output type for zero timesteps."""
    
    assert type(result) is pd.DataFrame
    
    for key in ('velocity', 'mass', 'angle', 'altitude',
                'distance', 'radius', 'time'):
        assert key in result.columns

    assert result.velocity.iloc[0] == input_data['velocity']
    assert result.angle.iloc[0] == input_data['angle']
    assert result.altitude.iloc[0] == input_data['init_altitude']
    assert result.distance.iloc[0] == 0.0
    assert result.radius.iloc[0] == input_data['radius']
    assert result.time.iloc[0] == 0.0

def test_calculate_energy(planet, result):

    energy = planet.calculate_energy(result=result)

    print(energy)

    assert type(energy) is pd.DataFrame
    
    for key in ('velocity', 'mass', 'angle', 'altitude',
                'distance', 'radius', 'time', 'dedz'):
        assert key in energy.columns

def test_analyse_outcome(planet, result):

    outcome = planet.analyse_outcome(result)

    assert type(outcome) is dict

def test_ensemble(planet, armageddon):

    fiducial_impact = {'radius': 0.0,
                       'angle': 0.0,
                       'strength': 0.0,
                       'velocity': 0.0,
                       'density': 0.0}
    
    ensemble = armageddon.ensemble.solve_ensemble(planet,
                                                  fiducial_impact,
                                                  variables=[], radians=False,
                                                  rmin=8, rmax=12)

    assert 'burst_altitude' in ensemble.columns


from math import sin, pi
from numpy import exp, linspace
import numpy as np

@fixture(scope='module')
def simpleplanet(armageddon):
    """Return a default planet with simplified assumptions, but with an exponential atmosphere"""
    return armageddon.Planet(atmos_func='exponential', Cd=1., Ch=0.1, Q=1e7, Cl=0, alpha=0, Rp=1e10, g=0, H=8000., rho0=1.2)

@fixture(scope='module')  
def result2(simpleplanet, input_data):
    """Solve a default impact for the default planet"""

    result2 = simpleplanet.solve_atmospheric_entry(**input_data)

    return result2

def test_analytical(simpleplanet, result2, input_data):
    
    def analytical_solution():
        H = simpleplanet.H # atmospheric scale height
        Cd = simpleplanet.Cd # drag coefficient
        r = input_data['radius'] ## radius
        A = pi*input_data['radius']**2 # cross-sectional area 
        P0 = simpleplanet.rho0 # air density at zero altitude
        v0 = input_data['velocity']# initial velocity 
        theta = input_data['angle'] # trajectory angle from horizontal
        z0 = input_data['init_altitude'] # initial z-position
        x0 = 0

        # calc mass:
        rho = 3000 # density
        vol = (4/3)*pi*r**3 # volume
        m = rho*vol # mass

        # timestepping parameters:
        t_0 = 0 # initisal time
        t_f = 17 # final times
        dt = 0.1 # timestep
        state_0 = np.array([v0, z0, x0]) # initial array for solver
        #z = linspace(0,100000,100) # z array
        z= result2["altitude"]
        t = np.arange(t_0, t_f, dt) # time array


        def f(t, state):
            """ meteor equations"""
            v, z, x = state
            f = np.zeros_like(state)
            f[0] = -Cd * P0* np.exp(-z/H) * A * v**2 / (2 * m)
            f[1] = -v*np.sin(theta)
            f[2] = v*np.cos(theta)
            return f

        # analytical solution:
        M = -H*Cd*A*P0/(2*m*sin(theta)) # collecting constant terms
        V = v0*exp(M*exp(-z/H) - M*exp(-z0/H)) # analytical solution
    
        atmo_den = 1.2*np.exp(z/8000)
        return z, V

    z, V = analytical_solution()

    from scipy.integrate import simps
    I1 = simps(result2["velocity"], z) #numerical area
    I2 = simps(V, z) #analytical area
    diff = abs(I2-I1)/I2 * 100
    assert diff < 5
    
@fixture(scope='module')    
def realistic_planet(armageddon):
    """Return a default planet with simpliefied assumptions, but with an exponential atmosphere"""
    return armageddon.Planet(atmos_func='exponential', Cd=2., Ch=0.1, Q=1e7, Cl=1e-3, alpha=0.3, Rp=6371e3, g=9.81, H=8000., rho0=1.2)

def test_chelyabinsk(realistic_planet):
    """Comparing solver output using Chelyabinsk inputs from Collins et al. 2017"""
    frame, out = realistic_planet.impact(99.75, 1.9e4, 3300, 2e6, 20)
    assert out['outcome'] == "Airburst"