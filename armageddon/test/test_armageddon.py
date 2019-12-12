from collections import OrderedDict
import pandas as pd
import pytest
from pytest import fixture

# Use pytest fixtures to generate objects we know we'll reuse.
# This makes sure tests run quickly

@pytest.mark.xfail
@fixture(scope='module')
def armageddon():
    """Perform the module import"""
    import armageddon
    return armageddon

@pytest.mark.xfail
@fixture(scope='module')
def planet(armageddon):
    """Return a default planet with a constant atmosphere"""
    return armageddon.Planet(atmos_func='constant')

@pytest.mark.xfail
@fixture(scope='module')
def input_data():
    input_data = {'radius': 1.,
                  'velocity': 2.0e4,
                  'density': 3000.,
                  'strength': 1e32,
                  'angle': 30.0,
                  'init_altitude':100e3,
                  'dt': 0.5,
                  'radians': False 
                 }
    return input_data

@pytest.mark.xfail
@fixture(scope='module')
def result(planet, input_data):
    """Solve a default impact for the default planet"""

    result = planet.solve_atmospheric_entry(**input_data)

    return result

@pytest.mark.xfail
def test_import(armageddon):
    """Check package imports"""
    assert armageddon

@pytest.mark.xfail
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

@pytest.mark.xfail
def test_attributes(planet):
    """Check planet has specified attributes."""
    for key in ('Cd', 'Ch', 'Q', 'Cl',
                'alpha', 'Rp', 'g', 'H', 'rho0'):
        assert hasattr(planet, key)

@pytest.mark.xfail
def test_solve_atmospheric_entry(result, input_data):
    """Check atmospheric entry solve.

    Currently only the output type for zero timesteps."""
    
    assert type(result) is pd.DataFrame
    
    for key in ('velocity', 'mass', 'angle', 'altitude',
                'distance', 'radius', 'time'):
        assert key in result.columns

    assert result.velocity.iloc[0] == input_data['velocity']
    assert np.rint(result.angle.iloc[0]) == np.rint(input_data['angle'])
    assert result.altitude.iloc[0] == input_data['init_altitude']
    assert result.distance.iloc[0] == 0.0
    assert result.radius.iloc[0] == input_data['radius']
    assert result.time.iloc[0] == 0.0

@pytest.mark.xfail
def test_calculate_energy(planet, result):

    energy = planet.calculate_energy(result=result)

    print(energy)

    assert type(energy) is pd.DataFrame
    
    for key in ('velocity', 'mass', 'angle', 'altitude',
                'distance', 'radius', 'time', 'dedz'):
        assert key in energy.columns

@pytest.mark.xfail
def test_analyse_outcome(planet, result):

    result = planet.calculate_energy(result)
    outcome = planet.analyse_outcome(result)

    assert type(outcome) is dict

@pytest.mark.xfail
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

@pytest.mark.xfail
def test_solve_atmospheric_entry_mass_sanity(planet, input_data):
    """
    The test is a sanity check for mass over time.
    Mass should be decreasing over time, hence the maximum mass should be the initial mass.
    
    """
    frame = planet.solve_atmospheric_entry(**input_data)
    assert frame['mass'].idxmax() == 0

@pytest.mark.xfail
def test_solve_atmospheric_entry_altitude_sanity(planet, input_data):
    """
    The test is a sanity check for altitude over time. 
    Altitude should be decreasing over time, hence the maximum altitude should be the initial altitude.
    """
    frame = planet.solve_atmospheric_entry(**input_data)
    assert frame['altitude'].idxmax() == 0

@pytest.mark.xfail
def test_solve_atmospheric_entry_distance_sanity(planet, input_data):
    """
    The test is a sanity check for distance over time. 
    Distance should be increasing over time, hence the minimum distance should be found in the first column.
    """
    frame = planet.solve_atmospheric_entry(**input_data)
    assert frame['distance'].idxmin() == 0

@pytest.mark.xfail
def test_solve_atmospheric_entry_angle_check(planet, input_data):
    """
    This tests that the output for angles is in degrees, and angles must be between (inclusively) 0 and 90 degrees.
    """
    frame = planet.solve_atmospheric_entry(**input_data)
    assert frame['angle'].min()>= 0 and frame['angle'].max()<= 90  

@pytest.mark.xfail
@fixture(scope='module')
def simpleplanet(armageddon):
    """
    This returns a simple planet with g=0, Cl=0, alpha=0, and with an exponential atmosphere. 
    This planet will be used to compare the solver against the analytical solution as well as solve_ivp (Scipy's solver).
    """
    return armageddon.Planet(atmos_func='exponential', Cd=1., Ch=0.1, Q=1e7, Cl=0, alpha=0, Rp=1e10, g=0, H=8000., rho0=1.2)

@pytest.mark.xfail
@fixture(scope='module')  
def result2(simpleplanet, input_data):
    """
    This returns a results dataframe on a simple planet (described above) with the initial conditions described by input_data.
    """

    result2 = simpleplanet.solve_atmospheric_entry(**input_data)

    return result2

@pytest.mark.xfail
def test_analytical(simpleplanet, result2, input_data):
    """
    The test that compares the solver to the analytical solution. 
    Using Simpson's rule, The area under the curve of velocity against altitude is compared between the two solutions. 
    If the difference is less than 5%, then the test passes.
    """
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
        rho = input_data['density'] # density
        vol = (4/3)*pi*r**3 # volume
        m = rho*vol # mass

        # timestepping parameters:
        t_0 = 0 # initisal time
        t_f = 400 # final times
        dt = 0.1 # timestep
        state_0 = np.array([v0, z0, x0]) # initial array for solver
        #z = linspace(0,100000,100) # z array
        z= result2["altitude"]
        t = np.arange(t_0, t_f, dt) # time array

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


@pytest.mark.xfail
def test_solveivp(simpleplanet, result2, input_data):
    """
    The test that compares the solver to Scipy's solver. Using Simpson's rule, 
    the area under the curve of velocity against altitude is compared between the two solutions. 
    If the difference is less than 5%, then the test passes.
    """
    def solution():
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
        rho = input_data['density'] # density
        vol = (4/3)*pi*r**3 # volume
        m = rho*vol # mass

        # timestepping parameters:
        t_0 = 0 # initial time
        t_f = 400 # final times
        dt = 0.1 # timestep
        state_0 = np.array([v0, z0, x0]) # initial array for solver
        #z = linspace(0,100000,100) # z array
        z= result2["altitude"]
        t = np.arange(t_0, t_f, dt) # time array

        from scipy.integrate import simps
        I1 = simps(result2["velocity"], z) #numerical area

        def f(t, state):
            v, z, x = state
            f = np.zeros_like(state)
            f[0] = -Cd * P0* np.exp(-z/H) * A * v**2 / (2 * m)
            f[1] = -v*np.sin(theta)
            f[2] = v*np.cos(theta)
            return f

        from scipy.integrate import solve_ivp
        ivp_sol = solve_ivp(f, [t_0, t_f], state_0, t_eval = t)
        I3 = simps(ivp_sol.y[1,:],ivp_sol.y[0,:]) #area from solve_ivp
        diff = abs(I3-I1)/I3 * 100
        assert diff < 5

@pytest.mark.xfail
@fixture(scope='module')    
def realistic_planet(armageddon):
    """
    This returns a more realistic planet (g≠0, Cl≠0, alpha≠0) with an exponential atmosphere and Cd=2 
    as described by Collins et al (2017). 
    This planet will be used to compare results from the solver against real case studies, 
    Chelyabinsk and Tunguska, with data from Collins et al. (2017). 
    """
    return armageddon.Planet(atmos_func='exponential', Cd=2., Ch=0.1, Q=1e7, Cl=1e-3, alpha=0.3, Rp=6371e3, g=9.81, H=8000., rho0=1.2)

@pytest.mark.xfail
def test_chelyabinsk_energy(realistic_planet):
   """
   This test compares solver output of max energy loss per unit height to Chelyabinsk thresholds 
   (between 80 to 110 Kt-1 km) from Collins et al. (2017). 
   The test passes if the solver outputs a result within the range.
   """
   frame = realistic_planet.solve_atmospheric_entry(9.75, 1.9e4, 3300, 2e6, 20)
   max_energy = realistic_planet.calculate_energy(frame)["dedz"].max()
   assert (max_energy >=80) and (max_energy <= 110)

@pytest.mark.xfail
def test_tunguska_energy(realistic_planet):
   """
   This test compares solver output of max energy loss per unit height to Tunguska thresholds 
   (between 800 to 1200 Kt-1 km) from Collins et al. (2017).
    The test passes if the solver outputs a result within the range.
    """
   frame = realistic_planet.solve_atmospheric_entry(25, 2.0e4, 3000, 1e6, 45)
   max_energy = realistic_planet.calculate_energy(frame)["dedz"].max()
   assert (max_energy >=800) and (max_energy <= 1200)

@pytest.mark.xfail
def test_chelyabinsk_outcome(realistic_planet):
    """
    This test asserts the solver's outcome of Chelyabinsk as an airburst event. 
    The inputs for Chelyabinsk come from from Collins et al. (2017).
    """
    frame, out = realistic_planet.impact(9.75, 1.9e4, 3300, 2e6, 20)
    assert out['outcome'] == "Airburst"

@pytest.mark.xfail
def test_tunguska_outcome(realistic_planet):
    """
    This test asserts the solver's outcome of Tunguska as an airburst event. 
    The inputs for Tunguska come from from Collins et al. (2017).
    """
    frame, out = realistic_planet.impact(25, 2.0e4, 3000, 1e6, 45)
    assert out['outcome'] == "Airburst"