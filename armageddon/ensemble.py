import numpy as np
import pandas as pd
import scipy.special as ssp
import scipy.optimize as sop 

class Dist():

    prob_val = 0.1

    def __init__(self, prob_vals):
        self.prob_vals = prob_vals
    
    def velocity_dist(self,v):
        return ssp.erf(v/(11*np.sqrt(2))) - (v/11)*(np.sqrt(2/np.pi)) * np.exp(-1*(v**2)/(2*(11**2))) - self.prob_val

    def density_dist(self, rho):
        return 0.5*( 1 + ssp.erf((rho-3000)/(1000*np.sqrt(2))) ) - self.prob_val

    def inverse_radius_distribution(self, rmin, rmax):
        return self.prob_vals*(rmax-rmin) + rmin 

    def inverse_strength_distribution(self,ymin=1e3,ymax=10e6):
        return ymin * (10**(self.prob_vals * np.log10(ymax/ymin)))

    def inverse_angle_distribution(self,amin=0,amax=np.pi/2):
        return np.arccos(np.sqrt(self.prob_vals))

    def inverse_velocity_distribution(self,v_guess=(50-11)/2):
        v_array = []
        for prob in self.prob_vals:
            self.prob_val = prob
            v_val = sop.newton_krylov(self.velocity_dist,v_guess)
            v_array.append(v_val)
            v_np = np.array(v_array)
        return v_np

    def inverse_density_distribution(self, rho_guess=(3000)):
        rho_array = []
        for prob in self.prob_vals:
            self.prob_val = prob
            rho_val = sop.diagbroyden(self.density_dist,rho_guess)
            rho_array.append(rho_val)
            rho_np = np.array(rho_array)
        return rho_np
        

def solve_ensemble(
        planet,
        fiducial_impact,
        variables,
        radians=False,
        rmin=8, rmax=12,
        ):
    """
    Run asteroid simulation for a distribution of initial conditions and
    find the burst distribution

    Parameters
    ----------

    planet : object
        The Planet class instance on which to perform the ensemble calculation

    fiducial_impact : dict
        Dictionary of the fiducial values of radius, angle, strength, velocity
        and density

    variables : list
        List of strings of all impact parameters to be varied in the ensemble
        calculation

    rmin : float, optional
        Minimum radius, in m, to use in the ensemble calculation,
        if radius is one of the parameters to be varied.

    rmax : float, optional
        Maximum radius, in m, to use in the ensemble calculation,
        if radius is one of the parameters to be varied.

    Returns
    -------

    ensemble : DataFrame
        DataFrame with columns of any parameters that are varied and the
        airburst altitude
    """

    #convert to degrees
    if radians:
        fiducial_impact['angle'] = fiducial_impact['angle'] * 180/np.pi

    #Number of samples
    N = 100
    prob_distribution = np.random.uniform(0.0,1.0,N)

    distribution = Dist(prob_distribution)

    ensemble_df = pd.DataFrame()

    for var in variables:
        # Remove these as you implement each distribution
        if var == 'radius':
            radius_dist = distribution.inverse_radius_distribution(rmin,rmax)
            fiducial_impact['radius'] = radius_dist
            ensemble_df['radius'] = radius_dist
        if var == 'angle':
            angle_dist = distribution.inverse_angle_distribution()
            angle_dist = angle_dist*180/np.pi     #convert to degrees
            fiducial_impact['angle'] = angle_dist
            ensemble_df['angle'] = angle_dist
        if var == 'strength':
            strength_dist = distribution.inverse_strength_distribution()
            fiducial_impact['strength'] = strength_dist
            ensemble_df['strength'] = strength_dist
        if var == 'velocity':
            velocity_dist = distribution.inverse_velocity_distribution()
            impact_dist = np.sqrt( (11e3)**2 + (velocity_dist*1000)**2 )
            fiducial_impact['velocity'] = impact_dist
            ensemble_df['velocity'] = impact_dist
        if var == 'density':
            density_dist = distribution.inverse_density_distribution()
            fiducial_impact['density'] = density_dist
            ensemble_df['density'] = density_dist

    #check for parameters in fiducial_impact that are not in variables
    const_vals = np.setdiff1d([*fiducial_impact], variables)
    
    for val in const_vals:
        fiducial_impact[val] = [fiducial_impact[val]] * N
        fiducial_impact[val] = np.array(fiducial_impact[val])
     
    burst_altitude = []
    
    for rad,ang,vel,dens,stren in np.stack([fiducial_impact['radius'], fiducial_impact['angle'], 
                                            fiducial_impact['velocity'],fiducial_impact['density'], 
                                            fiducial_impact['strength']], axis = -1):
            _, output = planet.impact(rad,vel,dens,stren,ang)
            if 'burst_altitude' in output:
                burst_altitude.append(output['burst_altitude'])
            else:
                burst_altitude.append(0.0)
    
    ensemble_df['burst_altitude'] = np.array(burst_altitude)
    
    return ensemble_df
