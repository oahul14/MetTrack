import numpy as np
import pandas as pd


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

    for var in variables:
        # Remove these as you implement each distribution
        if var == 'radius':
            raise NotImplementedError
        if var == 'angle':
            raise NotImplementedError
        if var == 'strength':
            raise NotImplementedError
        if var == 'velocity':
            raise NotImplementedError
        if var == 'density':
            raise NotImplementedError

    # Implement your ensemble function here

    return pd.DataFrame(columns=variables+['burst_altitude'], index=range(0))
