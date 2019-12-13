import numpy as np
import pandas as pd
import matplotlib.pyplot as plt #add ploting

class Planet():
    """
    The class called Planet is initialised with constants appropriate
    for the given target planet, including the atmospheric density profile
    and other constants
    """

    def __init__(self, atmos_func='exponential', atmos_filename=None,
                 Cd=1., Ch=0.1, Q=1e7, Cl=1e-3, alpha=0.3, Rp=6371e3,
                 g=9.81, H=8000., rho0=1.2):
        """
        Set up the initial parameters and constants for the target planet
        Parameters
        ----------
        atmos_func : string, optional
            Function which computes atmospheric density, rho, at altitude, z.
            Default is the exponential function ``rho = rho0 exp(-z/H)``.
            Options are ``exponential``, ``tabular``, ``constant`` and ``mars``
        atmos_filename : string, optional
            If ``atmos_func`` = ``'tabular'``, then set the filename of the table
            to be read in here.
        Cd : float, optional
            The drag coefficient
        Ch : float, optional
            The heat transfer coefficient
        Q : float, optional
            The heat of ablation (J/kg)
        Cl : float, optional
            Lift coefficient
        alpha : float, optional
            Dispersion coefficient
        Rp : float, optional
            Planet radius (m)
        rho0 : float, optional
            Air density at zero altitude (kg/m^3)
        g : float, optional
            Surface gravity (m/s^2)
        H : float, optional
            Atmospheric scale height (m)
        Returns
        -------
        None
        """


        # Input constants
        self.Cd = Cd
        self.Ch = Ch
        self.Q = Q
        self.Cl = Cl
        self.alpha = alpha
        self.Rp = Rp
        self.g = g
        self.H = H
        self.rho0 = rho0
        self.flag = 0 #flag variable to select atmosphere
        self.tabular_df = pd.read_csv('data/AltitudeDensityTable.csv', \
                    skiprows=6, delimiter=' ', header=None)

        if atmos_func == 'exponential':
            self.flag = 0
        elif atmos_func == 'tabular':
            self.flag = 1
        elif atmos_func == 'mars':
            self.flag = 2
        elif atmos_func == 'constant':
            self.flag = 3
        else:
            raise NotImplementedError

    def f(self, t, state, atmo_den):
        """ Coupled ODEs when ram pressure is 
        below strength (no changes in radius)
        """
        f = np.zeros_like(state)
        # unpack the state vector
        v, m, theta, z, x, r = state 
        #atmo_den = 1.2*np.exp(-z/8000)
        A = np.pi*r**2
        f[0] = -self.Cd*atmo_den*A*v**2/(2*m) +self.g*np.sin(theta)
        f[1] = -self.Ch*atmo_den*A*v**3/(2*self.Q)
        f[2] = self.g*np.cos(theta)/v - self.Cl*atmo_den*A*v/(2*m) - v*np.cos(theta)/(self.Rp+z)
        f[3] = -v*np.sin(theta)
        f[4] = v*np.cos(theta)/(1+z/self.Rp)
        f[5] = 0
        return f

    def f2(self, t, state, density, atmo_den):
        """ Coupled ODEs when ram pressure is 
        above strength (changes in radius and in area)
        """
        f2 = np.zeros_like(state)
        # unpack the state vector
        v, m, theta, z, x, r = state  
        #atmo_den = 1.2*np.exp(-z/8000)
        A = np.pi*r**2
        f2[0] = -(self.Cd*atmo_den*A*v**2)/(2*m) + self.g*np.sin(theta)
        f2[1] = -(self.Ch*atmo_den*A*v**3)/(2*self.Q)
        f2[2] = (self.g*np.cos(theta))/v - (self.Cl*atmo_den*A*v)/(2*m) - (v*np.cos(theta))/(self.Rp+z)
        f2[3] = -v*np.sin(theta)
        f2[4] = v*np.cos(theta)/(1+z/self.Rp)
        f2[5] = np.sqrt(7/2*self.alpha*atmo_den/density)*v
        return f2

    def RK4(self, u0, t0, t_max, dt, Y, density):
        u = np.array(u0)
        t = np.array(t0)
        u_all = [u0]
        t_all = [t0]
        #t_max = 0.5
        while (t < t_max) & (u[3] > 0) & (u[1] > 0):
            if self.flag == 0:
                atmo_den = 1.2*np.exp(-u[3]/8000)
            elif self.flag == 1:
                atmo_den = self.tabular_df.loc\
                    [np.abs(self.tabular_df[0]-u[3]) <= 5.][1].values[0]
            elif self.flag == 2:
                if u[3] >= 7000.:
                    T = 249.7-0.00222*u[3]
                else:
                    T = 242.1-0.000998*u[3]
                p = 0.699*np.exp(-0.00009*u[3])
                atmo_den = p/(0.1921*T)
            elif self.flag == 3:
                atmo_den = self.rho0
              

            if atmo_den*u[0]**2 > Y:
                k1 = dt*self.f2(t, u, density, atmo_den)
                k2 = dt*self.f2(t + 0.5*dt, u + 0.5*k1, density, atmo_den)
                k3 = dt*self.f2(t + 0.5*dt, u + 0.5*k2, density, atmo_den)
                k4 = dt*self.f2(t + dt, u + k3, density, atmo_den)
            else: #if below threshold => 
                k1 = dt*self.f(t, u, atmo_den)
                k2 = dt*self.f(t + 0.5*dt, u + 0.5*k1, atmo_den)
                k3 = dt*self.f(t + 0.5*dt, u + 0.5*k2, atmo_den)
                k4 = dt*self.f(t + dt, u + k3, atmo_den)

            
            u = (u + (1./6.)*(k1 + 2*k2 + 2*k3 + k4))
            u_all.append(u)
            t = t + dt
            t_all.append(t) 

#            if u[3] <= 0.: # stops if altitude < 0         
#                break
#            if u[2] <= 0.: # stops if mass < 0
#                break
           
        return np.array(u_all), np.array(t_all)

    def impact(self, radius, velocity, density, strength, angle,
               init_altitude=100e3, dt=0.05, radians=False):
        """
        Solve the system of differential equations for a given impact event.
        Also calculates the kinetic energy lost per unit altitude and
        analyses the result to determine the outcome of the impact.
        Parameters
        ----------
        radius : float
            The radius of the asteroid in meters
        velocity : float
            The entery speed of the asteroid in meters/second
        density : float
            The density of the asteroid in kg/m^3
        strength : float
            The strength of the asteroid (i.e., the ram pressure above which
            fragmentation and spreading occurs) in N/m^2 (Pa)
        angle : float
            The initial trajectory angle of the asteroid to the horizontal
            By default, input is in degrees. If 'radians' is set to True, the
            input should be in radians
        init_altitude : float, optional
            Initial altitude in m
        dt : float, optional
            The output timestep, in s
        radians : logical, optional
            Whether angles should be given in degrees or radians. Default=False
            Angles returned in the DataFrame will have the same units as the
            input
        Returns
        -------
        Result : DataFrame
            A pandas DataFrame containing the solution to the system.
            Includes the following columns:
            ``velocity``, ``mass``, ``angle``, ``altitude``,
            ``distance``, ``radius``, ``time``, ``dedz``
        outcome : Dict
            dictionary with details of airburst and/or cratering event.
            For an airburst, this will contain the following keys:
            ``burst_peak_dedz``, ``burst_altitude``, ``burst_total_ke_lost``.
            For a cratering event, this will contain the following keys:
            ``impact_time``, ``impact_mass``, ``impact_speed``.
            All events should also contain an entry with the key ``outcome``,
            which should contain one of the following strings:
            ``Airburst``, ``Cratering`` or ``Airburst and cratering``
        """
        
#        # filtering tests for inputs:
#        assert radius > 0, "Radius must be a positive value"
#        assert velocity > 0, "Velocity must be a positive value"
#        assert density > 0, "Density must be a positive value"
#        assert strength > 0, "Strength must be a positive value"
#        assert 0 < angle <= 90, "Angle must be in range 0 < angle <= 90"

        result = self.solve_atmospheric_entry(radius, velocity, density, strength, angle,
            init_altitude=init_altitude, dt=dt, radians=radians) 
        result2 = self.calculate_energy(result)
        result2 = result2.fillna(0)
        outcome = self.analyse_outcome(result2)
        return result2, outcome

    def get_only_outcome(self, radius, velocity, density, strength, angle,
                         init_altitude=100e3, dt=0.05, radians=False):
        result = self.solve_atmospheric_entry(radius, velocity, density, strength, angle,
            init_altitude=init_altitude, dt=dt, radians=radians)
        result2 = self.calculate_energy(result)
        result2 = result2.fillna(0)
        outcome = self.analyse_outcome(result2)
        return outcome

    def solve_atmospheric_entry(
            self, radius, velocity, density, strength, angle,
            init_altitude=100e3, dt=0.05, radians=False):
        """
        Solve the system of differential equations for a given impact scenario
        Parameters
        ----------
        radius : float
            The radius of the asteroid in meters
        velocity : float
        density : float
            The density of the asteroid in kg/m^3
        strength : float
            The strength of the asteroid (i.e., the ram pressure above which
            fragmentation and spreading occurs) in N/m^2 (Pa)
        angle : float
            The initial trajectory angle of the asteroid to the horizontal
            By default, input is in degrees. If 'radians' is set to True, the
            input should be in radians
        init_altitude : float, optional
            Initial altitude in m
        dt : float, optional
            The output timestep, in s
        radians : logical, optional
            Whether angles should be given in degrees or radians. Default=False
            Angles returned in the DataFrame will have the same units as the
            input
        Returns
        -------
        Result : DataFrame
            A pandas DataFrame containing the solution to the system.
            Includes the following columns:
            ``velocity``, ``mass``, ``angle``, ``altitude``,
            ``distance``, ``radius``, ``time``
        """
        if radians == False:
            angle = angle*np.pi/180 # converting to
        m=density*4/3*np.pi*radius**3
        state0 = np.array([velocity, m, angle, init_altitude,0, radius])
        X = self.RK4(state0,0, 1e10, dt, strength, density)
        result = np.zeros((len(X[0][:, 0])-1, 7))
        result[:, 0:-1] = X[0][:-1, :]
        result[:, -1] = (X[1][:-1])
        result[:,2] = result[:,2]*(180/np.pi) # converting back to degrees for output
        result = pd.DataFrame(result, columns=["velocity", "mass", "angle", "altitude", "distance", "radius", "time"])  
        return result

    def calculate_energy(self, result):
        """
        Function to calculate the kinetic energy lost per unit altitude in
        kilotons TNT per km, for a given solution.
        Parameters
        ----------
        result : DataFrame
            A pandas DataFrame with columns for the velocity, mass, angle,
            altitude, horizontal distance and radius as a function of time
        Returns
        -------
        Result : DataFrame
            Returns the DataFrame with additional column ``dedz`` which is the
            kinetic energy lost per unit altitude
        """

        # Replace these lines with your code to add the dedz column to
        # the result DataFrame
        result = result.copy()
        de = (1/2*result["mass"]*result["velocity"]**2)
        de = de.diff()/(4.184*10**9)
        dz = result["altitude"].diff()
        res = de/dz
        result.insert(len(result.columns),'dedz', res)  
        return result

    def analyse_outcome(self, result):
        """
        Inspect a prefound solution to calculate the impact and airburst stats
        Parameters
        ----------
        result : DataFrame
            pandas DataFrame with velocity, mass, angle, altitude, horizontal
            distance, radius and dedz as a function of time
        Returns
        -------
        outcome : Dict
            dictionary with details of airburst and/or cratering event.
            For an airburst, this will contain the following keys:
            ``burst_peak_dedz``, ``burst_altitude``, ``burst_total_ke_lost``.
            For a cratering event, this will contain the following keys:
            ``impact_time``, ``impact_mass``, ``impact_speed``.
            All events should also contain an entry with the key ``outcome``,
            which should contain one of the following strings:
            ``Airburst``, ``Cratering`` or ``Airburst and cratering``
        """
        # define outcome as a dictionary
        outcome = {}
        # find the maxium dedz and its corresponding burst altitude
        result2 = result.copy()
        dedz_max = result2["dedz"].max()
        # print(dedz_max)
        # print(dedz_max)
        # the row where maxium dedz is
        row_maxdedz = result2.loc[result2["dedz"] == dedz_max]
        # peak burst altitude
        burst_alt = row_maxdedz['altitude'].values
        # print(result2.loc[result2.index[-1], 'dedz'])


        if burst_alt > 5000:
            outcome = self.airburst(result2, row_maxdedz)
        elif result2.loc[result2.index[-1], 'dedz'] != dedz_max:
            outcome = self.craburst(result2, row_maxdedz)
        elif result2.loc[result2.index[-1], 'dedz'] == dedz_max:
            outcome = self.cratering(result2)
        return outcome

    def airburst(self, result, row_maxdedz):
        """
        Inspect a prefound solution to calculate the impact and airburst stats
	    when altitude > 5000 m, i.e. only airburst occurs

        Parameters

        -------
        result-DataFrame
            pandas DataFrame with velocity, mass, angle, altitude, horizontal
            distance, radius and dedz as a function of time
        -------
        row_maxdedz-DataFrame
            pandas DataFrame with velocity, mass, angle, altitude, horizontal
            distance, radius and dedz as a function of time when dedz is biggest

        Returns
        -------
        outcome-Dict
            dictionary with details of airburst this will contain the following keys:
            ``burst_peak_dedz``, ``burst_altitude``, ``burst_total_ke_lost``.
            it also contain an entry with the key ``outcome``,
            which should contain the following strings: ``Airburst``
        """
        # calculate the total energy loss till peak energy loss rate
        # m,v at airburst point and initial condition
        m_burst = result.loc[row_maxdedz.index[0], 'mass']
        v_burst = result.loc[row_maxdedz.index[0], 'velocity']
        m0 = result.loc[0, 'mass']
        v0 = result.loc[0, 'velocity']
        total_loss = np.abs(0.5*(m_burst*v_burst**2-m0*v0**2))/(4.184*10**12)

        outcome = {
            "outcome": "Airburst",
            "burst_peak_dedz": row_maxdedz.dedz.iloc[0],
            "burst_altitude": row_maxdedz.altitude.iloc[0],
            "burst_total_ke_lost" : total_loss
        }
        return outcome

    def craburst(self, result, row_maxdedz):
        """
        Inspect a prefound solution to calculate the impact and airburst stats
	    when 0 m <= altitude <= 5000 m, i.e. both airburst and cratering occur

        Parameters

        -------
        result-DataFrame
            pandas DataFrame with velocity, mass, angle, altitude, horizontal
            distance, radius and dedz as a function of time
        -------
        row_maxdedz-DataFrame
            pandas DataFrame with velocity, mass, angle, altitude, horizontal
            distance, radius and dedz as a function of time when dedz is biggest

        Returns
        -------
        outcome-Dict
            dictionary with details of airburst and cratering event.
            This will contain the following keys:
            ``burst_peak_dedz``, ``burst_altitude``, ``burst_total_ke_lost``
            ``impact_time``, ``impact_mass``, ``impact_speed``.
            IT should also contain an entry with the key ``outcome``,
            which should contain the following strings: ``Airburst and cratering``
        """

        # find the first row where altitude < 0
        # row_alt = result.loc[result.altitude > 0]
        # print(row_alt)
        # use the row before it to get data for cratering event
        row_alt0 = result.loc[result.index[-1]]

        # calculate the total energy loss till peak energy loss rate
        # m,v at airburst point and initial condition
        m_burst = row_maxdedz.mass
        v_burst = row_maxdedz.velocity
        m0 = result.loc[0, 'mass']
        v0 = result.loc[0, 'velocity']
        total_loss = np.abs(0.5*(m_burst*v_burst**2-m0*v0**2))/(4.184*10**12)

        outcome = {
            "outcome": "Airburst and cratering",
            "burst_peak_dedz": row_maxdedz.dedz,
            "burst_altitude": row_maxdedz.altitude,
            "burst_total_ke_lost" : total_loss,
            "impact_time" : row_alt0.time,
            "impact_mass" :row_alt0.mass,
            "impact_speed" :row_alt0.velocity
        }   
        return outcome

    def cratering(self, result):
        """
        Inspect a prefound solution to calculate the impact and airburst stats
	    when altitude < 0 m, i.e. only cratering occurs

        Parameters
        
        -------
        result-DataFrame
            pandas DataFrame with velocity, mass, angle, altitude, horizontal
            distance, radius and dedz as a function of time
        -------
        row_maxdedz-DataFrame
            pandas DataFrame with velocity, mass, angle, altitude, horizontal
            distance, radius and dedz as a function of time when dedz is biggest

        Returns
        -------
        outcome-Dict
            dictionary with details of airburst and/or cratering event.
            This will contain the following keys:
            ``impact_time``, ``impact_mass``, ``impact_speed``.
            It should also contain an entry with the key ``outcome``,
            which should contain the following string: ``Cratering``
        """
        # find the first row where altitude < 0 
        # row_alt = result.loc[result.altitude < 0]
        # use the row before it to get data for cratering event
        print(result)
        row_alt0 = result.loc[result.index[-1]]
        print(row_alt0)
        outcome = {
            "outcome": "Cratering",
            "impact_time" : row_alt0.time,
            "impact_mass" :row_alt0.mass,
            "impact_speed" :row_alt0.velocity
        }
        return outcome

# x = Planet()
# result, outcome = x.impact(120.,20e3, 3000, 1e3, 30, init_altitude=100e3, dt=0.05)
# # print(result['altitude'])
# # print(result)
# # outcome = x.analyse_outcome(result)
# print(outcome)
# plt.plot(result['altitude'], result['dedz'])
# plt.grid()
# plt.show()
