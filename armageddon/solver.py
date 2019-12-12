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
        while (t < t_max) & (u[3] > 0) & (u[2]>0):
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
        outcome = self.analyse_outcome(result)
        return result2, outcome

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
    
    def Lagrange_basis_poly(self, xi, x):
        """Calculate Lagrange basis polynomials.
        
        xi is the x-component of the data
        
        x is the array of x-locations we want the polynomials evaluated at
        
        Returns l, the Lagrange polynomials evaluated at x,
        so l is an array of size (len(xi), len(x))
        """
        # we have N+1 data points, and so the polynomial degree N must be the length of xi minus 1
        N = len(xi) - 1
        # the Lagrange basis polynomials are a product, so let's initialise them with 1
        # (cf. for a summation where we would most likely initialise with zero)
        # we have N+1 of them, and we want their values at locations x, hence size (N+1)xlen(x)
        l = np.ones((N+1, len(x)))
        # we want to iterate over i ranging from zero to N
        for i in range(0, N+1):
            for m in range(0, N+1):
                if (m != i):
                    l[i, :] = l[i, :] * (x - xi[m]) / (xi[i] - xi[m])
        return l

    def Lagrange_interp_poly(self, xi, yi, x):
        """Calculates Lagrange interpolation polynomial from N+1 data points.
        
        (xi, yi) are the N+1 data points (0, 1, ..., N)
        
        x is an array of x-locations the polynomial is evaluated at
        
        Returns L, the Lagrange interpolation polynomial evaluated at x
        """
        # first call our function above to calculate the individual basis functions l
        l = self.Lagrange_basis_poly(xi, x)
        # L is our Lagrange polynomial evaluated at the locations x
        L = np.zeros_like(x)
        for i in range(0, len(xi)):
            L = L + yi[i] * l[i]
        return L




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
        result2 = self.calculate_energy(result)
        dedz_max = np.max(result2["dedz"])
        # the row where maxium dedz is
        row_maxdedz = result2.loc[result2["dedz"] == dedz_max]
        # peak burst altitude
        burst_alt = row_maxdedz.altitude.iloc[0]
        if burst_alt > 5000:
            outcome = self.airburst(result2, row_maxdedz)
        elif (burst_alt >= 0) and (burst_alt <=5000):
            outcome = self.craburst(result2, row_maxdedz)
        elif burst_alt < 0:
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
        row_alt = result.loc[result.altitude > 0]
        # use the row before it to get data for cratering event
        row_alt0 = result.loc[result.index == row_alt.index[-1]]
        
        
        
        # calculate the total energy loss till peak energy loss rate
        # m,v at airburst point and initial condition
        m_burst = result.loc[row_maxdedz.index[0], 'mass']
        v_burst = result.loc[row_maxdedz.index[0], 'velocity']
        m0 = result.loc[0, 'mass']
        v0 = result.loc[0, 'velocity']
        total_loss = np.abs(0.5*(m_burst*v_burst**2-m0*v0**2))/(4.184*10**12)
        
        outcome = {
            "outcome": "Airburst and cratering",
            "burst_peak_dedz": row_maxdedz.dedz.iloc[0],
            "burst_altitude": row_maxdedz.altitude.iloc[0],
            "burst_total_ke_lost" : total_loss,
            "impact_time" : row_alt0.time.iloc[0],
            "impact_mass" :row_alt0.mass.iloc[0],
            "impact_speed" :row_alt0.velocity.iloc[0]
        }
        
#        #find the first row where altitude < 0 
#        row_alt = result.loc[result.altitude < 0]
#        row_lower = result.loc[row_alt.index[0] - 1 <= result.index]
#        row_upper = result.loc[(result.index <= row_alt.index[0] + 10)]
#        row_between = pd.merge(row_lower, row_upper, how='inner')
#        # raw data 
#        alt_i = np.array(row_between.altitude)
#        time_i = np.array(row_between.time)
#        mass_i = np.array(row_between.mass)
#        speed_i = np.array(row_between.velocity)
#        impact_time = float(self.Lagrange_interp_poly(alt_i, time_i, [0]))
#        impact_mass = float(self.Lagrange_interp_poly(alt_i, mass_i, [0]))
#        impact_speed = float(self.Lagrange_interp_poly(alt_i, speed_i, [0]))
#        
#        # calculate the total energy loss till peak energy loss rate
#        # m,v at airburst point and initial condition
#        m_burst = result.loc[row_maxdedz.index[0], 'mass']
#        v_burst = result.loc[row_maxdedz.index[0], 'velocity']
#        m0 = result.loc[0, 'mass']
#        v0 = result.loc[0, 'velocity']
#        total_loss = np.abs(0.5*(m_burst*v_burst**2-m0*v0**2))/(4.184*10**12)
#        
#        outcome = {
#            "outcome": "Airburst and cratering",
#            "burst_peak_dedz": row_maxdedz.dedz.iloc[0],
#            "burst_altitude": row_maxdedz.altitude.iloc[0],
#            "burst_total_ke_lost" : total_loss,
#            "impact_time" : impact_time,
#            "impact_mass" : impact_mass,
#            "impact_speed" : impact_speed
#        }
            
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
        row_alt = result.loc[result.altitude < 0]
        # use the row before it to get data for cratering event
        row_alt0 = result.loc[result.index == row_alt.index[0]-1]
        
        outcome = {
            "outcome": "Cratering",
            "impact_time" : row_alt0.time.iloc[0],
            "impact_mass" :row_alt0.mass.iloc[0],
            "impact_speed" :row_alt0.velocity.iloc[0]
        }
        return outcome
    

#        
#
x = Planet()
result, outcome = x.impact(10, 20e3, 3000, 10e5, 45)

plt.plot(result['altitude'], result['dedz'])
plt.grid()
print(outcome)
plt.show()

def optimisation(z, radius, strength):
    """
        Takes a radius and a strength, and returns the dedz values (calculated by
        the "impact" function inside the "Planet" class)

        Parameters
        ----------
        z: Any
        Unused (used to satisfy syntax of Scipy's optimiser)

        radius:
        Asteroid radius to optimise.

        strength:
        Strength to optimise. 
 
        Returns
        -------
        Result : Array
        Array containing dedz values for a given radius and stength
        """

    x = Planet()
    result, out = x.impact(radius, 19200, 3300, strength, 18.3)
    return   np.array(result["dedz"])
def map_enlarge(a, new_length):
    """
        Map vector of size M into a vector of size N (N>M) using interpolation.
        Works for unevenly spaced entries of the vector. 

        Parameters
        ----------
        a : Array
            Vector to be redimensioned. 
        new_length : Integer
            Length N of the new vector.

        Returns
        -------
        Result : Array
            Redimensioned vector. 
        """

    old_indices = np.arange(0,len(a))
    new_indices = np.linspace(0,len(a)-1,new_length)
    spl = UnivariateSpline(old_indices,a,k=3,s=0)
    return spl(new_indices)
def solve_optimisation():
    """
        Find the best fit to the Chelyabinsk impact data (y = dedz; x = altitude)
        Optimises the function "impact" from class "Planet" with respect to the parameters
        {radius, strength} to fit the observational data. 
        Plots the observatonal data and the fit.
        
        Returns
        -------
        Result : List
            List of length 2 containing the optimal radius and strength
        """

    fil= pd.read_csv('data/ChelyabinskEnergyAltitude.csv', delimiter = ',')
    x_data = np.array(fil["Height (km)"]*1e3)
    y_data = np.array(fil["Energy Per Unit Length (kt Km^-1)"]*1e6)
    x_data_enlarged = map_enlarge(y_data, 1001)

    popt, pcov = curve_fit(optimisation, x_data, x_data_enlarged)
    x = Planet()
    frame, out = x.impact(popt[0], 19200, 3300, popt[1], 18.3) #radius, velocity, density, strength, angle
    plt.figure()
    plt.plot(frame["dedz"], map_enlarge(x_data, 1001))
    plt.plot(y_data, x_data, 'x')
    return popt

#fil= pd.read_csv('data/ChelyabinskEnergyAltitude.csv', delimiter = ',')
#x_data = np.array(fil["Height (km)"]*1e3)
#y_data = np.array(fil["Energy Per Unit Length (kt Km^-1)"]*1e6)
#plt.figure()
#plt.plot(y_data, x_data, 'x')
#plt.figure()
#for radius in [10, 20, 30]:
#    for strength in [1e4, 1e5, 1e6]:
#        result, out = x.impact(radius, 19200, 3300, strength, 18.3)
#        plt.plot(result["dedz"], result["altitude"])
   
#solve_optimisation()

#plt.plot(result['altitude'], result['velocity'])
#plt.grid()
#
#plt.show()
#plt.plot(result['altitude'], result['mass'])
#plt.grid()
#
#plt.show()
#plt.plot(result['altitude'], result['angle'])
##plt.ylim([44,46])
#
#plt.show()
#plt.plot(result['altitude'], result['radius'])
#plt.grid()
#
#plt.show()
#plt.plot(result['altitude'], result['distance'])
#plt.grid()
#
#plt.grid()
#
#plt.show()
#
#######
##1. solve atmos
##2. calc ener
##3. anal out
#
##should be same as solve impact
x = Planet()
result, out= x.impact(1,20e3, 3000, 1e32, 30, init_altitude=100e3, dt=0.05)

plt.plot(result['velocity'], result['altitude'])
plt.grid()
print(outcome)
plt.show()


#x = Planet()
#frame, out = x.impact(1, 1e5, 3000, 1e32, 30, init_altitude=100e3, dt=0.05)
#print(frame.head())
#plt.figure()
#plt.plot(frame["altitude"], frame["velocity"])
