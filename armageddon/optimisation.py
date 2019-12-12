from armageddon.solver import Planet
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import itertools

def fit_chelyabinsk(ts, r_init=6, r_end=10, r_step=41, \
                    s_init=1e6, s_end=1e7, s_step=19):
    '''
    Set up start and end parameters to fit Chelyabinsk event
    Parameter
    ----------
    r_init: the start point of radius range
    r_end: the end point of radius range
    r_step: total steps user wants in the radius range
    s_init: the start point of strength range
    s_end: the end point of strength range
    s_step: total steps user wants in the strength range
    ts: the time step for the solver which dictates the speed of fitting

    Return
    -------
    coup: coupled (radius, strength) which returns the best fit
    dis: the distance of two arrays between best fit numerical 
        solution and actual data 
    '''
    ch_df = pd.read_csv("data/ChelyabinskEnergyAltitude.csv")
    strength = np.linspace(s_init, s_end, s_step)
    radius = np.linspace(r_init, r_end, r_step)
    rs = np.array(list(itertools.product(radius, strength)))
    p = np.poly1d(np.polyfit(ch_df.iloc[:, 0].values, \
        ch_df.iloc[:, 1], 10))

    def get_dis(r, s, t):
        '''
        Calculate the distance from numerical solution to the actual data
        Parameters
        -----------
        r: radius in permutations rs
        s: strength in permutations rs

        Return
        -------
        dis
        '''
        result, outcome = Planet().impact(radius=r, velocity=19.2e3,\
                                        density=3300, strength=s, angle=18.3, dt=t)
        alt = result.loc[(result['altitude']<42200) & (result\
                                        ['altitude']>21600)]['altitude'].values/1000
        dedz = result.loc[(result['altitude']<42200) & (result\
                                        ['altitude']>21600)]['dedz'].values
        z_new = np.linspace(21.6, 42.2, len(alt))
        u = np.stack((alt, dedz), axis=-1)[::-1]
        v = np.stack((alt, p(alt)), axis=-1)
        dis = cdist(u, v).min(axis=1).mean()
        print('Distance: %.4f, Radius: %s, Strength: %s' % (dis, r, s))
        return dis
    
    l = []
    for i in rs:
        dis = get_dis(i[0], i[1], ts)
        l.append(dis)
    coup = rs[np.where(l==min(l))[0]]
    return coup, dis 

# coup, dis = fit_chelyabinsk(0.2)
# result, outcome = Planet().impact(rs_min_dis[0][0], \
#                     19.2e3, 3300, rs_min_dis[0][1], 18.3)
# alt = result.loc[(result['altitude']<43000) & (result['altitude']>21000)]['altitude'].values/1000
# dedz = result.loc[(result['altitude']<43000) & (result['altitude']>21000)]['dedz'].values
# plt.plot(alt, dedz, 'b-', label='numerical')
# plt.plot(ch_df.iloc[:, 0], ch_df.iloc[:, 1], 'r--', label='actual')
# plt.ylabel('Energy Per Unit Length ($kt/Km$)')
# plt.xlabel('Altitude (km)')
# plt.legend()