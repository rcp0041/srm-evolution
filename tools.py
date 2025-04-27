#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GENOME:
    [N, Ro, Ri, Rp, f, epsilon, theta/2, grain_length, prop_i, material_i]
"""

import srm                             # Solid rocket motor performance library
import toml                            # For reading config file
import numpy as np                     # Numerical operations
from scipy import optimize             # Iterative solver
from scipy.integrate import solve_ivp  # RK45 solver
import matplotlib.pyplot as plt

# Create lists that allow numeric indexing of propellant/material dictionaries
propellants_list = list(srm.materials.propellants)
steel_list = list(srm.materials.steel)
num_propellants = len(propellants_list) - 1
num_steel = len(steel_list) - 1

# test_genome = np.array([11.30847159,6.,4.60885396,1.37349861,0.86655393,0.24527711,0.97914113,24.,2,10])

""" Use this genome for testing """
N = 5
Ro = 4
Ri = 0.8
Rp = 2.2
f = 0.25
epsilon = 0.7
halfTheta = np.deg2rad(60)
L = 24
prop_i = 0
material_i = 7

test_genome = np.array([N,Ro,Ri,Rp,f,epsilon,halfTheta,L,prop_i,material_i])

data = toml.load("./config.toml")
def get_bounds(key_string):
    key_string = key_string + "_bounds"
    return (data[key_string][0],data[key_string][1])

def get_setting(key_string):
    return (data[key_string])

def construct_neutral_motor(genome):
    Ro = genome[1]
    length = genome[7]
    
    N = int(genome[0])
    epsilon = genome[5]
    f = genome[4]
    Rp = genome[3]
    
    epsilonAngle = np.pi*N/epsilon
    H1 = Rp*np.sin(epsilonAngle)
    
    def neutral_Ri(Ri):
        return np.pi/2 + np.pi/N - np.arctan((H1*np.tan(epsilonAngle))/(H1 - Ri*np.tan(epsilonAngle))) - (H1 - Ri*np.tan(epsilonAngle))/(H1*np.tan(epsilonAngle))
    
    Ri = optimize.root_scalar(neutral_Ri, method='newton',x0=1).root
    genome[2] = Ri
    halfTheta = np.arctan((H1*np.tan(epsilonAngle))/(H1 - Ri*np.tan(epsilonAngle)))
    genome[6] = halfTheta
    propellant_index = int(genome[8])
    material_index = int(genome[9])
    propellant = srm.materials.propellants[propellants_list[propellant_index]]
    grain = srm.stargrain(N,Ro,Ri,Rp,f,epsilon,halfTheta,length)
    nozzle = srm.nozzle(exit_diameter=get_setting('exit_diameter'),
                        ambient_pressure=get_setting('ambient_pressure'),
                        exit_pressure=get_setting('exit_pressure'),
                        throat_diameter=get_setting('throat_diameter'))
    case = srm.case(material=srm.materials.steel[steel_list[material_index]],
                    safety_factor=get_setting('safety_factor'),
                    mass_fraction=get_setting('mass_fraction'))
    motor = srm.motor(propellant,grain,nozzle,case)
    if motor.grain.spoke_collision == True:
        # print("Spoke adjacency check failed. Good luck building this thing!")
        pass
    return motor

def construct_motor(genome):
    """ Constructs a motor object from a given genome """
    graintype = find_graintype(genome)
    N = int(genome[0])
    Ro = genome[1]
    Ri = genome[2]
    Rp = genome[3]
    f = genome[4]
    epsilon = genome[5]
    epsilonAngle = np.pi*N/epsilon
    if graintype == 'foobar': #'star':
        H1 = Rp*np.sin(epsilonAngle)
        halfTheta = np.arctan((H1*np.tan(epsilonAngle))/(H1-Ri*np.tan(epsilonAngle)))
    else:    
        halfTheta = genome[6]
    length = genome[7]
    propellant_index = int(genome[8])
    material_index = int(genome[9])
    # propellant_index = 4
    # material_index = 4
    propellant = srm.materials.propellants[propellants_list[propellant_index]]
    grain = srm.stargrain(N,Ro,Ri,Rp,f,epsilon,halfTheta,length)
    nozzle = srm.nozzle(exit_diameter=get_setting('exit_diameter'),
                        ambient_pressure=get_setting('ambient_pressure'),
                        exit_pressure=get_setting('exit_pressure'),
                        throat_diameter=get_setting('throat_diameter'))
    case = srm.case(material=srm.materials.steel[steel_list[material_index]],
                    safety_factor=get_setting('safety_factor'),
                    mass_fraction=get_setting('mass_fraction'))
    motor = srm.motor(propellant,grain,nozzle,case)
    if is_physical(motor) == False:
        """ Discard non-physical motor designs """
        # return None
        return motor
    else:
        return motor



def find_graintype(genome):
    N = int(genome[0])
    Ri = genome[2]
    Rp = genome[3]
    epsilon = genome[5]
    halfTheta = genome[6]
    epsilonAngle = np.pi*epsilon/N
    h = (Rp*np.cos(epsilonAngle)-((Rp*np.sin(epsilonAngle)/(np.tan(halfTheta))))-Ri)*np.sin(halfTheta)
    if h > 0:
        # Detect type of wagon wheel
        if Rp*np.cos(epsilonAngle)-Ri-((Rp*np.sin(epsilonAngle))/(np.sin(halfTheta))) > 0:
            graintype = "longspokewagonwheel"
        else:
            graintype = "shortspokewagonwheel"       
    else:
        graintype = "star"
    return graintype

def is_physical_genome(genome):
    Ro = genome[1]
    Ri = genome[2]
    Rp = genome[3]
    f = genome[4]
    epsilon = genome[5]
    halfTheta = genome[6]
    if Ri > Ro:
        return False
    elif Rp < Ri:
        return False
    elif epsilon < 0:
        return False
    elif halfTheta < 0:
        return False
    elif f < 0:
        return False
    else:
        return True

def is_physical(motor):
    if motor.grain.Ri > motor.grain.Ro:
        # print("Non-physical motor: Ri > Ro")
        return False
    elif motor.grain.Rp > motor.grain.Ri:
        # print("Non-physical motor: Rp > Ri")
        return False
    elif motor.grain.epsilon < 0:
        # print("Non-physical motor: epsilon < 0")
        return False
    elif motor.grain.halfTheta < 0:
        # print("Non-physical motor: theta < 0")
        return False
    elif motor.grain.f < 0:
        # print("Non-physical motor: f < 0")
        return False
    elif motor.grain.spoke_collision == True:
        return False
    elif motor.payload_mass() < 0:
        # print("Non-physical motor: payload mass < 0")
        return False
    else:
        return True

def init_bounds():
    N_bounds = get_bounds('N')
    Ro_bounds = get_bounds('Ro')
    Ri_bounds = get_bounds('Ri')
    Rp_bounds = get_bounds('Rp')
    f_bounds = get_bounds('f')
    epsilon_bounds = get_bounds('epsilon')
    halfTheta_bounds = get_bounds('halfTheta')
    length_bounds = get_bounds('length')
    propellant_index_bounds = (0,num_propellants)
    material_index_bounds = (0,num_steel)
    bounds = [N_bounds,Ro_bounds,Ri_bounds,Rp_bounds,f_bounds,epsilon_bounds,
              halfTheta_bounds,length_bounds,
              propellant_index_bounds,material_index_bounds]
    return bounds

rocket_bounds = init_bounds()

def print_genome(genome):
        """ This is meant to be copy-pased into a LaTeX table """
        # graintype = find_graintype(genome)
        N = int(genome[0])
        Ro = round(genome[1],3)
        Ri = round(genome[2],3)
        Rp = round(genome[3],3)
        f = round(genome[4],3)
        epsilon = round(genome[5],3)
        epsilonAngle = np.pi*N/epsilon
        length = round(genome[7],3)
        propellant_index = int(genome[8])
        material_index = int(genome[9])
        propellant = srm.materials.propellants[propellants_list[propellant_index]]
        material=srm.materials.steel[steel_list[material_index]]
        # [N, Ro, Ri, Rp, f, epsilon, theta/2, grain_length, prop_i, material_i]
        motor = construct_motor(genome)
        graintype = motor.grain.graintype
        if graintype == 'star':
            H1 = Rp*np.sin(epsilonAngle)
            halfTheta = np.arctan((H1*np.tan(epsilonAngle))/(H1-Ri*np.tan(epsilonAngle)))
        else:    
            halfTheta = genome[6]
        halfTheta = round(halfTheta,3)
        payload_mass = round(motor.payload_mass(),3)
        # max_pressure = round(motor.max_pressure(),3)
        avg_thrust = round(motor.avg_thrust(),0)
        print(f"{N} & {Ro} & {Ri} & {Rp} & {f} & {epsilon} & {halfTheta} & {length} & {propellant.name} & {material.name} & {payload_mass} & {avg_thrust} \\\\")



test_motor = construct_motor(test_genome)
motor = test_motor




def flight_performance(motor,launch_angle=0):
    """
    Models acceleration, thrust/weight ratio, final speed, etc.
    """
    
    # from srm.materials import propellants, steel
    # motor = srm.motor(propellant=propellants['HTPB_AP_AL'],grain=srm.wagonwheelgrain(N=5,Ro=4,Rp=2.2,Ri=0.8,length=24,halfTheta=1.0472,epsilon=0.7,f=0.25),
    #                   nozzle=srm.nozzle(throat_diameter=2,exit_pressure=14.7,ambient_pressure=14.7,expansion_ratio=8),case=srm.case(steel['HY-80'],1.5,0.85))
    
    timestep = 0.01
    Y = motor.burn_vector(timestep)
    t = Y[:,0]
    y = Y[:,1]
    chamber_pressure = Y[:,2]
    thrust = Y[:,3]
    mass = np.zeros(len(y))
    acceleration = np.zeros(len(y))
    speed = np.zeros(len(y))
    horiz_distance = np.zeros(len(y))
    mass_flow_rate = np.zeros(len(y))
    
    """ Burn phase """
    for i in range(0,len(y)):
        mass_flow_rate[i] = motor.burn_rate(y[i])*motor.grain.burn_area(y[i])*motor.propellant.density
        if i == 0:
            mass[i] = motor.propellant_mass(0) + motor.empty_mass()
            speed[i] = 0
            horiz_distance[i] = 0
        else:
            mass[i] = mass[i-1] - mass_flow_rate[i-1]*timestep
        acceleration[i] = thrust[i]/mass[i]
        speed[i] = speed[i-1] + acceleration[i-1]*timestep
        horiz_distance[i] = horiz_distance[i-1] + speed[i-1]*timestep
    
    """ Coast phase """
    
    
    """ Plot """
    
    # fig, ax = plt.subplots(figsize=(7, 7), dpi=96)
    # plt.title('Flight Performance')
    # plt.xlabel("Seconds")
    # plt.ylabel("")
    # # ax.plot(t,mass,label='Mass (lbm)')
    # # ax.plot(t,thrust,label="Thrust (lbf)")
    # # ax.plot(t,thrust/mass,label="Thrust/Weight Ratio")
    # # ax.plot(t,chamber_pressure,label="Chamber Pressure (psi)")
    # # ax.plot(t,mdot,label="Mass flow rate (lbm/s)")
    # ax.plot(t,acceleration,label="Acceleration (ft/s^2)")
    # ax.plot(t,speed,label="Speed (ft/s)")
    # ax.plot(t,horiz_distance,label="Horizontal distance (ft)")
    # plt.legend()
    # plt.show()
    
    return [t,chamber_pressure,thrust,mass,mass_flow_rate,acceleration]

def create_halt_event(motor):
    """
    This closure creates a burn-halting event from motor properties.
    """
    if hasattr(motor.grain,'yf'):
        yf = motor.grain.yf
        def halt_event(t,y):
            """ Halt when web burns out """
            return y[0] - yf
        halt_event.direction = 1
    elif hasattr(motor,'case'):
        def halt_event(t,y):
            """ Halt when motor mass goes down to dry mass """
            return y[1] - motor.empty_mass()
        halt_event.direction = -1
    else:
        def halt_event(t,y):
            """ If there's no case, just track propellant mass to zero """
            return y[1] - motor.propellant_mass(0)
        halt_event.direction = -1
    halt_event.terminal = True
    return halt_event

def integrate_motor_ballistics(motor,timestep=0.01):
    y0 = 0 # There may never be a good reason to change this
    runTime = 3600 # High value; hopefully the halt condition fires before this

    def create_grain_equation(motor):
        def grain_equation(t, grainvector):
            burn_distance = grainvector[0]
            a = motor.propellant.a
            n = motor.propellant.n
            rho = motor.propellant.density
            cstar = motor.propellant.cstar
            At = motor.nozzle.throat_area
            Ab = motor.grain.burn_area(burn_distance)
            P = ((Ab/At)*a*rho*cstar/srm.g0)**(1/(1-n))
            burn_rate = a*P**n
            mass_flow_rate = P*At/cstar*srm.g0
            return np.array([burn_rate,-mass_flow_rate])
        return grain_equation
    
    if hasattr(motor,'case'):
        m0 = motor.propellant_mass(y0)+motor.empty_mass()
    else:
        m0 = motor.propellant_mass(y0)
    return solve_ivp(fun=create_grain_equation(motor),
                     t_span=[0,runTime],
                     t_eval=np.arange(0,runTime,timestep),
                     y0=np.array([y0,m0]),
                     events=create_halt_event(motor)
                     )

def create_eom(motor,flight_angle_deg=90,powered_flight=True):
    """
    Creates equations of motion for the rocket. Uses the state vector:
    [x,y,xdot,ydot,burn_distance,mass]
    """
    flight_angle = np.deg2rad(flight_angle_deg)
    costheta = np.cos(flight_angle)
    sintheta = np.sin(flight_angle)
    def eom(t,statevector):
        burn_distance = statevector[4]
        xdot = statevector[2]
        ydot = statevector[3]
        mass = statevector[5]
        if powered_flight == True:
            # print("Creating EOM for powered flight.")
            a = motor.propellant.a
            n = motor.propellant.n
            rho = motor.propellant.density
            cstar = motor.propellant.cstar
            At = motor.nozzle.throat_area
            Ab = motor.grain.burn_area(burn_distance)
            P = ((Ab/At)*a*rho*cstar/srm.g0)**(1/(1-n))
            burn_rate = a*P**n
            mass_flow_rate = P*At/cstar*srm.g0
            acceleration = motor.thrust(burn_distance)/mass
        elif powered_flight == False:
            # print("Creating EOM for coast phase.")
            burn_rate = 0
            mass_flow_rate = 0
            acceleration = 0
        xddot = acceleration*costheta
        yddot = (acceleration*sintheta) + -srm.g0
        # print(f"Acceleration: {acceleration} ft/s^2\na_y: {acceleration*sintheta} ft/s^2")
        return np.array([xdot,ydot,xddot,yddot,burn_rate,-mass_flow_rate])
    return eom

def impact_event(t,y):
    """ Halt when rocket impacts ground, assuming altitude is y[1] """
    return y[1]

impact_event.terminal = True
impact_event.direction = -1

def integrate_motor_flight(motor,powered_flight=True,y0=np.array([0,0,0,0,0,0]),flight_angle_deg=90,timestep=0.01):
    runTime = 20 # placeholder
    b0 = 0
    if hasattr(motor,'case'):
        m0 = motor.propellant_mass(b0)+motor.empty_mass()
    else:
        m0 = motor.propellant_mass(b0)
    y0[5] = m0
    # print(f"Using initial conditions {y0}.")

    def create_halt_event(motor):
        """
        This closure creates a burn-halting event from motor properties.
        """
        if hasattr(motor.grain,'yf'):
            yf = motor.grain.yf
            def halt_event(t,y):
                """ Halt when web burns out """
                return y[4] - yf
            halt_event.direction = 1
        elif hasattr(motor,'case'):
            def halt_event(t,y):
                """ Halt when motor mass goes down to dry mass """
                return y[4] - motor.empty_mass()
            halt_event.direction = -1
        else:
            def halt_event(t,y):
                """ If there's no case, just track propellant mass to zero """
                return y[4] - motor.propellant_mass(0)
            halt_event.direction = -1
        halt_event.terminal = True
        return halt_event
    
    if powered_flight == True:
        events=create_halt_event(motor)
    elif powered_flight == False:
        events=impact_event
    
    eom = create_eom(motor,
                     flight_angle_deg=flight_angle_deg,
                     powered_flight=powered_flight)
    
    track = solve_ivp(fun=eom,
                     t_span=[0,runTime],
                     t_eval=np.arange(0,runTime,timestep),
                     y0=y0,
                     events=events
                     )
    # if powered_flight == True:
    #     print(f"Burnout time: {round(track.t[-1],3)} seconds.")
    # elif powered_flight == False:
    #     print(f"Impact time: {round(track.t[-1],3)} seconds.")
    return track

def plot_motor_ballistics(motor,timestep=0.01):
    G = integrate_motor_ballistics(motor,timestep)

    time = G.t
    burn_distance = G.y[0]
    mass = G.y[1]
    
    thrust = np.zeros(len(time))
    acceleration = np.zeros(len(time))
    speed = np.zeros(len(time))
    vert_distance = np.zeros(len(time))

    for i in range(0,len(time)):
        thrust[i] = motor.thrust(burn_distance[i])
        acceleration[i] = thrust[i]/mass[i] - srm.g0
        speed[i] = speed[i-1] + acceleration[i-1]*timestep
        vert_distance[i] = vert_distance[i-1] + speed[i-1]*timestep
    
    fig, ax = plt.subplots(figsize=(7, 7), dpi=96)
    plt.title('Grain Ballistics')
    plt.xlabel("Seconds")
    plt.ylabel("")
    ax.plot(time,burn_distance,label="Burn distance (in)")
    ax.plot(time,mass,label="Vehicle mass (lbm)")
    if hasattr(motor.grain,'yf') and hasattr(motor,'case'):
        plt.hlines(motor.propellant_mass(motor.grain.yf)+motor.empty_mass(),time[0],time[-1],color="black",linestyles=":",label="Burnout mass including sliver (lbm)")
    elif hasattr(motor.grain,'yf') and not hasattr(motor,'case'):
        plt.hlines(motor.propellant_mass(motor.grain.yf),time[0],time[-1],color="black",linestyles=":",label="Propellant sliver mass (lbm)")
    else:
        plt.hlines(motor.empty_mass(),time[0],time[-1],color="black",label="Vehicle dry mass (lbm)",linestyles=":")
    plt.legend()
    plt.show()
    
    fig, ax = plt.subplots(figsize=(7, 7), dpi=96)
    plt.title('Flight Performance')
    plt.xlabel("Seconds")
    plt.ylabel("")
    # ax.plot(t,mass,label='Mass (lbm)')
    # ax.plot(t,thrust,label="Thrust (lbf)")
    # ax.plot(t,thrust/mass,label="Thrust/Weight Ratio")
    # ax.plot(t,chamber_pressure,label="Chamber Pressure (psi)")
    # ax.plot(t,mdot,label="Mass flow rate (lbm/s)")
    ax.plot(time,acceleration,label="Acceleration (ft/s^2)")
    ax.plot(time,speed,label="Speed (ft/s)")
    ax.plot(time,vert_distance,label="Vertical distance (ft)")
    plt.legend()
    plt.show()

def plot_motor_flight(motor,flight_angle_deg=90,timestep=0.01):
    Y0 = np.array([0,0,0,0,0,motor.propellant_mass(0)+motor.empty_mass()])
    power = integrate_motor_flight(motor,
                                   powered_flight=True,
                                   y0=Y0,
                                   flight_angle_deg=flight_angle_deg)
    YB = power.y[:,-1]
    coast = integrate_motor_flight(motor,
                                   powered_flight=False,
                                   flight_angle_deg=flight_angle_deg,
                                   y0=YB)
    
    time = power.t
    flight = power.y
    x = flight[0]
    y = flight[1]
    xdot = flight[2]
    ydot = flight[3]
    burn_distance = flight[4]
    mass = flight[5]
    
    coast_time = coast.t + time[-1]
    coast_flight = coast.y
    coast_x = coast_flight[0]
    coast_y = coast_flight[1]
    coast_xdot = coast_flight[2]
    coast_ydot = coast_flight[3]
    coast_burn_distance = coast_flight[4]
    coast_mass = coast_flight[5]
    
    max_alt_y = max(coast_y)
    max_alt_x = coast_x[np.where(coast_y == max(coast_y))]
    max_alt_time = coast_time[np.where(coast_y ==max(coast_y))]
    
    burneom = create_eom(motor,flight_angle_deg=flight_angle_deg,powered_flight=True)
    dY0 = burneom(0,Y0)
    ax0 = dY0[2]
    ay0 = dY0[3]
    print(f"a_x(0): {ax0} ft/s^2\na_y(0): {ay0} ft/s^2")
    
    fig, ax = plt.subplots(figsize=(7, 7),dpi=96)
    plt.title(f"Altitude (Launch angle = {flight_angle_deg} deg)")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Altitude (feet)")
    ax.plot(time,y,'r',label="Powered phase")
    ax.plot(time[-1],y[-1],'rx',ms=9,label="Burnout")
    ax.plot(coast_time,coast_y,color='black',linestyle=':',label="Coast phase")
    ax.plot(max_alt_time,max_alt_y,'k.',ms=8,label=f"Max altitude ({int(max_alt_y)} ft)")
    ax.plot(coast_time[-1],coast_y[-1],'k*',ms=10,label="Impact point")
    # plt.axis('equal')
    plt.legend()
    plt.show()
    
def compute_max_altitude(motor,flight_angle_deg=90,timestep=0.01):
    Y0 = np.array([0,0,0,0,0,motor.propellant_mass(0)+motor.empty_mass()])
    power = integrate_motor_flight(motor,
                                   powered_flight=True,
                                   y0=Y0,
                                   flight_angle_deg=flight_angle_deg)
    YB = power.y[:,-1]
    coast = integrate_motor_flight(motor,
                                   powered_flight=False,
                                   flight_angle_deg=flight_angle_deg,
                                   y0=YB)
    
    time = power.t
    coast_time = coast.t + time[-1]
    coast_flight = coast.y
    coast_x = coast_flight[0]
    coast_y = coast_flight[1]
    coast_xdot = coast_flight[2]
    coast_ydot = coast_flight[3]
    coast_b = coast_flight[4]
    coast_mass = coast_flight[5]
    
    max_alt_n = np.where(coast_y == max(coast_y))
    max_alt_time = coast_time[max_alt_n]
    max_alt_x = coast_x[max_alt_n][0]
    max_alt_y = coast_y[max_alt_n][0]
    max_alt_xdot = coast_xdot[max_alt_n][0]
    max_alt_ydot = coast_ydot[max_alt_n][0]
    max_alt_b = coast_b[max_alt_n][0]
    max_alt_mass = coast_mass[max_alt_n][0]
    return np.array([max_alt_x,max_alt_y,max_alt_xdot,max_alt_ydot,max_alt_b,max_alt_mass])

Ymax = compute_max_altitude(motor)