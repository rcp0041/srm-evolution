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

# Create lists that allow numeric indexing of propellant/material dictionaries
propellants_list = list(srm.materials.propellants)
steel_list = list(srm.materials.steel)
num_propellants = len(propellants_list) - 1
num_steel = len(steel_list) - 1

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
    # TODO: Implement spoke-check retry (will allow total parameter variation)
    """ Constructs a motor object from a given genome """
    graintype = find_graintype(genome)
    N = int(genome[0])
    Ro = genome[1]
    Ri = genome[2]
    Rp = genome[3]
    f = genome[4]
    epsilon = genome[5]
    epsilonAngle = np.pi*N/epsilon
    if graintype == 'star':
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
    if motor.grain.spoke_collision == True:
        # print("Spoke adjacency check failed. Good luck building this thing!")
        pass
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

def is_physical(motor):
    if motor.grain.Ri > motor.grain.Ro:
        return False
    elif motor.grain.Rp > motor.grain.Ri:
        return False
    elif motor.grain.epsilon < 0:
        return False
    elif motor.grain.halfTheta < 0:
        return False
    elif motor.grain.f < 0:
        return False
    elif motor.payload_mass() < 0:
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
        max_pressure = round(motor.max_pressure(),3)
        print(f"{graintype} & {N} & {Ro} & {Ri} & {Rp} & {f} & {epsilon} & {halfTheta} & {length} & {propellant.name} & {material.name} & {payload_mass} & {max_pressure} \\\\")