#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GENOME:
    [N, Ro, Ri, Rp, f, epsilon, theta/2, grain_length, prop_i, material_i]
"""

import srm                             # Solid rocket motor performance library
from leap_ec.simple import ea_solve    # Evolutionary algorithm
from leap_ec.multiobjective.problems import MultiObjectiveProblem
import toml                            # For reading config file
import os                              # Enables access to /dev/null
import numpy as np                     # Numerical operations
from scipy import optimize

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
    nozzle = srm.nozzle(expansion_ratio=8,
                        ambient_pressure=14.7,
                        exit_pressure=14.7,
                        throat_diameter=2.0)
    case = srm.case(material=srm.materials.steel[steel_list[material_index]],
                    safety_factor=1.3,
                    mass_fraction=0.85)
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
    nozzle = srm.nozzle(expansion_ratio=8,
                        ambient_pressure=14.7,
                        exit_pressure=14.7,
                        throat_diameter=2.0)
    case = srm.case(material=srm.materials.steel[steel_list[material_index]],
                    safety_factor=1.3,
                    mass_fraction=0.85)
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

def compute_payload_mass(genome):
    motor = construct_motor(genome)
    
    # Try to throw out non-physical designs
    if motor.grain.Ri > motor.grain.Ro:
        return 0
    elif motor.grain.Rp > motor.grain.Ri:
        return 0
    elif motor.grain.epsilon < 0:
        return 0
    elif motor.grain.halfTheta < 0:
        return 0
    elif motor.grain.f < 0:
        return 0
    elif motor.payload_mass() < 0:
        return 0
    else:
        return motor.payload_mass()

def compute_max_pressure(genome):
    motor = construct_motor(genome)
    
    # Try to throw out non-physical designs
    if motor.grain.Ri > motor.grain.Ro:
        return 0
    elif motor.grain.Rp > motor.grain.Ri:
        return 0
    elif motor.grain.epsilon < 0:
        return 0
    elif motor.grain.halfTheta < 0:
        return 0
    elif motor.grain.f < 0:
        return 0
    elif motor.payload_mass() < 0:
        return 0
    else:
        return 1/motor.max_pressure()
    
def evolve_single_parameter(fitness_function):    
    generations = get_setting('generations')
    pop_size = get_setting('pop_size')
    viz = get_setting('viz')
    maximize = get_setting('maximize')
    hard_bounds = get_setting('hard_bounds')
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
    
    results = ea_solve(fitness_function,
                       generations=generations,
                       pop_size=pop_size,
                       viz=viz,
                       bounds=bounds,
                       hard_bounds=hard_bounds,
                       stream=open(os.devnull, 'w'),
                       maximize=maximize)
    # return construct_motor(results)
    return results

""" A SOO problem """
# SOmotor = construct_motor(evolve_single_parameter(compute_payload_mass))
# print(SOmotor.grain)
# print(f"SO Payload mass: {SOmotor.payload_mass()}")
# print(f"SO Max pressure: {SOmotor.max_pressure()}")
# SOmotor.plot(['pressure','thrust'])

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

""" Attempting a MOO problem """
class MO_Rocket_Problem(MultiObjectiveProblem):
    def evaluate(self,phenome):
        # motor = construct_motor(phenome)
        # if is_physical(motor) == False:
        #     return np.array([0,0])
        # else:
        #     return np.array([motor.payload_mass(),1/motor.max_pressure()])
        
        payload_mass = compute_payload_mass(phenome)
        max_pressure = compute_max_pressure(phenome)
        if payload_mass <= 0 or max_pressure <= 0:
            return np.array([0,0])
        else:
            return np.array([payload_mass,max_pressure])

def initialize_rocket_bounds():
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

rocket_bounds = initialize_rocket_bounds()
    
from leap_ec.representation import Representation
from leap_ec.ops import random_selection, clone, evaluate, pool
from leap_ec.ops import tournament_selection
from leap_ec.real_rep.initializers import create_real_vector
from leap_ec.real_rep.ops import mutate_gaussian
from leap_ec.multiobjective.nsga2 import generalized_nsga_2

pop_size = get_setting('pop_size')
max_generations = get_setting('generations')
final_pop = generalized_nsga_2(
    max_generations=max_generations, pop_size=pop_size,
    problem=MO_Rocket_Problem([1,1]),
    representation=Representation(
        initialize=create_real_vector(bounds=rocket_bounds)
    ),
    pipeline=[
        random_selection,
        # tournament_selection,
        clone,
        mutate_gaussian(std=0.5, expected_num_mutations=1,bounds=rocket_bounds),
        # mutate_gaussian(std=0.1,expected_num_mutations=1,bounds=rocket_bounds),
        evaluate,
        pool(size=pop_size),
    ]
)

def get_motor(i):
    motor = construct_motor(final_pop[i].genome)
    # print(motor.grain)
    print(f"MO Payload mass: {motor.payload_mass()}")
    print(f"MO Max pressure: {motor.max_pressure()}")
    # motor.plot(['pressure'])
    return motor

motor = get_motor(0)