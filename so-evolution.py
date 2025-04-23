#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GENOME:
    [N, Ro, Ri, Rp, f, epsilon, theta/2, grain_length, prop_i, material_i]
"""

from leap_ec.simple import ea_solve    # Evolutionary algorithm
import os                              # Enables access to /dev/null
from tools import construct_motor, is_physical, print_genome, init_bounds
from tools import get_setting

bounds = init_bounds()
generations = get_setting('generations')
pop_size = get_setting('pop_size')
viz = get_setting('viz')
maximize = get_setting('maximize')
hard_bounds = get_setting('hard_bounds')

def compute_payload_mass(genome):
    motor = construct_motor(genome)
    if is_physical(motor) == False:
        print("Non-physical motor")
        return 0
    else:
        return motor.payload_mass()

def compute_max_pressure(genome):
    motor = construct_motor(genome)
    if is_physical(motor) == False:
        return 0
    else:
        return -motor.max_pressure()

def compute_avg_thrust(genome):
    motor = construct_motor(genome)
    if is_physical(motor) == False:
        return 0
    else:
        return motor.avg_thrust()
    
def evolve_single_parameter(fitness_function):    
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

fitness_function = get_setting('fitness_function')
if fitness_function == 'max_pressure':
    results = evolve_single_parameter(compute_max_pressure)
elif fitness_function == 'payload_mass':
    results = evolve_single_parameter(compute_payload_mass)
elif fitness_function == 'avg_thrust':
    results = evolve_single_parameter(compute_avg_thrust)

motor = construct_motor(results)
print(f"Max pressure: {round(motor.max_pressure(),3)} Payload mass: {round(motor.payload_mass(),3)} Average thrust: {round(motor.avg_thrust(),3)}")
if motor.grain.halfTheta < 0:
    print(f"theta/2 = {motor.grain.halfTheta} ... wtf")
print_genome(results)

# This thing is basically a bomb
# import numpy as np
# genome = np.array([3,6.0,5.9,5.821,0.344,0.9,0.52,24.0,4,2])
# motor = construct_motor(genome)