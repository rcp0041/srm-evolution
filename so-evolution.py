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
        return 0
    else:
        return motor.payload_mass()

def compute_max_pressure(genome):
    motor = construct_motor(genome)
    if is_physical(motor) == False:
        return 0
    else:
        return 1/motor.max_pressure()
    
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
print_genome(results)