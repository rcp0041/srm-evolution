#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GENOME:
    [N, Ro, Ri, Rp, f, epsilon, theta/2, grain_length, prop_i, material_i]
"""

from tools import get_setting, init_bounds, construct_motor, print_genome, create_eom
from problems import MaxPayload_MinPressure
from problems import MaxPayload_MaxAlt
from problems import MaxThrust_MaxAlt
from tools import plot_motor_flight

from leap_ec.representation import Representation
from leap_ec.ops import clone, evaluate, pool
from leap_ec.ops import random_selection
from leap_ec.ops import tournament_selection
from leap_ec.real_rep.initializers import create_real_vector
from leap_ec.real_rep.ops import mutate_gaussian
from leap_ec.multiobjective.nsga2 import generalized_nsga_2

bounds = init_bounds()
std=get_setting('std')
expected_num_mutations=get_setting('expected_num_mutations')
selector_type=get_setting('selector')
if selector_type == 'random':
    selector = random_selection
elif selector_type == 'tournament':
    selector = tournament_selection
pop_size = get_setting('pop_size')
max_generations = get_setting('generations')

from tools import is_physical_genome
def initialize_physical_motor(bounds):
    """ Keeps trying until it gets a physically-valid motor """
    def create():
        create_genome = create_real_vector(bounds=bounds)
        physicality = False
        while physicality == False:
            genome = create_genome()
            physicality = is_physical_genome(genome)
        return genome
    return create
    
def evolve_multiple_parameters(problem):
    return generalized_nsga_2(
        max_generations=max_generations, pop_size=pop_size,
        problem=problem(problem.maximize),
        representation=Representation(
            # initialize=create_real_vector(bounds=bounds)
            initialize=initialize_physical_motor(bounds=bounds)
        ),
        pipeline=[
            selector,
            clone,
            mutate_gaussian(std=std,
                            expected_num_mutations=expected_num_mutations,
                            bounds=bounds),
            evaluate,
            pool(size=pop_size),
        ]
    )

final_pop = evolve_multiple_parameters(MaxThrust_MaxAlt)

def fitness_list():
    for individual in final_pop:
        print(individual.fitness)

motor = construct_motor(final_pop[0].genome)
burn_eom = create_eom(motor)
print(f"Max pressure: {round(motor.max_pressure(),3)} Payload mass: {round(motor.payload_mass(),3)} Average thrust: {round(motor.avg_thrust(),3)} Max thrust: {round(motor.max_thrust(),3)}")
if motor.grain.halfTheta < 0:
    print(f"theta/2 = {motor.grain.halfTheta} ... wtf")
print_genome(final_pop[0].genome)
plot_motor_flight(motor)