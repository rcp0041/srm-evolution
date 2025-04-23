#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GENOME:
    [N, Ro, Ri, Rp, f, epsilon, theta/2, grain_length, prop_i, material_i]
"""

from numpy import array
from tools import construct_motor, is_physical, init_bounds, print_genome
from tools import get_setting

from leap_ec.multiobjective.problems import MultiObjectiveProblem   
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

class MO_Rocket_Problem(MultiObjectiveProblem):
    def evaluate(self,phenome):
        motor = construct_motor(phenome)
        if is_physical(motor) == False:
            return array([0,0])
        else:
            return array([motor.payload_mass(),1/motor.max_pressure()])

final_pop = generalized_nsga_2(
    max_generations=max_generations, pop_size=pop_size,
    problem=MO_Rocket_Problem([1,1]),
    representation=Representation(
        initialize=create_real_vector(bounds=bounds)
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

print_genome(final_pop[0].genome)