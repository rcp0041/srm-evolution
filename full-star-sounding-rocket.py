#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  3 00:09:19 2025

@author: ray
"""

# SRM requirements
from srm2 import Motor, Nozzle, Case
from srm2.grain import Star, WagonWheel
import srm2.utilities as utilities
import srm2.performance as performance
from srm2.propellant import propellants
prop_list = list(propellants.keys())
from srm2.materials import steel
steel_list = list(steel.keys())

# LEAP-EC requirements
from leap_ec.multiobjective.problems import MultiObjectiveProblem
from leap_ec.representation import Representation
from leap_ec.ops import clone, evaluate, pool
from leap_ec.ops import tournament_selection
from leap_ec.real_rep.initializers import create_real_vector
from leap_ec.real_rep.ops import mutate_gaussian
from leap_ec.multiobjective.nsga2 import generalized_nsga_2

import numpy as np

# Problem bounds
min_allowable_payload = 10
min_allowable_apogee = 62
max_allowable_g = 150
launch_angle_deg = 90

# Star/WW-grain sounding rocket phenotype parameters:
# [N, Ro, Ri, Rp, f, epsilon, halfTheta, length, expansion_ratio, SF, MF, prop_i, steel_i]
bounds = np.array([
    (3,13), # N
    (1,10), # Ro
    (1,10), # Ri
    (1,10), # Rp
    (0.1,0.9), # f
    (0.1,0.9), # epsilon
    (0.1,2.0), # halfTheta
    (1,60), # Grain length
    (3,10), # Expansion ratio
    (1.1,1.1), # Safety factor
    (0.85,0.85), # Mass fraction
    (0,len(prop_list)-1), # Chooses propellant type
    (0,len(steel_list)-1) # Chooses case material
    ])

# Evolution parameters
std=0.1
expected_num_mutations='isotropic' # 1
selector = tournament_selection
pop_size = 10
max_generations = 60

def decode_Star_WW_genome(genome,verbose=False):
    def set_nonviable(genome,message):
        gene_dict['viable'] = False
        if 'message' in genome.keys():
            gene_dict['message'] = gene_dict['message'] + message
        else:
            gene_dict['message'] = message
    
    gene_dict = {}
    gene_dict['N'] = genome[0]
    gene_dict['Ro'] = genome[1]
    gene_dict['Ri'] = genome[2]
    gene_dict['Rp'] = genome[3]
    gene_dict['f'] = genome[4]
    gene_dict['epsilon'] = genome[5]
    gene_dict['halfTheta'] = genome[6]
    gene_dict['length'] = genome[7]
    gene_dict['expansion_ratio'] = genome[8]
    gene_dict['SF'] = genome[9]
    gene_dict['MF'] = genome[10]
    gene_dict['prop_i'] = genome[11]
    gene_dict['steel_i'] = genome[12]
    
    gene_dict['viable'] = True
    
    # Ri < Rp < Ro for a viable grain
    if gene_dict['Ri'] >= gene_dict['Ro']:
        set_nonviable(gene_dict,'Ri >= Ro.\n')
    if gene_dict['Ri'] >= gene_dict['Rp']:
        set_nonviable(gene_dict,'Ri >= Rp.\n')
    if gene_dict['Rp'] >= gene_dict['Ro']:
        set_nonviable(gene_dict,'Rp >= Ro.\n')
    if gene_dict['expansion_ratio'] <= 1:
        set_nonviable(gene_dict,'Expansion ratio is 1 or less.\n')
    if gene_dict['viable'] == False and verbose == True:
        print(gene_dict['message'])
    return gene_dict

def construct_grain(genome):
    gene_dict = decode_Star_WW_genome(genome)
    N = int(gene_dict['N'])
    Ro = gene_dict['Ro']
    Ri = gene_dict['Ri']
    Rp = gene_dict['Rp']
    f = gene_dict['f']
    epsilon = gene_dict['epsilon']
    epsilonAngle = np.pi*epsilon/N
    halfTheta = gene_dict['halfTheta']
    length = gene_dict['length']
    
    # Detect star (h < 0) or wagon wheel (h > 0)
    h = (Rp*np.cos(epsilonAngle)-((Rp*np.sin(epsilonAngle)/(np.tan(halfTheta))))-Ri)*np.sin(halfTheta)
    if h > 0:
        # Must be a wagon wheel
        return WagonWheel(N,Ro,Ri,Rp,f,epsilon,halfTheta,length)
    else:
        # Must be a star
        return Star(N,Ro,Ri,Rp,f,epsilon,halfTheta,length)

def construct_motor(genome,allow_nonviable=False,compute_motor=False,verbose=False):
    def emit_motor(gene_dict,compute_motor: bool):
        expansion_ratio = gene_dict['expansion_ratio']
        SF = gene_dict['SF']
        MF = gene_dict['MF']
        prop_i = int(gene_dict['prop_i'])
        grain = construct_grain(genome)
        propellant = propellants[prop_list[prop_i]]
        steel_i = int(gene_dict['steel_i'])
        material = steel[steel_list[steel_i]]
        return Motor(propellant=propellant,
                      grain=grain,
                      nozzle=Nozzle(exit_radius=grain.Ro,
                                    expansion_ratio=expansion_ratio,
                                    gamma=propellant.specific_heat_ratio),
                      case=Case(material=material,
                                safety_factor=SF,
                                mass_fraction=MF),
                      compute=compute_motor
                      )
    
    # Decode genome into dictionary
    gene_dict = decode_Star_WW_genome(genome,verbose=verbose)
    
    # Check viability before constructing motor (if applicable)
    if allow_nonviable == False:
        if gene_dict['viable'] == False:
            return None
        elif gene_dict['viable'] == True:
            return emit_motor(gene_dict,compute_motor)
    elif allow_nonviable == True:
        return emit_motor(gene_dict,compute_motor)

def initialize_viable_motor(bounds):
    # Keeps trying until it gets a physically-viable motor
    def create():
        create_genome = create_real_vector(bounds=bounds)
        viable = False
        while viable == False:
            genome = create_genome()
            viable = decode_Star_WW_genome(genome)['viable']
        return genome
    return create

class SoundingRocket_Problem(MultiObjectiveProblem):
    maximize = [1,1,1]
    def evaluate(self,phenome):
        genome_dict = decode_Star_WW_genome(phenome)
        viable = genome_dict['viable']
        payload_mass_score,apogee_score,max_g_score=0,0,0
        if viable == True:
            motor = construct_motor(phenome)
            flight = performance.Flight(motor,launch_angle_deg)
            payload_mass = motor.payload_mass
            apogee = utilities.feet2miles(flight.stats.max_y)
            max_g = performance.compute_max_g(motor,launch_angle_deg)
            
            if payload_mass >= min_allowable_payload:
                payload_mass_score = 1
            if apogee >= min_allowable_apogee:
                apogee_score = 1
            if max_g < max_allowable_g:
                max_g_score = 1
        return np.array([payload_mass_score,apogee_score,max_g_score])

def evolve_motor(problem):
    return generalized_nsga_2(
        max_generations=max_generations, pop_size=pop_size,
        problem=problem(problem.maximize),
        representation=Representation(
            initialize=initialize_viable_motor(bounds=bounds),
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

def get_motor_emit_table():
    final_pop = evolve_motor(SoundingRocket_Problem)
    motor = construct_motor(final_pop[0].genome,compute_motor=True)
    flight = performance.Flight(motor,launch_angle_deg)
    flight.plot('altitude')
    gene_dict = decode_Star_WW_genome(final_pop[0].genome)
    N = int(gene_dict['N'])
    Ro = round(gene_dict['Ro'],2)
    Ri = round(gene_dict['Ri'],2)
    Rp = round(gene_dict['Rp'],2)
    f = round(gene_dict['f'],2)
    epsilon = round(gene_dict['epsilon'],2)
    halfTheta = round(gene_dict['halfTheta'],2)
    length = round(gene_dict['length'],2)
    prop = motor.propellant.name
    material = motor.case.material.name
    payload = round(motor.payload_mass,2)
    apogee = round(utilities.feet2miles(flight.apogee_y),2)
    max_g = round(performance.compute_max_g(motor,launch_angle_deg),2)
    print(f"Apogee height: {apogee} mi\nPayload mass: {payload} lbm\nPeak acceleration: {max_g} g")
    print(f"{N} & {Ro} & {Ri} & {Rp} & {f} & {epsilon} & {halfTheta} & {length} & {prop} & {material} & {payload} & {apogee} & {max_g}")
    return motor
    
motor = get_motor_emit_table()