#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from toolz import pipe
from leap_ec import Individual
from leap_ec import ops, util
from leap_ec.decoder import IdentityDecoder
from leap_ec.problem import ScalarProblem
from leap_ec.real_rep.initializers import create_real_vector
from leap_ec.real_rep.ops import mutate_gaussian


import numpy as np                     # Math operations
from srm import srm                    # Solid rocket motor performance library
from srm.materials import propellants  # Dictionary of solid rocket propellants
from srm.materials import steel        # Dictionary of case materials
from matplotlib import pyplot as plt   # Plotting library
import toml                            # For reading config file

data = toml.load("./config.toml")
def get_bounds(key_string):
    key_string = key_string + "_bounds"
    return (data[key_string][0],data[key_string][1])



def max_pressure(motor,timestep=0.1):
    Y = motor.burn_vector(timestep)
    return max(Y[:,2])

def payload_mass(case,motor):
    # Get case properties
    SF = case.safety_factor
    rho = case.density
    sigma = case.yield_strength
    MF = case.mass_fraction
    
    # Get motor properties
    Rc = motor.grain.Ro
    L = motor.grain.length
    Pc = max_pressure(motor)
    Dt = motor.nozzle.throat_diameter
    
    t = (SF*Rc*Pc)/sigma
    M_cyl = 2*np.pi*Rc*t*L*rho
    t_end = SF*(Rc*Pc)/(2*sigma)
    M_nose = 2*np.pi*Rc**2 * t_end * rho
    phi = np.arctan(Dt/(2*Rc))
    M_aft = M_nose*np.cos(phi)
    t_nozzle = SF*((Dt*Pc)/(2*sigma))
    Rt = Dt/2
    Re = Rt*np.sqrt(motor.nozzle.expansion_ratio)
    L_nozzle = (Re-Rt)/np.tan(phi)
    M_nozzle = rho*((np.pi*L_nozzle)/3) * (((Re+t_nozzle)**2 + (Re+t_nozzle)*(Rt+t_nozzle) + (Rt+t_nozzle)**2)-(Re**2+Re*Rt+Rt**2))
    M_inert = ((1/MF) - 1)*motor.propellant_mass(0)
    M_payload = M_inert - M_cyl - M_nose - M_aft - M_nozzle
    return M_payload

class PayloadMassProblem(ScalarProblem):
    def __init__(self, maximize=False):
        super().__init__(maximize)

    def evaluate(self, phenome):
        N = phenome[0]
        Ro=phenome[1]
        Ri=phenome[2]
        Rp=phenome[3]
        f=phenome[4]
        epsilon=phenome[5]
        halfTheta=phenome[6]
        length=phenome[7]
        propellant_index=phenome[8]
        material_index=phenome[9]
        
        prop = propellants[list(propellants)[int(propellant_index)]]
        grain = srm.stargrain(int(N),Ro,Ri,Rp,f,epsilon,halfTheta,length)
        case = srm.case(material=steel[list(steel)[int(material_index)]],safety_factor=1.3,mass_fraction=0.85)
        nozzle = srm.nozzle(expansion_ratio=8,
                            ambient_pressure=14.7,
                            exit_pressure=14.7,
                            throat_diameter=2.0)
        motor = srm.motor(prop,grain,nozzle)
        M_payload = payload_mass(case,motor)
        return M_payload

    def worse_than(self, first_fitness, second_fitness):
        return super().worse_than(first_fitness, second_fitness)

    def __str__(self):
        return PayloadMassProblem.__name__

def print_population(population, generation):
    """ Convenience function for pretty printing a population that's
    associated with a given generation

    :param population:
    :param generation:
    :return: None
    """
    for individual in population:
        print(generation, individual.genome, individual.fitness)

def get_best(population, generation, maximize=True):
    best_fitness = population[0].fitness
    i = 1
    best_individual = 0
    while i < len(population):
        if maximize == True:
            if population[i].fitness > best_fitness:
                best_individual = i
                best_fitness = population[best_individual].fitness
            i = i+1
        elif maximize == False:
            if population[i].fitness < best_fitness:
                best_individual = i
                best_fitness = population[best_individual].fitness
            i = i+1
    return population[best_individual]

def print_best(population,generation,maximize=True):
    best_individual = get_best(population,generation,maximize=True)
    print("Best individual: {} Fitness: {}".format(population[best_individual].genome,population[best_individual].fitness))
    

# BROOD_SIZE = 3  # how many offspring each parent will reproduce
# POPULATION_SIZE = 24
# MAX_GENERATIONS = 100

BROOD_SIZE = data['brood_size']
POPULATION_SIZE = data['pop_size']
MAX_GENERATIONS = data['max_generations']

if __name__ == '__main__':
    # N,Ro,Ri,Rp,f,epsilon,halfTheta,length,propellant_index,material_index)
    N_bounds = get_bounds('N')
    Ro_bounds = get_bounds('Ro')
    Ri_bounds = get_bounds('Ri')
    Rp_bounds = get_bounds('Rp')
    f_bounds = get_bounds('f')
    epsilon_bounds = get_bounds('epsilon')
    halfTheta_bounds = get_bounds('halfTheta')
    length_bounds = get_bounds('length')
    propellant_index_bounds = (0,len(propellants)-1)
    material_index_bounds = (0,len(steel)-1)
    bounds = [N_bounds,Ro_bounds,Ri_bounds,Rp_bounds,f_bounds,epsilon_bounds,halfTheta_bounds,length_bounds,propellant_index_bounds,material_index_bounds]
    parents = Individual.create_population(POPULATION_SIZE,
                                                initialize=create_real_vector(
                                                    bounds),
                                                decoder=IdentityDecoder(),
                                                problem=PayloadMassProblem(maximize=True))
    # Evaluate initial population
    parents = Individual.evaluate_population(parents)
    max_generation = MAX_GENERATIONS
    # Set up a generation counter using the default global context variable
    generation_counter = util.inc_generation()

    while generation_counter.generation() < max_generation:
        offspring = pipe(parents,
                         ops.random_selection,
                         ops.clone,
                         mutate_gaussian(std=.1, expected_num_mutations=1),
                         ops.evaluate,
                         ops.pool(
                             size=len(parents) * BROOD_SIZE),
                         # create the brood
                         ops.insertion_selection(parents=parents))

        parents = offspring
        generation_counter()  # increment to the next generation
    
    # print_population(parents,generation=max_generation)
    # print_best(parents,generation=max_generation,maximize=True)
    winner = get_best(parents,generation=max_generation).genome
    N = winner[0]
    Ro = winner[1]
    Ri = winner[2]
    Rp = winner[3]
    f = winner[4]
    epsilon = winner[5]
    halfTheta = winner[6]
    length = winner[7]
    propellant = propellants[list(propellants)[int(winner[8])]]
    grain = srm.stargrain(int(N),Ro,Ri,Rp,f,epsilon,halfTheta,length)
    case = srm.case(material=steel[list(steel)[int(winner[9])]],safety_factor=1.3,mass_fraction=0.85)
    nozzle = srm.nozzle(expansion_ratio=8,
                        ambient_pressure=14.7,
                        exit_pressure=14.7,
                        throat_diameter=2.0)
    motor = srm.motor(propellant,grain,nozzle)
    M_payload = payload_mass(case,motor)
    print("Winning design:\n\n{}\n{}\n{}\nPayload mass: {} lbm".format(motor.propellant.name,case,motor.grain,M_payload))
    
    timestep = 0.01
    Y = motor.burn_vector(timestep)
    time = Y[:,0]
    burn_distance = Y[:,1]
    pressure = Y[:,2]
    thrust = Y[:,3]
    # Isp = Y[:,4]
    mdot = np.zeros_like(time)
    mdot[0] = motor.mass_flow_rate(0)
    impulse = np.zeros_like(time)
    burn_area = np.zeros_like(time)
    Isp = np.zeros_like(time)
    Isp[0] = thrust[0]/(motor.mass_flow_rate(0)*srm.g0)
    burn_area[0] = motor.grain.burn_area(0)
    for i in range(1,len(time)):
        impulse[i] = impulse[i-1] + thrust[i]*timestep
        burn_area[i] = motor.grain.burn_area(time[i])
        mdot[i] = motor.mass_flow_rate(time[i])
        Isp[i] = thrust[i]/(mdot[i]*srm.g0)
        
    
    plotdir = "figures/run{}".format(data['run'])

    fig, ax = plt.subplots(figsize=(7, 7), dpi=96)
    ax.plot(time,pressure)
    plt.title("Chamber pressure")
    plt.xlabel("Burn Time (s)")
    plt.ylabel("Pressure (psi)")
    plt.savefig('{}/pressure.png'.format(plotdir),dpi=100)
    plt.show()

    fig, ax = plt.subplots(figsize=(7, 7), dpi=96)
    ax.plot(time,thrust)
    plt.title("Thrust")
    plt.xlabel("Burn Time (s)")
    plt.ylabel("Thrust (lbf)")
    plt.savefig('{}/thrust.png'.format(plotdir),dpi=100)
    plt.show()

    fig, ax = plt.subplots(figsize=(7, 7), dpi=96)
    ax.plot(time,Isp)
    plt.title("Specific Impulse (CP grain)")
    plt.xlabel("Burn Time (s)")
    plt.ylabel("Isp (s)")
    plt.savefig('{}/isp.png'.format(plotdir),dpi=100)
    plt.show()

    # fig, ax = plt.subplots(figsize=(7, 7), dpi=96)
    # ax.plot(time,impulse)
    # plt.title("Total Impulse (CP grain)")
    # plt.xlabel("Burn Time (s)")
    # plt.ylabel("I (lbf-s)")
    # plt.show()

    fig, ax = plt.subplots(figsize=(7, 7), dpi=96)
    ax.plot(time,mdot*srm.g0)
    plt.title("Mass flow rate (CP grain)")
    plt.xlabel("Burn Time (s)")
    plt.ylabel("Mdot (lbm/s)")
    plt.savefig('{}/mdot.png'.format(plotdir),dpi=100)
    plt.show()