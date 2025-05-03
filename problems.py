#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 15:09:56 2025

@author: ray
"""

from leap_ec.multiobjective.problems import MultiObjectiveProblem
from tools import construct_motor, compute_max_altitude
import numpy as np

class MaxPayload_MinPressure(MultiObjectiveProblem):
    """
    Maximizes payload mass and minimizes peak chamber pressure.
    """
    maximize = [1,1]
    def evaluate(self,phenome):
        motor = construct_motor(phenome)
        payload_mass = motor.payload_mass()
        max_pressure = motor.max_pressure()
        return np.array([payload_mass,-max_pressure])

class MaxPayload_MaxAvgThrust_MinThrustDiff(MultiObjectiveProblem):
    """
    Maximizes payload mass and average thrust while minimizing the
    difference between maximum thrust and average thrust.
    """
    maximize = [1,1,1]
    def evaluate(self,phenome):
        motor = construct_motor(phenome)
        payload_mass = motor.payload_mass()
        # max_pressure = motor.max_pressure()
        avg_thrust = motor.avg_thrust()
        max_thrust = motor.max_thrust()
        return np.array([payload_mass,avg_thrust,-(max_thrust-avg_thrust)])

class MaxPayload_MaxTW(MultiObjectiveProblem):
    """
    Maximizes payload mass and (initial) thrust/weight ratio.
    """
    maximize = [1,1]
    def evaluate(self,phenome):
        motor = construct_motor(phenome)
        payload_mass = motor.payload_mass()
        initial_thrust = motor.thrust(0)
        dry_mass = motor.dry_mass
        initial_prop_mass = motor.propellant_mass(0)
        Y = motor.burn_vector(0.01)
        burn_distance = Y[:,1]
        thrust = np.zeros(len(burn_distance))
        weight = np.zeros(len(burn_distance))
        thrust_weight_ratio = np.zeros(len(burn_distance))
        for i in range(0,len(burn_distance)):
            thrust[i] = motor.thrust(burn_distance[i])
            weight[i] = motor.propellant_mass(burn_distance[i]) + dry_mass
            thrust_weight_ratio[i] = thrust[i]/weight[i]
        return np.array([payload_mass,thrust_weight_ratio])

class MaxPayload_MaxAlt(MultiObjectiveProblem):
    """
    Maximizes payload mass and throw height for a vertical launch.
    """
    # maximize = [1,1]
    maximize = [1]
    def evaluate(self,phenome):
        motor = construct_motor(phenome)
        payload_mass = motor.payload_mass()
        y_max = compute_max_altitude(motor)[1]
        # return np.array([payload_mass,y_max])
        return np.array([y_max])

class MaxThrust_MaxAlt(MultiObjectiveProblem):
    """
    Maximizes initial thrust and throw height for a vertical launch.
    """
    maximize = [1,1]
    
    def evaluate(self,phenome):
        motor = construct_motor(phenome)
        max_thrust = motor.max_thrust()
        y_max = compute_max_altitude(motor)[1]
        return np.array([max_thrust,y_max])