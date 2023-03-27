import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import random
from pathlib import Path

##INFO
p_constant = 5
potential_type = 'tslj'
boltzmann = 1
# temperature = 0.01
diffusion_translational = 0.01
diffusion_rotational = 1  # Dr ∈ {0.25, 1.0}
dt = 1e-5
rho = 0.2  # total particle area to box_area ratio  ρ ∈ {0.1, 0.2}
length_scale = 1
interaction_radius = 3 * length_scale
duplicate_search_threshold = interaction_radius
N = 10
particle_area = (length_scale / 2) ** 2 * np.pi
box_area = N * particle_area / rho
box_len = np.sqrt(box_area)
if box_len / 2 < duplicate_search_threshold:
    raise Exception("The box length is very small in comparison to the interaction radius")

increment = length_scale / 10000
F_P = 60
energy_parameter = 1
interact_across_borders = True

its = int(1e3)  # 1e5
sample_its = 10
sample_indices_vector = np.linspace(0, its - 1, sample_its, dtype=int)

ADDITIONS = [np.array([0, 0]), np.array([box_len, 0]), np.array([-box_len, 0]), np.array([0, box_len]),
             np.array([0, -box_len]), np.array([box_len, box_len]), np.array([-box_len, box_len]),
             np.array([box_len, -box_len]), np.array([-box_len, -box_len])]


def get_particles(n_particles=N):
    coordinates = np.zeros((n_particles, 2))

    # Ensuring that particles don't start too close to each other
    coord1 = np.random.uniform(-box_len / 2, box_len / 2, 2)
    coord2 = np.zeros(2)
    for i in range(n_particles):
        j = 0
        while j < i:
            coord2 = coordinates[j]
            d_v = coord1 - coord2
            d = np.sqrt(d_v.dot(d_v))
            if d < length_scale * 1.1:
                coord1 = np.random.uniform(-box_len / 2, box_len / 2, 2)
                j = 0
            else:
                j += 1
        coordinates[i] = coord1

    orientations = np.random.uniform(low=0, high=2 * np.pi, size=(n_particles, 1))
    return coordinates, orientations


##POTENTIALS AND THEIR GRADIENTS - FUNCTIONS
def get_lj_truncation_value(truncation_distance):
    return get_lj_potential(truncation_distance)


def get_lj_potential(distance, truncation_distance=box_len, truncation_value=0):
    if distance < truncation_distance:
        potential = 4 * energy_parameter * ((1 / distance) ** 12 - (1 / distance) ** 6) - truncation_value
    else:
        potential = 0
    return potential


def get_lj_potential_gradient(distance_vector, truncation_distance, truncation_value):
    distance = np.sqrt(distance_vector.dot(distance_vector))
    difference = get_lj_potential(distance + increment, truncation_distance, truncation_value) - get_lj_potential(
        distance - increment, truncation_distance, truncation_value)
    gradient_magnitude = difference / (2 * increment)
    gradient = gradient_magnitude * -distance_vector / distance  # minus the distance vector since this gives the correct force directions w.r.t. potential gradient sign
    return gradient


def get_tslj_potential(distance):
    return get_lj_potential(distance, truncation_distance=2.5, truncation_value=get_lj_truncation_value(2.5))


def get_tslj_potential_gradient(distance_vector):
    return get_lj_potential_gradient(distance_vector, truncation_distance=2.5,
                                     truncation_value=get_lj_truncation_value(2.5))


def get_wca_potential(distance):
    r_cut = length_scale * (2 ** (1 / 6))
    return get_lj_potential(distance, truncation_distance=r_cut, truncation_value=get_lj_truncation_value(r_cut))


def get_wca_potential_gradient(distance_vector):
    r_cut = length_scale * 2 ** (1 / 6)
    return get_lj_potential_gradient(distance_vector, truncation_distance=r_cut,
                                     truncation_value=get_lj_truncation_value(r_cut))


def get_srs_potential(r, n=14, k0=10 / length_scale, eps_s=1, sig_s=2.5):
    return energy_parameter * (length_scale / r) ** n + 1 / 2 * eps_s * (1 - np.tanh(k0 * (r - sig_s)))


def get_srs_potential_gradient(distance_vector):
    distance = np.sqrt(distance_vector.dot(distance_vector))
    difference = get_srs_potential(distance + increment) - get_srs_potential(distance - increment)
    gradient_magnitude = difference / (2 * increment)
    gradient = gradient_magnitude * -distance_vector / distance  # minus the distance vector since this gives the correct force directions w.r.t. potential gradient sign
    return gradient


def get_potential(distance, key):
    if key == 'tslj':
        return get_tslj_potential(distance)
    elif key == 'wca':
        return get_wca_potential(distance)
    elif key == 'srs':
        return get_srs_potential(distance)


def get_potential_gradient(distance_vector, key):
    if key == 'tslj':
        return get_tslj_potential_gradient(distance_vector)
    elif key == 'wca':
        return get_wca_potential_gradient(distance_vector)
    elif key == 'srs':
        return get_srs_potential_gradient(distance_vector)

    ##GET INFO - FUNCTIONS


def get_velocity(orientation, f_p, summed_gradients_vector):
    # thermal_force = np.sqrt(2*diffusion_translational) * np.random.randn(2)  #~0.1
    active_force = np.squeeze(np.array([np.cos(orientation), np.sin(orientation)]) * f_p)  # ~0.3
    passive_force = -summed_gradients_vector  # diffusion_translational * -summed_gradients_vector  #varies based on density
    # print(passive_force)

    return active_force + passive_force


def get_velocity_separated(orientation, f_p, summed_gradients_vector):
    # thermal_force = np.sqrt(2*diffusion_translational) * np.random.randn(2)  #~0.1
    active_force = np.squeeze(np.array([np.cos(orientation), np.sin(orientation)]) * f_p)  # ~0.3
    passive_force = -summed_gradients_vector  # varies based on density
    # print(passive_force)

    return active_force + passive_force, active_force, passive_force


def get_energies(coordinates, orientations, velocities, truncation_distance, truncation_value, t):  # OLD AND UNUSED
    n_particles = coordinates.shape[0]
    new_coordinates = coordinates
    new_orientations = orientations
    saved_distances = np.zeros(shape=(n_particles, n_particles, 2))
    kinetic_energy = 0
    potential_energy = 0

    for i, pos0 in enumerate(coordinates):
        gradients_sum = np.zeros(2)
        for j, pos1 in enumerate(coordinates):
            d_v = pos1 - pos0
            saved_distances[i, j, :] = d_v
            if d_v[0] < duplicate_search_threshold and d_v[1] < duplicate_search_threshold and i != j:
                gradients_sum += get_lj_potential_gradient(d_v, truncation_distance, truncation_value)
            if i != j:
                d = np.sqrt(d_v.dot(d_v))
                potential_energy += get_lj_potential(d, truncation_distance, truncation_value)

        velocity = get_velocity(orientations[i], F_P, gradients_sum)
        kinetic_energy += velocity.dot(velocity)  # m = 1
    kinetic_energy = kinetic_energy / 2
    potential_energy = potential_energy / 2
    total_energy = kinetic_energy + potential_energy

    return kinetic_energy, potential_energy, total_energy


##UPDATING - FUNCTIONS
def update_data(coordinates, orientations, full_data_dict, potential_key, t):
    full_data_dict['centroid-0'][N * t: N + N * t] = coordinates[:, 0]
    full_data_dict['centroid-1'][N * t: N + N * t] = coordinates[:, 1]
    full_data_dict['orientation'][N * t: N + N * t] = np.squeeze(orientations)

    new_coordinates = coordinates
    new_orientations = orientations

    for i, pos0 in enumerate(coordinates):
        gradients_sum = np.zeros(2)
        for addition in ADDITIONS:
            for j, pos1 in enumerate(coordinates + addition):
                d_v = pos1 - pos0
                d = np.sqrt(d_v.dot(d_v))
                if d < duplicate_search_threshold and i != j:
                    gradients_sum += get_potential_gradient(d_v, potential_key)

        velocity, active_vel, passive_vel = get_velocity_separated(orientations[i], F_P,
                                                                   gradients_sum)  # in this case, velocity = force
        angular_diffusion_step = np.sqrt(2 * diffusion_rotational * dt) * np.random.randn()
        diffusion_step = np.sqrt(dt * diffusion_translational * 2) * np.random.randn(2)
        new_coordinates[i, :] += dt * velocity + diffusion_step
        new_orientations[i] += angular_diffusion_step

        while coordinates[i][0] > box_len / 2:
            coordinates[i][0] -= box_len
        while coordinates[i][0] < -box_len / 2:
            coordinates[i][0] += box_len
        while coordinates[i][1] > box_len / 2:
            coordinates[i][1] -= box_len
        while coordinates[i][1] < -box_len / 2:
            coordinates[i][1] += box_len

        while orientations[i] < 0:
            orientations[i] += np.pi * 2
        while orientations[i] > np.pi * 2:
            orientations[i] -= np.pi * 2

        velocity_array = np.array([p_constant, p_constant])
        full_data_dict['solution'][i + t * N, :] = velocity_array  # check if size correct

    coordinates = new_coordinates
    orientations = new_orientations


##SIMULATION
## Arrays to save stuff into
centroids_x = np.zeros((N * sample_its))
centroids_y = np.zeros((N * sample_its))
orientations = np.zeros((N * sample_its))
labels = np.arange(N * sample_its,
                   dtype=np.int64)  # Set this to this since label is assumed to not matter for the moment
solutions = []  # solution structured as: [4-length np array for passive & active force, 4-length np array for passive & active force, ...]
frames = np.zeros((N * sample_its), dtype=np.int64)
for t in np.arange(sample_its, dtype=np.int64):
    for i in range(N):
        frames[i + t * N] = t
sets = np.zeros(N * sample_its, dtype=np.int64)
# print(frames)
data_dict = {'label': labels, 'centroid-0': centroids_x, 'centroid-1': centroids_y, 'orientation': orientations,
             'solution': solutions, 'frame': frames, 'set': sets}

all_centroids_x = np.zeros((N * its))
all_centroids_y = np.zeros((N * its))
all_orientations = np.zeros((N * its))
all_solutions = np.zeros((N * its, 2))
full_data_dict = {'centroid-0': all_centroids_x, 'centroid-1': all_centroids_y, 'orientation': all_orientations,
                  'solution': all_solutions}

coordinates, orientations = get_particles()
tic = time.time()
for t in range(its):
    update_data(coordinates, orientations, full_data_dict, potential_type, t)
    if not np.mod(t, its / 100):
        toc = time.time()
        T = (toc - tic)
        print(str(int(t / its * 100)) + ' %, runtime: ' + str(T) + 's.', end='\r')
for st, t in enumerate(sample_indices_vector):
    data_dict['centroid-0'][N * st: N + N * st] = all_centroids_x[N * t: N + N * t]
    data_dict['centroid-1'][N * st: N + N * st] = all_centroids_y[N * t: N + N * t]
    data_dict['orientation'][N * st: N + N * st] = all_orientations[N * t: N + N * t]
    for i in range(N):
        solutions.append(all_solutions[
                             i + N * t])  # I found this structure to be a way to make vector input to one column in pandas work
toc = time.time()
T = (toc - tic) / its
print('Total runtime: ' + str(T * its)[:10] + ' s, (' + str(T)[:] + ' s per iteration)')

##SAVING DATA
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
path = Path(dir_path)
parent_path = path.parent.absolute()
datasets_path = str(parent_path) + '/datasets/'
np.save(datasets_path + potential_type + '/CPASSIVE N' + str(N) + ' samples' + str(
    sample_its) + ' F_P' + str(F_P), {**data_dict,
                                                    **{'box_len': box_len, 'interaction_radius': interaction_radius,
                                                       'potential_type': potential_type}})

##PLOTS
color_v = plt.cm.brg(np.linspace(0, 1, N))
for i in range(N):
    for t in range(0, its, int(its / 40)):
        if t == 0:
            plt.scatter(all_centroids_x[i + t * N], all_centroids_y[i + t * N], color=color_v[i], s=40, alpha=0.5,
                        marker='*')
        else:
            plt.scatter(all_centroids_x[i + t * N], all_centroids_y[i + t * N], color=color_v[i], s=4, alpha=0.5)

        cir = plt.Circle((all_centroids_x[i + t * N], all_centroids_y[i + t * N]), 0.5, color='b', fill=False,
                         alpha=0.2)
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        ax.add_patch(cir)

        plt.title("Scatter trajectories. Radius = " + str(length_scale))
        plt.xlim([-box_len / 2, box_len / 2])
        plt.ylim([-box_len / 2, box_len / 2])

plt.show()
