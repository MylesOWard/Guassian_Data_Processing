#!/usr/bin/env python3
import numpy as np
import numba as nb
import matplotlib.pyplot as plt
import statistics
import argparse

# Shared parameters
a = 1.0    # Length of each segment
angle_P = 0.0 * np.pi / 180  # P angle in radians (0 degrees)
angle_M = 60.0 * np.pi / 180  # M angle in radians (60 degrees)

def rotation_matrix_from_axis_angle(axis, angle):
    """Return a 3x3 rotation matrix for rotation about 'axis' by 'angle'.
    Handles small-axis magnitude safely."""
    ax = np.asarray(axis, dtype=float)
    norm_ax = np.linalg.norm(ax)
    if norm_ax < 1e-12:
        return np.eye(3)
    x, y, z = ax / norm_ax
    c = np.cos(angle)
    s = np.sin(angle)
    t = 1 - c
    return np.array([
        [t*x*x + c,   t*x*y - s*z, t*x*z + s*y],
        [t*x*y + s*z, t*y*y + c,   t*y*z - s*x],
        [t*x*z - s*y, t*y*z + s*x, t*z*z + c  ]
    ])

def unit(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v * 0.0
    return v / n

def add_bond_in_local_frame(prev_pos, prev_bond, bond_length, valence_angle, dihedral_angle):
    """Construct new bond vector given previous bond direction.
    Works by creating the new bond in the local coordinates where prev_bond is the +z axis,
    then rotating it back into the global frame."""
    # local bond coordinates (in frame where previous bond = +z)
    # spherical coords: theta = valence_angle from z-axis, phi = dihedral_angle in xy-plane
    sin_t = np.sin(valence_angle)
    local = np.array([
        bond_length * sin_t * np.cos(dihedral_angle),
        bond_length * sin_t * np.sin(dihedral_angle),
        bond_length * np.cos(valence_angle)
    ])
    # rotation to bring local z to prev_bond direction
    z_axis = np.array([0.0, 0.0, 1.0])
    v = unit(prev_bond)
    # If prev_bond is already (close to) +z, rotation is identity
    if np.allclose(v, z_axis, atol=1e-10):
        R = np.eye(3)
    elif np.allclose(v, -z_axis, atol=1e-10):
        # 180-degree rotation about any axis perpendicular to z (choose x)
        R = rotation_matrix_from_axis_angle(np.array([1.0, 0.0, 0.0]), np.pi)
    else:
        # axis that rotates z_axis onto v is cross(z_axis, v)
        axis = np.cross(z_axis, v)
        axis = unit(axis)
        angle = np.arccos(np.clip(np.dot(z_axis, v), -1.0, 1.0))
        R = rotation_matrix_from_axis_angle(axis, angle)
    # rotate local bond into global frame
    new_bond = R.dot(local)
    new_pos = prev_pos + new_bond
    return new_pos, new_bond

@nb.njit
def rotation_matrix_from_axis_angle_nb(axis, angle):
    """3x3 rotation matrix for rotation about axis by angle."""
    norm_ax = np.sqrt(axis[0]**2 + axis[1]**2 + axis[2]**2)
    if norm_ax < 1e-12:
        return np.eye(3)
    x = axis[0] / norm_ax
    y = axis[1] / norm_ax
    z = axis[2] / norm_ax
    c = np.cos(angle)
    s = np.sin(angle)
    t = 1.0 - c
    R = np.empty((3,3), dtype=np.float64)
    R[0,0] = t*x*x + c
    R[0,1] = t*x*y - s*z
    R[0,2] = t*x*z + s*y
    R[1,0] = t*x*y + s*z
    R[1,1] = t*y*y + c
    R[1,2] = t*y*z - s*x
    R[2,0] = t*x*z - s*y
    R[2,1] = t*y*z + s*x
    R[2,2] = t*z*z + c
    return R

@nb.njit
def unit_nb(v):
    n = np.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
    if n < 1e-12:
        return np.zeros(3)
    return v / n

@nb.njit
def generate_polymer_chain_nb(N, a, angle_P, angle_M, p_P):
    coords = np.zeros((N + 1, 3), dtype=np.float64)

    # First bond: along +z
    coords[1, 2] = a
    prev_bond = coords[1] - coords[0]

    # Pre-generate random choices
    rand_vals = np.random.random(N - 1)  # decide P or M
    dihedrals = np.random.uniform(0.0, 2.0*np.pi, N - 1)

    for i in range(2, N + 1):
        # pick bond angle
        if rand_vals[i - 2] < p_P:
            valence_angle = angle_P
        else:
            valence_angle = angle_M

        dihedral_angle = dihedrals[i - 2]

        # local new bond (prev bond = +z)
        sin_t = np.sin(valence_angle)
        local = np.array([
            a * sin_t * np.cos(dihedral_angle),
            a * sin_t * np.sin(dihedral_angle),
            a * np.cos(valence_angle)
        ], dtype=np.float64)

        # rotate local to global
        z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        v = unit_nb(prev_bond)

        if (abs(v[0]) < 1e-10 and abs(v[1]) < 1e-10 and abs(v[2]-1.0) < 1e-10):
            R = np.eye(3)
        elif (abs(v[0]) < 1e-10 and abs(v[1]) < 1e-10 and abs(v[2]+1.0) < 1e-10):
            R = rotation_matrix_from_axis_angle_nb(np.array([1.0, 0.0, 0.0]), np.pi)
        else:
            axis = np.array([
                z_axis[1]*v[2] - z_axis[2]*v[1],
                z_axis[2]*v[0] - z_axis[0]*v[2],
                z_axis[0]*v[1] - z_axis[1]*v[0]
            ], dtype=np.float64)
            axis = unit_nb(axis)
            dot = v[2]  # z_axis dot v
            if dot > 1.0: dot = 1.0
            elif dot < -1.0: dot = -1.0
            angle = np.arccos(dot)
            R = rotation_matrix_from_axis_angle_nb(axis, angle)

        new_bond = R.dot(local)
        coords[i] = coords[i - 1] + new_bond
        prev_bond = new_bond

    return coords

def check_segment_angles(coords):
    """Return list of bond angles (degrees) between consecutive segments."""
    angles = []
    for i in range(1, len(coords) - 1):
        v1 = coords[i] - coords[i - 1]
        v2 = coords[i + 1] - coords[i]
        # If lengths are fixed this is small overhead but kept for clarity
        v1u = unit(v1)
        v2u = unit(v2)
        cosang = np.clip(np.dot(v1u, v2u), -1.0, 1.0)
        angle = np.arccos(cosang)
        angles.append(np.degrees(angle))
    return angles

def check_dihedral_angles(coords):
    """Return list of dihedral angles (degrees) for quadruples of points."""
    dihedrals = []
    for i in range(3, len(coords)):
        p0 = coords[i - 3]
        p1 = coords[i - 2]
        p2 = coords[i - 1]
        p3 = coords[i]
        b1 = p1 - p0
        b2 = p2 - p1
        b3 = p3 - p2
        n1 = np.cross(b1, b2)
        n2 = np.cross(b2, b3)
        n1u = unit(n1)
        n2u = unit(n2)
        if np.linalg.norm(n1) < 1e-12 or np.linalg.norm(n2) < 1e-12:
            # degenerate (colinear) case - define dihedral as 0
            dihedrals.append(0.0)
            continue
        x_val = np.dot(n1u, n2u)
        y_val = np.dot(np.cross(n1u, unit(b2)), n2u)
        dihedral = np.degrees(np.arctan2(y_val, x_val))
        dihedrals.append(dihedral)
    return dihedrals

def calculate_squared_end_to_end_distance(coords):
    squared_distance = np.sum((coords[-1] - coords[0])**2)
    return squared_distance

def test_end_to_end_distance(N, M, p_P):
    squared_distances = []
    for _ in range(M):
        coords = generate_polymer_chain_nb(N, a, angle_P, angle_M, p_P)
        squared_distances.append(calculate_squared_end_to_end_distance(coords))
    mean_squared_distance = statistics.mean(squared_distances)
    rms_distance = np.sqrt(mean_squared_distance)
    root_distances = np.sqrt(squared_distances)
    print(f"Empirical RMS end-to-end (over {M} chains): {rms_distance:.6f}")

    # Freely-jointed chain reference
    expected_fj = np.sqrt(N) * a
    print(f"Freely-jointed RMS (sqrt(N)*a): {expected_fj:.6f}")

    # Approximate freely-rotating formula using average cos(theta)
    avg_cos_theta = p_P * np.cos(angle_P) + (1.0 - p_P) * np.cos(angle_M)
    if abs(1 - avg_cos_theta) < 1e-12:
        expected_fr = np.nan
    else:
        expected_fr = np.sqrt(N) * a * np.sqrt((1 + avg_cos_theta) / (1 - avg_cos_theta))
    print(f"Freely-rotating approx RMS: {expected_fr:.6f}")

    plt.figure()
    plt.hist(root_distances, bins=30, edgecolor='black')
    if not np.isnan(expected_fr):
        plt.axvline(expected_fr, color='red', linestyle='dashed', linewidth=2, label='Freely-rotating approx')
    plt.axvline(expected_fj, color='green', linestyle='dashed', linewidth=2, label='Freely-jointed')
    plt.title('Histogram of end-to-end distances (RMS values per chain)')
    plt.xlabel('End-to-end distance')
    plt.ylabel('Frequency')
    plt.legend()
#    plt.show()

def plot_the_chain(N, p_P):
    coords = generate_polymer_chain(N, a, angle_P, angle_M, p_P)
    squared_distance = calculate_squared_end_to_end_distance(coords)
    rms_distance = np.sqrt(squared_distance)
    print(f"Single-chain end-to-end distance: {rms_distance:.6f}")

    angles = check_segment_angles(coords)
    dihedrals = check_dihedral_angles(coords)
    print(f"Angles between segments (first 10): {angles[:10]}")
    print(f"Mean angle: {np.mean(angles):.2f} deg, std: {np.std(angles):.2f} deg")
    print(f"Dihedral angles (first 10): {dihedrals[:10]}")
    print(f"Mean dihedral: {np.mean(dihedrals):.2f} deg, std: {np.std(dihedrals):.2f} deg")

    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    plt.hist(angles, bins=50, range=(0, 180), edgecolor='black')
    plt.axvline(np.degrees(angle_P), color='red', linestyle='dashed', linewidth=2, label='Angle P')
    plt.axvline(np.degrees(angle_M), color='blue', linestyle='dashed', linewidth=2, label='Angle M')
    plt.title('Histogram of bond angles (deg)')
    plt.xlabel('Angle (deg)')
    plt.ylabel('Frequency')
    plt.legend()

    plt.subplot(122)
    plt.hist(dihedrals, bins=50, range=(-180, 180), edgecolor='black')
    plt.title('Histogram of dihedral angles (deg)')
    plt.xlabel('Dihedral angle (deg)')
    plt.ylabel('Frequency')
    plt.tight_layout()
#    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(coords[:,0], coords[:,1], coords[:,2], marker='o')
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
#    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Polymer Chain Simulation (local-frame growth)")
    parser.add_argument("routine", choices=["test", "plot"], help="Choose 'test' for end-to-end distance test or 'plot' for chain visualization")
    parser.add_argument("-L", "--length", type=int, default=100, help="Chain length (number of segments)")
    parser.add_argument("-M", "--chains", type=int, default=500, help="Number of chains to simulate (only for 'test' routine)")
    parser.add_argument("-p", "--prob", type=float, default=0.5, help="Probability of para state")

    args = parser.parse_args()

    if args.routine == "test":
        test_end_to_end_distance(args.length, args.chains, args.prob)
    else:
        plot_the_chain(args.length, args.prob)

if __name__ == "__main__":
    main()
