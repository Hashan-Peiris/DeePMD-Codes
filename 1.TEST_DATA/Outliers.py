#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt

def read_forces(filename, n_atoms):
    """
    Reads force data from a file, ignoring lines that start with '#'.
    Each valid line has 6 columns:
      data_fx, data_fy, data_fz, pred_fx, pred_fy, pred_fz
    Returns:
      reference_forces: shape (n_frames, n_atoms, 3)
      predicted_forces: shape (n_frames, n_atoms, 3)
      n_frames
    """
    raw_data = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            values = [float(x) for x in line.split()]
            if len(values) == 6:
                raw_data.append(values)
    raw_data = np.array(raw_data)
    total_lines = raw_data.shape[0]

    if total_lines % n_atoms != 0:
        raise ValueError(
            f"[ERROR] In '{filename}': total data lines ({total_lines}) not divisible by n_atoms ({n_atoms})."
        )
    n_frames = total_lines // n_atoms
    raw_data = raw_data.reshape(n_frames, n_atoms, 6)
    reference_forces = raw_data[:, :, 0:3]
    predicted_forces = raw_data[:, :, 3:6]
    return reference_forces, predicted_forces, n_frames

def read_energies(filename):
    """
    Reads energy data from a file, ignoring lines that start with '#'.
    Each valid line has 2 columns:
      data_e, pred_e
    Returns:
      reference_energy: shape (n_frames,)
      predicted_energy: shape (n_frames,)
      n_frames
    """
    raw_data = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            values = [float(x) for x in line.split()]
            if len(values) == 2:
                raw_data.append(values)
    raw_data = np.array(raw_data)
    reference_energy = raw_data[:, 0]
    predicted_energy = raw_data[:, 1]
    n_frames = reference_energy.shape[0]
    return reference_energy, predicted_energy, n_frames

def read_stresses(filename):
    """
    Reads stress data from a file, ignoring lines that start with '#' or empty lines.
    Each valid line has 18 columns:
      data_vxx, data_vxy, data_vxz, data_vyx, data_vyy, data_vyz,
      data_vzx, data_vzy, data_vzz, pred_vxx, pred_vxy, pred_vxz,
      pred_vyx, pred_vyy, pred_vyz, pred_vzx, pred_vzy, pred_vzz
    Returns:
      reference_stresses: shape (n_frames, 9)
      predicted_stresses: shape (n_frames, 9)
      n_frames
    """
    raw_data = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            values = [float(x) for x in line.split()]
            if len(values) == 18:
                raw_data.append(values)
    raw_data = np.array(raw_data)
    n_frames = raw_data.shape[0]
    reference_stresses = raw_data[:, :9]
    predicted_stresses = raw_data[:, 9:]
    return reference_stresses, predicted_stresses, n_frames

def load_dpdata(deepmd_dir="deepmd_data"):
    """
    Loads the dpdata LabeledSystem from the specified directory.
    """
    try:
        import dpdata
    except ImportError as e:
        print(f"Error: Required module missing: {e}")
        exit(1)
    try:
        ls = dpdata.LabeledSystem(deepmd_dir, fmt="deepmd/npy")
    except Exception as e:
        print(f"Error loading dpdata LabeledSystem from {deepmd_dir}: {e}")
        exit(1)
    return ls

def detect_atom_count(energies_file, forces_file):
    """
    Determines the number of atoms per frame from the forces file based on the
    number of frames detected in the energies file.
    Returns:
      n_atoms, n_frames_energy, reference_energy, predicted_energy
    """
    reference_energy, predicted_energy, n_frames_energy = read_energies(energies_file)
    with open(forces_file, 'r') as f:
        valid_lines = [line for line in f if line.strip() and not line.strip().startswith('#')]
    total_lines = len(valid_lines)
    if total_lines % n_frames_energy != 0:
        raise ValueError(
            f"[ERROR] In '{forces_file}': total data lines ({total_lines}) not divisible by "
            f"number of frames from energies ({n_frames_energy}). Cannot deduce number of atoms."
        )
    n_atoms = total_lines // n_frames_energy
    print(f"Automatically detected total number of atoms per frame: {n_atoms}\n")
    return n_atoms, n_frames_energy, reference_energy, predicted_energy

def process_forces(ls, forces_file, n_atoms, export_top_n=100, top_k=10):
    """
    Processes forces data: computes the per-frame mean force error, prints the top outliers,
    and exports the corresponding POSCAR files using dpdata's conversion.
    """
    print("Reading forces from '{}'...".format(forces_file))
    reference_forces, predicted_forces, n_frames_forces = read_forces(forces_file, n_atoms)
    print(f" - Detected {n_frames_forces} frames total from forces data.")

    force_diff = predicted_forces - reference_forces
    force_norms = np.linalg.norm(force_diff, axis=2)
    frame_force_errors = np.mean(force_norms, axis=1)
    sorted_indices_forces = np.argsort(-frame_force_errors)

    print(f"\nTop {top_k} frames with highest force error:")
    for i in range(min(top_k, len(sorted_indices_forces))):
        frame_idx = sorted_indices_forces[i]
        print(f"  Frame {frame_idx} => Mean Force Error = {frame_force_errors[frame_idx]:.6f} eV/Å")

    # Export outlier frames as POSCAR files
    force_dir = "force_outliers"
    os.makedirs(force_dir, exist_ok=True)
    force_error_file = os.path.join(force_dir, "force_errors.txt")
    with open(force_error_file, "w") as f_force:
        f_force.write("Frame_Index\tRank\tForce_Error (eV/Å)\n")
        for rank, frame_idx in enumerate(sorted_indices_forces[:min(export_top_n, len(sorted_indices_forces))], start=1):
            poscar_name = os.path.join(force_dir, f"POSCAR_{rank}")
            try:
                ls.to_vasp_poscar(poscar_name, frame_idx=frame_idx)
            except Exception as e:
                print(f"Error converting frame {frame_idx} to POSCAR: {e}")
                continue
            f_force.write(f"{frame_idx}\t{rank}\t{frame_force_errors[frame_idx]:.6f}\n")
    return sorted_indices_forces, frame_force_errors, n_frames_forces

def process_energies(ls, energies_file, reference_energy, predicted_energy, n_atoms, export_top_n=100, top_k=10):
    """
    Processes energies data: computes the per-frame energy error, prints the top outliers,
    and exports the corresponding POSCAR files using dpdata's conversion.
    The output file now includes an extra column for the per-atom energy error.
    """
    print("\nRe-checking energies from '{}'...".format(energies_file))
    n_frames_energy = reference_energy.shape[0]
    print(f" - Detected {n_frames_energy} frames total from energy data.")

    energy_error = predicted_energy - reference_energy
    abs_energy_error = np.abs(energy_error)
    sorted_indices_energy = np.argsort(-abs_energy_error)

    print(f"\nTop {top_k} frames with highest energy error:")
    for i in range(min(top_k, len(sorted_indices_energy))):
        frame_idx = sorted_indices_energy[i]
        print(f"  Frame {frame_idx} => Energy Error = {energy_error[frame_idx]:.6f} eV")

    energy_dir = "energy_outliers"
    os.makedirs(energy_dir, exist_ok=True)
    energy_error_file = os.path.join(energy_dir, "energy_errors.txt")
    with open(energy_error_file, "w") as f_energy:
        f_energy.write("Frame_Index\tRank\tEnergy_Error (eV)\tEnergy_Error_Per_Atom (eV/atom)\n")
        for rank, frame_idx in enumerate(sorted_indices_energy[:min(export_top_n, len(sorted_indices_energy))], start=1):
            poscar_name = os.path.join(energy_dir, f"POSCAR_{rank}")
            try:
                ls.to_vasp_poscar(poscar_name, frame_idx=frame_idx)
            except Exception as e:
                print(f"Error converting frame {frame_idx} to POSCAR: {e}")
                continue
            total_err = energy_error[frame_idx]
            per_atom_err = total_err / n_atoms
            f_energy.write(f"{frame_idx}\t{rank}\t{total_err:.6f}\t{per_atom_err:.6f}\n")
    return sorted_indices_energy, energy_error, n_frames_energy

def process_stresses(ls, stresses_file="RESULTS.v_peratom.out", export_top_n=100, top_k=10):
    """
    Processes stress data: computes the per-frame mean relative stress error,
    prints the top outliers, and exports the corresponding POSCAR files using dpdata's conversion.
    The relative error is computed per component (using the mean absolute reference stress) and then averaged.
    
    Function process_stresses: This function uses read_stresses to obtain the data, then computes the stress error as 
    relative error = (∣predicted−reference∣) / (mean of |reference|) which is computed per component and then averages these over all nine components. 
    It then identifies the top outlier frames and exports their corresponding POSCAR files to a new directory (stress_outliers) using dpdata.
    """
    print("\nReading stresses from '{}'...".format(stresses_file))
    try:
        ref_stresses, pred_stresses, n_frames_stress = read_stresses(stresses_file)
        print(f" - Detected {n_frames_stress} frames total from stress data.")
    except Exception as e:
        print(f"Error reading stress data: {e}")
        return

    stress_diff = pred_stresses - ref_stresses
    # Compute the mean absolute reference stress for each of the 9 components
    mean_ref_stress = np.mean(np.abs(ref_stresses), axis=0)
    eps = 1e-8  # avoid division by zero
    mean_ref_stress = np.where(mean_ref_stress == 0, eps, mean_ref_stress)
    relative_errors = np.abs(stress_diff) / mean_ref_stress
    frame_stress_errors = np.mean(relative_errors, axis=1)
    sorted_indices_stress = np.argsort(-frame_stress_errors)

    print(f"\nTop {top_k} frames with highest stress error (mean relative):")
    for i in range(min(top_k, len(sorted_indices_stress))):
        frame_idx = sorted_indices_stress[i]
        print(f"  Frame {frame_idx} => Mean Relative Stress Error = {frame_stress_errors[frame_idx]:.6f}")

    # Export outlier frames as POSCAR files
    stress_dir = "stress_outliers"
    os.makedirs(stress_dir, exist_ok=True)
    stress_error_file = os.path.join(stress_dir, "stress_errors.txt")
    with open(stress_error_file, "w") as f_stress:
        f_stress.write("Frame_Index\tRank\tMean_Relative_Stress_Error\n")
        for rank, frame_idx in enumerate(sorted_indices_stress[:min(export_top_n, len(sorted_indices_stress))], start=1):
            poscar_name = os.path.join(stress_dir, f"POSCAR_{rank}")
            try:
                ls.to_vasp_poscar(poscar_name, frame_idx=frame_idx)
            except Exception as e:
                print(f"Error converting frame {frame_idx} to POSCAR: {e}")
                continue
            f_stress.write(f"{frame_idx}\t{rank}\t{frame_stress_errors[frame_idx]:.6f}\n")
    print(f"Export complete: Stress outlier POSCARs and errors written in directory '{stress_dir}'\n")
    return sorted_indices_stress, frame_stress_errors, n_frames_stress

def plot_trajectory_errors(energies_file="RESULTS.e.out", 
                           forces_file="RESULTS.f.out", 
                           stresses_file="RESULTS.v_peratom.out",
                           normalize=True):
    """
    Reads the full trajectory from the energies, forces, and stresses files,
    computes the error for each frame, and plots a line graph with the frame number
    on the x-axis and the corresponding errors on the y-axis.
    
    If 'normalize' is True, each error is divided by its maximum value so that 
    all curves are scaled between 0 and 1. The plot is then saved as 'trajectory_errors.png'.
    """
    # Energy errors
    ref_energy, pred_energy, n_frames_energy = read_energies(energies_file)
    energy_errors = np.abs(pred_energy - ref_energy)
    
    # Forces errors (detect n_atoms first)
    n_atoms, _, _, _ = detect_atom_count(energies_file, forces_file)
    ref_forces, pred_forces, n_frames_forces = read_forces(forces_file, n_atoms)
    force_diff = pred_forces - ref_forces
    force_norms = np.linalg.norm(force_diff, axis=2)
    force_errors = np.mean(force_norms, axis=1)
    
    # Stress errors
    ref_stresses, pred_stresses, n_frames_stress = read_stresses(stresses_file)
    stress_diff = pred_stresses - ref_stresses
    mean_ref_stress = np.mean(np.abs(ref_stresses), axis=0)
    eps = 1e-8
    mean_ref_stress = np.where(mean_ref_stress == 0, eps, mean_ref_stress)
    relative_errors = np.abs(stress_diff) / mean_ref_stress
    stress_errors = np.mean(relative_errors, axis=1)
    
    # Use the minimum number of frames among the three datasets
    n_frames = min(n_frames_energy, n_frames_forces, n_frames_stress)
    frames = np.arange(n_frames)
    
    plt.figure(figsize=(10,6))
    if normalize:
        # Normalize each error vector by its maximum value
        energy_norm = energy_errors[:n_frames] / np.max(energy_errors)
        force_norm = force_errors[:n_frames] / np.max(force_errors)
        stress_norm = stress_errors[:n_frames] / np.max(stress_errors)
        plt.plot(frames, energy_norm, label="Energy Error (normalized)", color="red")
        plt.plot(frames, force_norm, label="Force Error (normalized)", color="blue")
        plt.plot(frames, stress_norm, label="Stress Error (normalized)", color="green")
        plt.ylim(0, 1)
        ylabel = "Normalized Error"
    else:
        plt.plot(frames, energy_errors[:n_frames], label="Energy Error (eV)", color="red")
        plt.plot(frames, force_errors[:n_frames], label="Force Error (eV/Å)", color="blue")
        plt.plot(frames, stress_errors[:n_frames], label="Stress Error (rel. error)", color="green")
        ylabel = "Error"
    
    plt.xlabel("Frame Number")
    plt.ylabel(ylabel)
    plt.title("Trajectory Error Overview")
    plt.legend()
    plt.tight_layout()
    # Save the figure instead of showing it
    plt.savefig("trajectory_errors.png", dpi=300)
    plt.close()
    print("Trajectory error plot saved as 'trajectory_errors.png'.")

def plot_individual_errors(energies_file="RESULTS.e.out", 
                           forces_file="RESULTS.f.out", 
                           stresses_file="RESULTS.v_peratom.out"):
    """
    Plots three separate figures showing the absolute error values for each snapshot.
    - Energy errors are shown as per-atom energy errors.
    - Force errors are the mean force errors (eV/Å).
    - Stress errors are the mean relative stress errors.
    
    Each plot is saved to an individual file.
    """
    # Energy Plot (per-atom energy error)
    n_atoms, n_frames_energy, ref_energy, pred_energy = detect_atom_count(energies_file, forces_file)
    energy_errors = np.abs(pred_energy - ref_energy) / n_atoms
    frames_energy = np.arange(n_frames_energy)
    plt.figure(figsize=(10,6))
    plt.plot(frames_energy, energy_errors, color="red", label="Per-atom Energy Error (eV/atom)")
    plt.xlabel("Frame Number")
    plt.ylabel("Per-atom Energy Error (eV/atom)")
    plt.title("Per-atom Energy Error over Trajectory")
    plt.legend()
    plt.tight_layout()
    plt.savefig("energy_error.png", dpi=300)
    plt.close()
    print("Energy error plot saved as 'energy_error.png'.")

    # Force Plot (mean force error)
    ref_forces, pred_forces, n_frames_forces = read_forces(forces_file, n_atoms)
    force_diff = pred_forces - ref_forces
    force_norms = np.linalg.norm(force_diff, axis=2)
    force_errors = np.mean(force_norms, axis=1)
    frames_force = np.arange(n_frames_forces)
    plt.figure(figsize=(10,6))
    plt.plot(frames_force, force_errors, color="blue", label="Mean Force Error (eV/Å)")
    plt.xlabel("Frame Number")
    plt.ylabel("Mean Force Error (eV/Å)")
    plt.title("Mean Force Error over Trajectory")
    plt.legend()
    plt.tight_layout()
    plt.savefig("force_error.png", dpi=300)
    plt.close()
    print("Force error plot saved as 'force_error.png'.")

    # Stress Plot (mean relative stress error)
    ref_stresses, pred_stresses, n_frames_stress = read_stresses(stresses_file)
    stress_diff = pred_stresses - ref_stresses
    mean_ref_stress = np.mean(np.abs(ref_stresses), axis=0)
    eps = 1e-8
    mean_ref_stress = np.where(mean_ref_stress == 0, eps, mean_ref_stress)
    relative_errors = np.abs(stress_diff) / mean_ref_stress
    stress_errors = np.mean(relative_errors, axis=1)
    frames_stress = np.arange(n_frames_stress)
    plt.figure(figsize=(10,6))
    plt.plot(frames_stress, stress_errors, color="green", label="Mean Relative Stress Error")
    plt.xlabel("Frame Number")
    plt.ylabel("Mean Relative Stress Error")
    plt.title("Mean Relative Stress Error over Trajectory")
    plt.legend()
    plt.tight_layout()
    plt.savefig("stress_error.png", dpi=300)
    plt.close()
    print("Stress error plot saved as 'stress_error.png'.")

def main():
    print("=== Outlier Identification Script ===")
    print("Big Chungus! Running automatic atom count detection.\n")

    energies_file = "RESULTS.e.out"
    forces_file = "RESULTS.f.out"

    # Determine number of frames and atoms
    print("Reading energies from '{}'...".format(energies_file))
    n_atoms, n_frames_energy, reference_energy, predicted_energy = detect_atom_count(energies_file, forces_file)
    print(f" - Detected {n_frames_energy} frames total from energy data.\n")

    # Load dpdata dataset
    ls = load_dpdata("deepmd_data")

    # Process forces
    print("Processing forces data...")
    process_forces(ls, forces_file, n_atoms, export_top_n=100, top_k=10)

    # Process energies (now with per-atom error column)
    print("\nProcessing energies data...")
    process_energies(ls, energies_file, reference_energy, predicted_energy, n_atoms, export_top_n=100, top_k=10)

    # Process stresses
    print("\nProcessing stresses data...")
    process_stresses(ls, stresses_file="RESULTS.v_peratom.out", export_top_n=100, top_k=10)

    print("=== Outlier Identification Complete ===\n")

    # Save the combined trajectory error plot
    print("Saving trajectory errors plot...")
    plot_trajectory_errors(energies_file, forces_file, stresses_file="RESULTS.v_peratom.out")

    # Save individual error plots for energies, forces, and stresses
    print("Saving individual error plots...")
    plot_individual_errors(energies_file, forces_file, stresses_file="RESULTS.v_peratom.out")

if __name__ == "__main__":
    main()
