import numpy as np
import matplotlib.pyplot as plt

def read_data(filename, num_columns):
    """
    Reads data from a file, skipping comment lines starting with '#'.
    Returns a NumPy array with the specified number of columns.
    """
    data = []
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            values = [float(x) for x in line.strip().split()]
            if len(values) != num_columns:
                continue  # Skip lines that don't have the expected number of columns
            data.append(values)
    return np.array(data)

def plot_pred_vs_ref(reference, predicted, xlabel, ylabel, title, filename):
    """
    Plots predicted values vs. reference values and saves the plot.
    """
    error = predicted - reference
    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(error**2))
    
    plt.figure(figsize=(8,6))
    plt.scatter(reference, predicted, s=5, alpha=0.7)
    min_val = min(np.min(reference), np.min(predicted))
    max_val = max(np.max(reference), np.max(predicted))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Fit')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f'{title}\nMAE: {mae:.4e}, RMSE: {rmse:.4e}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    #plt.show()
    
    return error

def plot_error_distribution(error, xlabel, title, filename):
    """
    Plots the distribution of errors as a histogram and saves the plot.
    """
    plt.figure(figsize=(8,6))
    plt.hist(error, bins=50, alpha=0.7, edgecolor='black')
    mean_error = np.mean(error)
    std_error = np.std(error)
    plt.xlabel(xlabel)
    plt.ylabel('Frequency')
    plt.title(f'{title}\nMean Error: {mean_error:.4e}, Std Dev: {std_error:.4e}')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    #plt.show()

def plot_pred_vs_ref_frame(reference, predicted, xlabel, ylabel, title, filename, points_per_frame=1, cmap='viridis', frame_offset=1):
    """
    Plots predicted vs. reference values with points colored by their frame number.
    'points_per_frame' indicates how many data points correspond to one frame.
    'frame_offset' is added so that the first frame is labeled as frame_offset (default 1).
    """
    error = predicted - reference
    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(error**2))
    
    # Determine the number of frames and generate frame indices
    n_frames = len(reference) // points_per_frame
    frame_indices = np.repeat(np.arange(n_frames) + frame_offset, points_per_frame)
    
    # In case there are leftover points, assign them the last frame number.
    if len(frame_indices) < len(reference):
        remainder = len(reference) - len(frame_indices)
        frame_indices = np.concatenate((frame_indices, np.full(remainder, n_frames - 1 + frame_offset)))
    
    plt.figure(figsize=(8,6))
    sc = plt.scatter(reference, predicted, s=5, alpha=0.7, c=frame_indices, cmap=cmap)
    plt.colorbar(sc, label='Frame Number')
    min_val = min(np.min(reference), np.min(predicted))
    max_val = max(np.max(reference), np.max(predicted))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Fit')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f'{title} (Colored by Frame)\nMAE: {mae:.4e}, RMSE: {rmse:.4e}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    #plt.show()
    
    return error

def get_color_palette(palette_name, num_colors=9):
    """
    Returns a list of 'num_colors' colors from the specified matplotlib colormap.
    If 'default' is chosen, returns a preset list of colors.
    """
    if palette_name.lower() == 'default':
        return ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive']
    else:
        cmap = plt.get_cmap(palette_name)
        return [cmap(i) for i in np.linspace(0, 1, num_colors)]

def plot_total_stress_colored(ref_stress_ev, pred_stress_ev, conversion_factor, component_labels, palette='Set1', output_filename='total_stress_colored_plot.png'):
    """
    Plots the total stress tensor (predicted vs. reference) colored by stress component.
    Calculates the overall MAE and RMSE and includes them in the plot title.
    """
    # Convert stress units from eV/Å³ to GPa
    ref_all = ref_stress_ev * conversion_factor
    pred_all = pred_stress_ev * conversion_factor
    
    # Calculate overall error metrics
    error = pred_all.flatten() - ref_all.flatten()
    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(error**2))
    
    # Determine axis limits
    min_val = min(np.min(ref_all), np.min(pred_all))
    max_val = max(np.max(ref_all), np.max(pred_all))
    
    # Get vivid colors from the selected palette
    colors = get_color_palette(palette, len(component_labels))
    
    plt.figure(figsize=(8,6))
    # Plot each stress component with its corresponding color
    for i, label in enumerate(component_labels):
        ref_comp = ref_stress_ev[:, i] * conversion_factor
        pred_comp = pred_stress_ev[:, i] * conversion_factor
        plt.scatter(ref_comp, pred_comp, color=colors[i], s=5, alpha=0.7, label=label)
    
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Fit')
    plt.xlabel('Reference Stress (GPa)')
    plt.ylabel('Predicted Stress (GPa)')
    plt.title(f'Total Stress Tensor: Predicted vs Reference (Colored by Component)\nMAE: {mae:.4e}, RMSE: {rmse:.4e}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    #plt.show()
    
    return error

def plot_pred_vs_ref_species(reference, predicted, species, xlabel, ylabel, title, filename, cmap='tab10'):
    """
    Plots predicted vs. reference values with points colored by species type.
    'species' is an array of the same length as the flattened reference and predicted arrays,
    containing the element symbol for each data point.
    """
    error = predicted - reference
    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(error**2))
    
    unique_species = np.unique(species)
    cmap_obj = plt.get_cmap(cmap)
    colors = {sp: cmap_obj(i/len(unique_species)) for i, sp in enumerate(unique_species)}
    
    plt.figure(figsize=(8,6))
    for sp in unique_species:
        mask = (species == sp)
        plt.scatter(reference[mask], predicted[mask], s=5, alpha=0.7, color=colors[sp], label=sp)
    
    min_val = min(np.min(reference), np.min(predicted))
    max_val = max(np.max(reference), np.max(predicted))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Fit')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f'{title}\nMAE: {mae:.4e}, RMSE: {rmse:.4e}')
    plt.legend(title='Species')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    #plt.show()
    
    return error

# -----------------------------------------------------------
# Main Script
# -----------------------------------------------------------

# Energy per atom (units: eV/atom)
energy_data = read_data('RESULTS.e_peratom.out', 2)
reference_energy = energy_data[:, 0]
predicted_energy = energy_data[:, 1]
energy_error = plot_pred_vs_ref(
    reference_energy,
    predicted_energy,
    'Reference Energy (eV/atom)',
    'Predicted Energy (eV/atom)',
    'Energy: Predicted vs Reference',
    'energy_plot.png'
)
plot_error_distribution(
    energy_error,
    'Energy Error (eV/atom)',
    'Energy Error Distribution',
    'energy_error_distribution.png'
)
plot_pred_vs_ref_frame(
    reference_energy,
    predicted_energy,
    'Reference Energy (eV/atom)',
    'Predicted Energy (eV/atom)',
    'Energy: Predicted vs Reference',
    'energy_frame_plot.png',
    points_per_frame=1,
    frame_offset=1
)

# Stress per atom
# Conversion factor: 1 eV/Å³ = 160.21766208 GPa
stress_data = read_data('RESULTS.v_peratom.out', 18)
reference_stress_ev = stress_data[:, :9]
predicted_stress_ev = stress_data[:, 9:]
conversion_factor = 160.21766208
reference_stress = reference_stress_ev.flatten() * conversion_factor
predicted_stress = predicted_stress_ev.flatten() * conversion_factor

stress_error = plot_pred_vs_ref(
    reference_stress,
    predicted_stress,
    'Reference Stress (GPa)',
    'Predicted Stress (GPa)',
    'Stress: Predicted vs Reference',
    'stress_plot.png'
)
plot_error_distribution(
    stress_error,
    'Stress Error (GPa)',
    'Stress Error Distribution',
    'stress_error_distribution.png'
)
plot_pred_vs_ref_frame(
    reference_stress,
    predicted_stress,
    'Reference Stress (GPa)',
    'Predicted Stress (GPa)',
    'Stress: Predicted vs Reference',
    'stress_frame_plot.png',
    points_per_frame=9,
    frame_offset=1
)

# Forces (units: eV/Å)
force_data = read_data('RESULTS.f.out', 6)
reference_force = force_data[:, :3].flatten()
predicted_force = force_data[:, 3:].flatten()

# ----------------------------------------------------------------------
# Determine atoms_per_frame from species file instead of hardcoding
species_single = np.loadtxt('./deepmd_data/type.raw', dtype=int)
atoms_per_frame = len(species_single)  # Number of atoms per snapshot

force_error = plot_pred_vs_ref(
    reference_force,
    predicted_force,
    'Reference Force (eV/Å)',
    'Predicted Force (eV/Å)',
    'Force: Predicted vs Reference',
    'force_plot.png'
)
plot_error_distribution(
    force_error,
    'Force Error (eV/Å)',
    'Force Error Distribution',
    'force_error_distribution.png'
)
plot_pred_vs_ref_frame(
    reference_force,
    predicted_force,
    'Reference Force (eV/Å)',
    'Predicted Force (eV/Å)',
    'Force: Predicted vs Reference',
    'force_frame_plot.png',
    points_per_frame=atoms_per_frame * 3,  # 3 force components per atom
    frame_offset=1
)

# ----------------------------------------------------------------------
# New Plot: Force Plot Colored by Species Type
# ----------------------------------------------------------------------
# Read species data for one snapshot from deepmd_data directory
# type.raw contains the atomic type indices for one snapshot.
species_single = np.loadtxt('./deepmd_data/type.raw', dtype=int)

# Read type_map.raw correctly: each line is an element symbol
type_map = {}
with open('./deepmd_data/type_map.raw', 'r') as f:
    for idx, line in enumerate(f):
        symbol = line.strip()
        if symbol:  # non-empty line
            type_map[idx] = symbol

# Convert species indices to element symbols for one snapshot.
species_single_converted = np.array([type_map[i] for i in species_single])

# Determine number of frames from force data.
n_frames = int(len(force_data) / atoms_per_frame)

# Build species array for all frames (each frame has the same species ordering).
species_all = np.tile(species_single_converted, n_frames)
# For the flattened force arrays, repeat each species 3 times (one for each force component).
species_all_for_plot = np.repeat(species_all, 3)

# Plot predicted vs. reference colored by species for forces.
plot_pred_vs_ref_species(
    reference_force,
    predicted_force,
    species_all_for_plot,
    'Reference Force (eV/Å)',
    'Predicted Force (eV/Å)',
    'Force: Predicted vs Reference (Colored by Species)',
    'force_species_plot.png',
    cmap='tab10'
)

# ----------------------------------------------------------------------
# Plot each stress component individually
component_labels = ['vxx', 'vxy', 'vxz', 'vyx', 'vyy', 'vyz', 'vzx', 'vzy', 'vzz']
for i, label in enumerate(component_labels):
    ref_comp = reference_stress_ev[:, i] * conversion_factor
    pred_comp = predicted_stress_ev[:, i] * conversion_factor
    comp_error = plot_pred_vs_ref(
        ref_comp,
        pred_comp,
        f'Reference Stress {label} (GPa)',
        f'Predicted Stress {label} (GPa)',
        f'Stress {label}: Predicted vs Reference',
        f'stress_{label}_plot.png'
    )
    plot_error_distribution(
        comp_error,
        f'Stress {label} Error (GPa)',
        f'Stress {label} Error Distribution',
        f'stress_{label}_error_distribution.png'
    )

# ----------------------------------------------------------------------
# Total Stress Tensor Plot Colored by Component
stress_palette = 'Dark2'
plot_total_stress_colored(
    reference_stress_ev,
    predicted_stress_ev,
    conversion_factor,
    component_labels,
    palette=stress_palette,
    output_filename='total_stress_colored_plot.png'
)
