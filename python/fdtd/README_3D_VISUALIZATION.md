# 3D Wave Propagation Visualization Guide

This guide explains how to set up 3D grid-based receivers and export to VTKHDF format for ParaView visualization.

## Overview

The workflow consists of:
1. **Grid Generation**: Generate a 3D grid of receivers matching simulation grid spacing
2. **Boundary Filtering**: Automatically filter receivers that clash with boundary voxels
3. **Simulation**: Run FDTD simulation as normal
4. **Export**: Convert outputs to VTKHDF format for ParaView

## Usage

### Step 1: Setup with Grid-Based Receivers

Modify your setup script to use grid-based receivers:

```python
from sim_setup import sim_setup

sim_setup(
    # ... other parameters ...
    use_receiver_grid=True,  # Enable grid-based receivers
    receiver_grid_spacing=0.1,  # Receiver grid spacing in meters (default: 0.1m)
    receiver_grid_boundary_margin=0.1,  # Minimum distance from boundaries in meters (default: 0.1m)
    # ... rest of parameters ...
)
```

### Step 2: Run Simulation

Run the simulation as normal (Python or CUDA engine).

### Step 3: Post-Process Outputs

Process the outputs with filtering (resampling is optional and only needed for WAV files):

```bash
python -m fdtd.process_outputs --data_dir='../data/sim_data/mv_fcc/gpu/' \
    --fcut_lowpass 2500.0 --N_order_lowpass=8 --symmetric \
    --fcut_lowcut 10.0 --N_order_lowcut=4 --air_abs_filter='stokes' \
    --plot
```

**Note:** Resampling is now optional and only happens if:
- You explicitly pass `--resample_Fs <rate>` 
- Or if you use `--save_wav` (auto-resamples to 48kHz for audio)

For VTKHDF export, you don't need resampling - the raw simulation sample rate is fine.

### Step 4: Export to VTKHDF

Export to VTKHDF format for ParaView (follows official VTKHDF specification):

```bash
python -m fdtd.export_vtkhdf --data_dir='../data/sim_data/mv_fcc/gpu/' \
    --output='wave_propagation.vtkhdf'
```

Or from Python:

```python
from fdtd.export_vtkhdf import export_to_vtkhdf

export_to_vtkhdf(
    data_dir='../data/sim_data/mv_fcc/gpu/',
    output_file='wave_propagation.vtkhdf',
    use_processed=True,
    fill_invalid_with_nan=True
)
```

**Note:** The output file will have `.vtkhdf` extension (or `.hdf`), which ParaView recognizes natively.

### Step 5: Generate Coordinate Mapping (Optional)

To understand coordinate transformations:

```bash
python -m fdtd.coordinate_mapping --data_dir='../data/sim_data/mv_fcc/gpu/' \
    --output='coordinate_mapping.csv'
```

This creates a CSV file documenting:
- Real-space coordinates (x, y, z in meters)
- Voxel indices (ix, iy, iz)
- 1D linear indices (ixyz)
- Verification of coordinate transformations

## Coordinate System Reference

### 1D Index Formula
```
ixyz = ix * Ny * Nz + iy * Nz + iz
```

Where:
- `ix, iy, iz` are voxel indices (0-based)
- `Ny, Nz` are grid dimensions
- `z` is the contiguous dimension

### Inverse Transformation
```python
iz = ixyz % Nz
iy = (ixyz - iz) // Nz % Ny
ix = ((ixyz - iz) // Nz - iy) // Ny
```

## File Formats

### Receiver Grid Metadata (`receiver_grid_metadata.json`)

Contains:
- `use_grid`: Boolean indicating grid-based receivers
- `grid_shape`: Tuple (Nx, Ny, Nz) of grid dimensions
- `N_receivers_total`: Total receivers generated
- `N_receivers_valid`: Valid receivers after filtering
- `N_receivers_filtered`: Number filtered out
- `valid_Rxyz`: List of valid receiver positions
- `valid_grid_indices`: List of grid indices for valid receivers

### VTKHDF Output

The exporter creates a single `.vtkhdf` file following the [official VTKHDF specification](https://www.kitware.com/how-to-write-time-dependent-data-in-vtkhdf-files/) for time-dependent data.

**File Structure:**
- **VTKHDF root group** with Version 2.0 and Type 'UnstructuredGrid'
- **Points**: Grid point coordinates (static, same for all time steps)
- **Cells**: Cell connectivity (static, vertex cells for each point)
- **PointData/Pressure**: Pressure data flattened across all time steps
- **Steps group**: Time-dependent metadata including:
  - `Values`: Time values for each step
  - `PointOffsets`: Offsets for reading points (static geometry)
  - `PointDataOffsets/Pressure`: Offsets for reading pressure data at each time step
  - `CellOffsets` and `ConnectivityIdOffsets`: Offsets for topology (static)

**Key Features:**
- Single file contains all time steps (efficient I/O)
- Uses HDF5 compression (GZIP level 3)
- Follows official VTKHDF 2.0 specification
- ParaView recognizes time series automatically

Invalid receivers (clashing with boundaries) are marked with NaN.

## ParaView Visualization

1. **Open ParaView**
2. **File → Open** → Select the `.vtkhdf` file
3. ParaView should automatically recognize it as a time series
4. **Use the time slider** at the top to animate through all time steps
5. **Apply filters as needed:**
   - **Threshold**: Remove NaN values (set min/max to exclude NaN)
   - **Glyph**: Visualize as arrows/spheres
   - **Slice**: Create cross-sections
   - **Contour**: Isosurfaces
   - **Volume Rendering**: For 3D visualization

**Time Series:**
- The file contains all time steps in a single file
- ParaView's time slider will show all available time steps
- You can play/pause the animation or scrub through time manually

## Notes

- Receiver grid spacing is configurable in meters (default: 0.1m), independent of simulation grid spacing
- Boundary margin defaults to 0.1m (configurable in meters)
- Invalid receivers are marked with NaN (not removed) for visualization
- Coordinate mapping utility helps debug indexing issues
- VTKHDF uses single file with all time steps for efficient I/O

## Troubleshooting

### Receivers Clashing with Boundaries

If you see errors about receivers clashing:
- Increase `receiver_grid_boundary_margin`
- Check that room bounds are correct
- Verify grid spacing is appropriate

### Coordinate Mapping Issues

If you have indexing problems:
- Run `coordinate_mapping.py` to generate reference CSV
- Check that 1D indices match expected formula
- Verify grid dimensions match metadata

### VTKHDF Import Issues

If ParaView can't read the file:
- **Check ParaView version**: VTKHDF support requires ParaView 5.12.0+ or VTK 9.3.0+
- **File extension**: Ensure the file has `.vtkhdf` or `.hdf` extension
- **Verify file structure**: Use `h5dump` or HDFView to inspect the file structure
- **Check Steps group**: Verify that `Steps/Values` contains time values and `Steps/PointDataOffsets/Pressure` contains offsets
- **Update ParaView**: If using an older version, update to the latest release

The exporter follows the official VTKHDF specification, so it should work with compatible ParaView versions.
