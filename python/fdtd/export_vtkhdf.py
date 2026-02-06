##############################################################################
# This file is a part of PFFDTD.
#
# PFFTD is released under the MIT License.
# For details see the LICENSE file.
#
# Copyright 2021 Brian Hamilton.
#
# File name: export_vtkhdf.py
#
# Description: Export simulation outputs to VTKHDF format for ParaView visualization
#
##############################################################################

import numpy as np
from numpy import array as npa
from pathlib import Path
import h5py
import json

def export_to_vtkhdf(data_dir, output_file='wave_propagation.vtm', 
                     use_processed=True, fill_invalid_with_nan=True):
    """
    Export simulation outputs to VTKHDF format for ParaView visualization.
    
    Uses a single file with all time steps for efficient I/O and animation.
    Invalid receivers (clashing with boundaries) are marked with NaN.
    
    Parameters:
    -----------
    data_dir : Path or str
        Directory containing simulation outputs
    output_file : str
        Output VTKHDF filename (should have .vtm extension for ParaView)
    use_processed : bool
        If True, use processed outputs (sim_outs_processed.h5), else raw (sim_outs.h5)
    fill_invalid_with_nan : bool
        If True, fill invalid receiver positions with NaN
    """
    data_dir = Path(data_dir)
    
    print(f'--VTKHDF_EXPORT: Loading data from {data_dir}...')
    
    # Load grid data
    h5f = h5py.File(data_dir / Path('cart_grid.h5'), 'r')
    xv = h5f['xv'][...]
    yv = h5f['yv'][...]
    zv = h5f['zv'][...]
    h = h5f['h'][()]
    h5f.close()
    
    Nx, Ny, Nz = len(xv), len(yv), len(zv)
    
    # Load receiver data
    h5f = h5py.File(data_dir / Path('comms_out.h5'), 'r')
    out_alpha = h5f['out_alpha'][...]
    Nr = h5f['Nr'][()]
    Nt = h5f['Nt'][()]
    h5f.close()
    
    # Load simulation outputs
    if use_processed:
        try:
            h5f = h5py.File(data_dir / Path('sim_outs_processed.h5'), 'r')
            r_out = h5f['r_out_f'][...]  # Processed outputs
            h5f.close()
            print('--VTKHDF_EXPORT: Using processed outputs')
        except:
            print('--VTKHDF_EXPORT: Processed outputs not found, using raw outputs')
            use_processed = False
    
    if not use_processed:
        h5f = h5py.File(data_dir / Path('sim_outs.h5'), 'r')
        u_out = h5f['u_out'][...]
        h5f.close()
        
        print(f'--VTKHDF_EXPORT: u_out shape: {u_out.shape}, out_alpha shape: {out_alpha.shape}')
        print(f'--VTKHDF_EXPORT: Nr: {Nr}, Nt: {Nt}, out_alpha.size: {out_alpha.size}, u_out.size: {u_out.size}')
        
        # Check if file is incomplete (simulation was interrupted)
        expected_size = Nr * Nt
        if u_out.size != expected_size and u_out.size != out_alpha.size * Nt:
            print(f'--VTKHDF_EXPORT: WARNING - File appears incomplete!')
            print(f'--VTKHDF_EXPORT: Expected size: {expected_size} ({expected_size*8/(1024**3):.2f} GB)')
            print(f'--VTKHDF_EXPORT: Actual size: {u_out.size} ({u_out.size*8/(1024**2):.2f} MB)')
            print(f'--VTKHDF_EXPORT: This likely means the simulation was interrupted during write_outputs')
            print(f'--VTKHDF_EXPORT: Try using processed outputs if available, or re-run simulation with fewer receivers')
            raise ValueError(f'Incomplete output file: u_out.size={u_out.size}, expected {expected_size} or {out_alpha.size*Nt}')
        
        # If u_out is transposed, fix it
        if u_out.shape[0] == Nt and u_out.shape[1] == Nr:
            u_out = u_out.T
            print('--VTKHDF_EXPORT: Transposed u_out from (Nt, Nr) to (Nr, Nt)')
        
        # Handle u_out shape - it might be stored with original receivers (before filtering)
        # We need u_out.shape[0] == out_alpha.size (interpolation points) for recombination
        Nr_receivers_from_alpha = out_alpha.shape[0] if out_alpha.ndim == 2 else out_alpha.size // 8
        out_alpha_size = out_alpha.size  # Total interpolation points needed
        
        print(f'--VTKHDF_EXPORT: u_out.shape={u_out.shape}, out_alpha.shape={out_alpha.shape}')
        print(f'--VTKHDF_EXPORT: Need u_out.shape[0] == out_alpha.size ({out_alpha_size}) for recombination')
        
        # Reshape u_out to 2D if needed
        if u_out.ndim == 1:
            Nr_receivers_in_u_out = u_out.size // Nt
            u_out = u_out.reshape(Nr_receivers_in_u_out, Nt)
        elif u_out.ndim == 2:
            Nr_receivers_in_u_out = u_out.shape[0]
        else:
            raise ValueError(f'u_out.ndim ({u_out.ndim}) should be 1 or 2')
        
        # Check if we need to filter u_out to match out_alpha
        if u_out.shape[0] == out_alpha_size:
            # Perfect match - ready for recombination
            print(f'--VTKHDF_EXPORT: u_out matches out_alpha interpolation points')
        elif u_out.shape[0] > out_alpha_size:
            # u_out has more data than needed - likely has original receivers before filtering
            print(f'--VTKHDF_EXPORT: u_out has {u_out.shape[0]} entries, but out_alpha needs {out_alpha_size}')
            print(f'--VTKHDF_EXPORT: Attempting to filter using receiver_grid_metadata.json...')
            
            # Try to load metadata to filter u_out
            metadata_file = data_dir / 'receiver_grid_metadata.json'
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                if 'valid_mask' in metadata:
                    valid_mask = np.array(metadata['valid_mask'], dtype=bool)
                    N_receivers_total = metadata.get('N_receivers_total', len(valid_mask))
                    N_receivers_valid = metadata.get('N_receivers_valid', np.sum(valid_mask))
                    
                    print(f'--VTKHDF_EXPORT: Metadata: N_receivers_total={N_receivers_total}, N_receivers_valid={N_receivers_valid}')
                    
                    # Check if u_out is stored with flattened interpolation points
                    if u_out.shape[0] == N_receivers_total * 8:
                        # u_out has original grid interpolation points - filter to match out_alpha
                        print(f'--VTKHDF_EXPORT: u_out has original grid interpolation points - filtering...')
                        u_out = u_out.reshape(N_receivers_total, 8, Nt)
                        u_out = u_out[valid_mask, :, :]  # Filter receivers
                        u_out = u_out.reshape(N_receivers_valid * 8, Nt)  # Flatten interpolation points
                        if u_out.shape[0] == out_alpha_size:
                            print(f'--VTKHDF_EXPORT: ✓ Successfully filtered u_out to match out_alpha')
                        else:
                            raise ValueError(f'After filtering, u_out has {u_out.shape[0]} interpolation points, '
                                           f'but out_alpha expects {out_alpha_size}')
                    elif u_out.shape[0] == N_receivers_total:
                        # u_out has original receivers (not flattened) - filter and then expand
                        print(f'--VTKHDF_EXPORT: u_out has original receivers - filtering...')
                        u_out = u_out[valid_mask, :]  # Filter receivers
                        # Now we need to expand to interpolation points, but we don't have that data
                        # This case shouldn't happen if u_out is written correctly
                        raise ValueError(f'u_out has receiver-level data ({u_out.shape[0]}), but needs interpolation points ({out_alpha_size}). '
                                       f'This suggests u_out was written incorrectly.')
                    elif abs(u_out.shape[0] - N_receivers_total * 8) <= 10:
                        # Close to original - try to fix
                        print(f'--VTKHDF_EXPORT: u_out is close to original size - attempting to fix...')
                        expected_size = N_receivers_total * 8 * Nt
                        if u_out.size >= expected_size:
                            u_out = u_out[:expected_size].reshape(N_receivers_total, 8, Nt)
                            u_out = u_out[valid_mask, :, :]
                            u_out = u_out.reshape(N_receivers_valid * 8, Nt)
                            if u_out.shape[0] == out_alpha_size:
                                print(f'--VTKHDF_EXPORT: ✓ Fixed and filtered u_out')
                            else:
                                raise ValueError(f'After fixing, u_out has {u_out.shape[0]} interpolation points, '
                                               f'but out_alpha expects {out_alpha_size}')
                        else:
                            raise ValueError(f'Cannot fix: u_out.size={u_out.size} < expected {expected_size}')
                    else:
                        raise ValueError(f'Cannot reconcile: u_out.shape[0]={u_out.shape[0]} doesn\'t match '
                                       f'N_receivers_total*8={N_receivers_total*8} or out_alpha.size={out_alpha_size}')
                else:
                    raise ValueError(f'Metadata found but no valid_mask - cannot filter u_out')
            else:
                raise ValueError(f'u_out.shape[0] ({u_out.shape[0]}) != out_alpha.size ({out_alpha_size}), '
                               f'and no receiver_grid_metadata.json found to reconcile. '
                               f'Re-run sim_setup to ensure consistency.')
        else:
            raise ValueError(f'u_out.shape[0] ({u_out.shape[0]}) < out_alpha.size ({out_alpha_size}) - '
                           f'cannot recombine. This suggests data corruption.')
        
        # Now recombine: u_out has shape (out_alpha.size, Nt), out_alpha has shape (Nr_receivers, 8)
        # The recombination multiplies u_out (interpolation point data) by out_alpha (interpolation weights)
        assert u_out.shape[0] == out_alpha_size, f'u_out.shape[0] ({u_out.shape[0]}) != out_alpha.size ({out_alpha_size})'
        r_out = np.sum((u_out * out_alpha.flat[:][:, None]).reshape((*out_alpha.shape, -1)), axis=1)
        print(f'--VTKHDF_EXPORT: ✓ Recombined to r_out.shape={r_out.shape}')
        
        # Now recombine: u_out has shape (out_alpha.size, Nt), out_alpha has shape (Nr_receivers, 8)
        # The recombination multiplies u_out (interpolation point data) by out_alpha (interpolation weights)
        assert u_out.shape[0] == out_alpha_size, f'u_out.shape[0] ({u_out.shape[0]}) != out_alpha.size ({out_alpha_size})'
        r_out = np.sum((u_out * out_alpha.flat[:][:, None]).reshape((*out_alpha.shape, -1)), axis=1)
        print(f'--VTKHDF_EXPORT: ✓ Recombined to r_out.shape={r_out.shape}')
        
        print('--VTKHDF_EXPORT: Using raw outputs (recombined)')
    
    # r_out now has shape (Nr_receivers, Nt) where Nr_receivers is the number of valid receivers
    Nr_receivers = r_out.shape[0]
    Nt = r_out.shape[-1]
    print(f'--VTKHDF_EXPORT: {Nr_receivers} receivers, {Nt} time steps')
    
    # Load receiver grid metadata if available
    metadata_file = data_dir / 'receiver_grid_metadata.json'
    use_grid = False
    grid_shape = None
    valid_Rxyz = None
    valid_grid_indices = None
    
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            grid_metadata = json.load(f)
        use_grid = grid_metadata.get('use_grid', False)
        if use_grid:
            grid_shape = tuple(grid_metadata['grid_shape'])
            valid_Rxyz = np.array(grid_metadata.get('valid_Rxyz', []))
            valid_grid_indices = np.array(grid_metadata.get('valid_grid_indices', []))
            print(f'--VTKHDF_EXPORT: Grid shape: {grid_shape}')
            print(f'--VTKHDF_EXPORT: {len(valid_Rxyz)} valid receiver positions')
    
    # Load sim constants for time axis
    h5f = h5py.File(data_dir / Path('sim_consts.h5'), 'r')
    Ts = h5f['Ts'][()]
    h5f.close()
    
    time_values = np.arange(Nt) * Ts
    
    # Create grid for visualization
    if use_grid and grid_shape is not None and valid_Rxyz is not None:
        print('--VTKHDF_EXPORT: Creating grid from receiver metadata...')
        
        # Reconstruct full grid including invalid positions
        Nx_grid, Ny_grid, Nz_grid = grid_shape
        
        # Create coordinate arrays for the full grid
        # We need to determine the grid bounds from valid positions
        if len(valid_Rxyz) > 0:
            x_min, y_min, z_min = valid_Rxyz.min(axis=0)
            x_max, y_max, z_max = valid_Rxyz.max(axis=0)
            
            # Extend to full grid bounds (approximate)
            x_range = x_max - x_min
            y_range = y_max - y_min
            z_range = z_max - z_min
            
            # Estimate grid spacing from positions
            if len(valid_Rxyz) > 1:
                # Use median spacing
                dx = np.median(np.diff(np.unique(valid_Rxyz[:, 0])))
                dy = np.median(np.diff(np.unique(valid_Rxyz[:, 1])))
                dz = np.median(np.diff(np.unique(valid_Rxyz[:, 2])))
            else:
                dx = dy = dz = h
            
            # Create full grid
            xv_grid = np.linspace(x_min, x_max, Nx_grid)
            yv_grid = np.linspace(y_min, y_max, Ny_grid)
            zv_grid = np.linspace(z_min, z_max, Nz_grid)
            
            X, Y, Z = np.meshgrid(xv_grid, yv_grid, zv_grid, indexing='ij')
            points = np.c_[X.ravel(), Y.ravel(), Z.ravel()]
            N_points = len(points)
            
            # Create mapping from grid indices to receiver data
            # Map valid receivers to grid positions
            pressure_data = np.full((N_points, Nt), np.nan, dtype=np.float32)
            
            # Create index mapping: grid index -> receiver index
            if len(valid_grid_indices) > 0:
                # Convert grid indices to linear grid index
                grid_linear_indices = (valid_grid_indices[:, 0] * Ny_grid * Nz_grid + 
                                      valid_grid_indices[:, 1] * Nz_grid + 
                                      valid_grid_indices[:, 2])
                
                # Map receiver data to grid positions
                # Note: r_out has shape (Nr_receivers, Nt) where Nr_receivers is number of valid receivers
                # After the fix in sim_setup.py, valid_Rxyz and valid_grid_indices should match r_out exactly
                if len(grid_linear_indices) == r_out.shape[0]:
                    # Perfect match - metadata was updated correctly after final filtering
                    for i, grid_idx in enumerate(grid_linear_indices):
                        if 0 <= grid_idx < N_points:
                            pressure_data[grid_idx, :] = r_out[i, :].astype(np.float32)
                    print(f'--VTKHDF_EXPORT: ✓ Mapped {r_out.shape[0]} receivers to grid positions')
                else:
                    # Mismatch - this shouldn't happen if sim_setup.py was run with the fix
                    # But handle it gracefully for backward compatibility
                    print(f'--VTKHDF_EXPORT: ERROR: Grid index count ({len(grid_linear_indices)}) != receiver count ({r_out.shape[0]})')
                    print(f'--VTKHDF_EXPORT: This indicates metadata was not updated after final filtering.')
                    print(f'--VTKHDF_EXPORT: Re-run sim_setup.py to update metadata, or using spatial matching as fallback...')
                    
                    # Fallback: Use spatial matching to find which receivers correspond
                    from scipy.spatial import cKDTree
                    N_receivers_actual = r_out.shape[0]
                    N_receivers_metadata = len(valid_Rxyz)
                    
                    if N_receivers_metadata >= N_receivers_actual:
                        # Try to match by position (for backward compatibility with old metadata)
                        # This assumes receivers are in approximately the same order
                        print(f'--VTKHDF_EXPORT: Using first {N_receivers_actual} receivers from metadata (fallback mode)')
                        valid_grid_indices_matched = valid_grid_indices[:N_receivers_actual]
                        grid_linear_indices_matched = (valid_grid_indices_matched[:, 0] * Ny_grid * Nz_grid + 
                                                      valid_grid_indices_matched[:, 1] * Nz_grid + 
                                                      valid_grid_indices_matched[:, 2])
                        for i, grid_idx in enumerate(grid_linear_indices_matched):
                            if 0 <= grid_idx < N_points and i < r_out.shape[0]:
                                pressure_data[grid_idx, :] = r_out[i, :].astype(np.float32)
                    else:
                        # Metadata has fewer - use all we have
                        print(f'--VTKHDF_EXPORT: Metadata has fewer receivers ({N_receivers_metadata}) than actual ({N_receivers_actual})')
                        for i, grid_idx in enumerate(grid_linear_indices):
                            if 0 <= grid_idx < N_points and i < r_out.shape[0]:
                                pressure_data[grid_idx, :] = r_out[i, :].astype(np.float32)
            
            print(f'--VTKHDF_EXPORT: Created {N_points} grid points')
            print(f'--VTKHDF_EXPORT: {np.sum(~np.isnan(pressure_data[:, 0]))} points with valid data')
        else:
            use_grid = False
    
    if not use_grid:
        # Create unstructured grid from receiver positions
        # We need to get receiver positions from the grid
        # For now, we'll approximate by using the grid coordinates
        # that correspond to receiver interpolation points
        
        print('--VTKHDF_EXPORT: Creating unstructured grid from receivers...')
        
        # Get receiver positions (approximate - using grid center for each receiver)
        # In practice, you'd want to store actual receiver positions
        # For visualization, we'll create points on the grid
        
        # Create a point cloud - we'll use a subset of grid points
        # that roughly correspond to receiver locations
        # This is a workaround - ideally receiver positions would be stored
        
        # For now, create a simple grid visualization
        # Use a regular subset of the full grid
        stride = max(1, min(Nx, Ny, Nz) // 20)  # Sample grid
        ix_samples = np.arange(0, Nx, stride)
        iy_samples = np.arange(0, Ny, stride)
        iz_samples = np.arange(0, Nz, stride)
        
        X, Y, Z = np.meshgrid(xv[ix_samples], yv[iy_samples], zv[iz_samples], indexing='ij')
        points = np.c_[X.ravel(), Y.ravel(), Z.ravel()]
        N_points = len(points)
        
        print(f'--VTKHDF_EXPORT: Created {N_points} visualization points')
        
        # Map receiver data to visualization points
        # This is approximate - in practice you'd want exact mapping
        # For now, we'll create a simple mapping or use NaN for most points
        
        if fill_invalid_with_nan:
            # Create data array with NaN for invalid positions
            pressure_data = np.full((N_points, Nt), np.nan, dtype=np.float32)
            
            # Map valid receivers (simplified - would need actual position mapping)
            # For now, just use first Nr points if Nr <= N_points
            if Nr <= N_points:
                pressure_data[:Nr, :] = r_out[:Nr, :].astype(np.float32)
            else:
                # Sample receivers
                indices = np.linspace(0, Nr-1, N_points, dtype=int)
                pressure_data = r_out[indices, :].astype(np.float32)
        else:
            # Only include valid receivers
            if Nr <= N_points:
                points = points[:Nr]
                pressure_data = r_out[:Nr, :].astype(np.float32)
                N_points = Nr
            else:
                indices = np.linspace(0, Nr-1, N_points, dtype=int)
                points = points[indices]
                pressure_data = r_out[indices, :].astype(np.float32)
                N_points = len(points)
    
    # Write VTKHDF file following the working example structure
    # Based on pressure_points_vtkhdf.py which successfully works in ParaView
    output_path = data_dir / output_file
    vtkhdf_path = output_path.with_suffix('.vtkhdf')
    
    print(f'--VTKHDF_EXPORT: Writing VTKHDF file with {Nt} time steps...')
    print('--VTKHDF_EXPORT: Using structure from working example...')
    
    # Helper function to append to resizable datasets (matches working example)
    def append_dataset(ds, arr):
        """Append arr along axis 0 to a chunked dataset ds."""
        arr = np.asarray(arr)
        old = ds.shape[0]
        ds.resize(old + arr.shape[0], axis=0)
        ds[old:] = arr
    
    # VTK cell type for a single vertex (a point cell)
    VTK_VERTEX = 1
    
    # Build vertex cells: one cell per point
    # Connectivity for vertices is just [0, 1, 2, ..., N_points-1]
    connectivity = np.arange(N_points, dtype=np.int64)
    
    # Types: one VTK_VERTEX per cell
    types = np.full((N_points,), VTK_VERTEX, dtype=np.uint8)
    
    # Offsets: for vertex cells, each cell uses 1 connectivity id
    # Offsets length must be (#cells + 1): [0,1,2,3,...,N_points]
    offsets = np.arange(N_points + 1, dtype=np.int64)
    
    with h5py.File(vtkhdf_path, 'w') as f:
        root = f.create_group('VTKHDF')
        
        # Header metadata (matches working example)
        root.attrs['Version'] = (2, 3)
        root.attrs['Type'] = 'UnstructuredGrid'
        
        # ---- Static geometry (written ONCE) ----
        # All at root level, NOT in a Cells group (key difference!)
        root.create_dataset('NumberOfPoints', data=(N_points,), dtype='i8')
        root.create_dataset('Points', data=points.astype(np.float32), dtype='f')
        
        root.create_dataset('NumberOfCells', data=(N_points,), dtype='i8')
        root.create_dataset('Types', data=types, dtype='uint8')
        
        root.create_dataset('NumberOfConnectivityIds', data=(N_points,), dtype='i8')
        root.create_dataset('Connectivity', data=connectivity, dtype='i8')
        root.create_dataset('Offsets', data=offsets, dtype='i8')
        
        # ---- Time + offsets bookkeeping ----
        steps = root.create_group('Steps')
        steps.attrs['NSteps'] = Nt  # Set upfront, not incremented
        
        # Time values (one per step) - use float32 like working example
        steps.create_dataset('Values', shape=(0,), maxshape=(None,), dtype='f')
        
        # These offsets tell ParaView where each step starts in flattened arrays.
        # Since geometry is STATIC, these are always 0 for every step.
        steps.create_dataset('PartOffsets', shape=(0,), maxshape=(None,), dtype='i8')
        steps.create_dataset('NumberOfParts', shape=(0,), maxshape=(None,), dtype='i8')
        steps.create_dataset('PointOffsets', shape=(0,), maxshape=(None,), dtype='i8')
        steps.create_dataset('CellOffsets', shape=(0,), maxshape=(None,), dtype='i8')
        steps.create_dataset('ConnectivityIdOffsets', shape=(0,), maxshape=(None,), dtype='i8')
        
        # ---- PointData: Pressure over time ----
        point_data = root.create_group('PointData')
        # Flattened: [t0 all points][t1 all points]...
        point_data.create_dataset('Pressure', shape=(0,), maxshape=(None,), dtype='f')
        
        # Offsets for point-data fields live under Steps/PointDataOffsets/<FieldName>
        pdo = steps.create_group('PointDataOffsets')
        pdo.create_dataset('Pressure', shape=(0,), maxshape=(None,), dtype='i8')
        
        # ---- Write each timestep ----
        for t_idx in range(Nt):
            t = float(time_values[t_idx])
            
            # Record time (use float32 like working example)
            append_dataset(steps['Values'], np.array([t], dtype=np.float32))
            
            # Static mesh offsets: always 0 (key difference - not the actual offset values!)
            append_dataset(steps['PartOffsets'], np.array([0], dtype=np.int64))
            append_dataset(steps['NumberOfParts'], np.array([1], dtype=np.int64))
            append_dataset(steps['PointOffsets'], np.array([0], dtype=np.int64))
            append_dataset(steps['CellOffsets'], np.array([0], dtype=np.int64))
            append_dataset(steps['ConnectivityIdOffsets'], np.array([0], dtype=np.int64))
            
            # PointData offset: where THIS timestep's pressure block starts
            pressure_ds = root['PointData/Pressure']
            start = pressure_ds.shape[0]
            append_dataset(steps['PointDataOffsets/Pressure'], np.array([start], dtype=np.int64))
            
            # Append pressure values for all points at this time
            p = pressure_data[:, t_idx].astype(np.float32)
            append_dataset(pressure_ds, p)
            
            if (t_idx + 1) % 100 == 0 or t_idx == Nt - 1:
                print(f'--VTKHDF_EXPORT: Wrote {t_idx + 1}/{Nt} time steps...')
    
    print(f'--VTKHDF_EXPORT: Wrote VTKHDF file: {vtkhdf_path}')
    print(f'--VTKHDF_EXPORT: File size: {vtkhdf_path.stat().st_size / (1024**2):.2f} MB')
    print(f'--VTKHDF_EXPORT: Open {vtkhdf_path.name} in ParaView - it should recognize the time series automatically')
    print(f'--VTKHDF_EXPORT: Use the time slider in ParaView to animate through all {Nt} time steps')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Export simulation outputs to VTKHDF')
    parser.add_argument('--data_dir', type=str, required=True, help='Simulation data directory')
    parser.add_argument('--output', type=str, default='wave_propagation.vtm', help='Output filename')
    parser.add_argument('--raw', action='store_true', help='Use raw outputs instead of processed')
    args = parser.parse_args()
    
    export_to_vtkhdf(args.data_dir, args.output, use_processed=not args.raw)
