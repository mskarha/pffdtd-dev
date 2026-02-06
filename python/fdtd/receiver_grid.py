##############################################################################
# This file is a part of PFFDTD.
#
# PFFTD is released under the MIT License.
# For details see the LICENSE file.
#
# Copyright 2021 Brian Hamilton.
#
# File name: receiver_grid.py
#
# Description: Generate 3D grid of receivers for visualization
#
##############################################################################

import numpy as np
from numpy import array as npa
from pathlib import Path
import h5py

def generate_receiver_grid(bmin, bmax, spacing, boundary_margin=0.1):
    """
    Generate a 3D grid of receiver positions with specified spacing.
    
    Parameters:
    -----------
    bmin : array-like, shape (3,)
        Minimum bounds of the room (x, y, z) in meters
    bmax : array-like, shape (3,)
        Maximum bounds of the room (x, y, z) in meters
    spacing : float
        Grid spacing in meters (default: 0.1m)
    boundary_margin : float
        Minimum distance from boundaries in meters (default: 0.1m)
        
    Returns:
    --------
    Rxyz : array, shape (N, 3)
        Receiver positions in real space (meters)
    grid_shape : tuple (Nx, Ny, Nz)
        Shape of the 3D grid
    grid_indices : array, shape (N, 3)
        Grid indices (ix, iy, iz) for each receiver
    """
    bmin = npa(bmin, dtype=np.float64)
    bmax = npa(bmax, dtype=np.float64)
    
    # Apply boundary margin (in meters)
    bmin_grid = bmin + boundary_margin
    bmax_grid = bmax - boundary_margin
    
    # Ensure valid bounds
    assert np.all(bmax_grid > bmin_grid), "Boundary margin too large for room size"
    
    # Generate grid positions with specified spacing
    # Align with grid that would start at integer multiples of spacing
    x_start = np.ceil(bmin_grid[0] / spacing) * spacing
    y_start = np.ceil(bmin_grid[1] / spacing) * spacing
    z_start = np.ceil(bmin_grid[2] / spacing) * spacing
    
    x_end = np.floor(bmax_grid[0] / spacing) * spacing
    y_end = np.floor(bmax_grid[1] / spacing) * spacing
    z_end = np.floor(bmax_grid[2] / spacing) * spacing
    
    # Generate grid
    xv = np.arange(x_start, x_end + spacing/2, spacing)
    yv = np.arange(y_start, y_end + spacing/2, spacing)
    zv = np.arange(z_start, z_end + spacing/2, spacing)
    
    Nx, Ny, Nz = len(xv), len(yv), len(zv)
    
    # Create meshgrid and flatten
    X, Y, Z = np.meshgrid(xv, yv, zv, indexing='ij')
    
    Rxyz = np.c_[X.ravel(), Y.ravel(), Z.ravel()]
    grid_shape = (Nx, Ny, Nz)
    
    # Create grid indices
    ix, iy, iz = np.meshgrid(np.arange(Nx), np.arange(Ny), np.arange(Nz), indexing='ij')
    grid_indices = np.c_[ix.ravel(), iy.ravel(), iz.ravel()]
    
    return Rxyz, grid_shape, grid_indices

def filter_receivers_by_boundaries(Rxyz, bn_ixyz, xv, yv, zv, h, boundary_margin=0.1):
    """
    Filter receivers that clash with boundary voxels.
    
    Parameters:
    -----------
    Rxyz : array, shape (N, 3)
        Receiver positions in real space
    bn_ixyz : array
        Boundary node 1D indices
    xv, yv, zv : array
        Grid coordinate vectors (simulation grid)
    h : float
        Simulation grid spacing (for checking boundaries)
    boundary_margin : float
        Additional margin in meters (default: 0.1m)
        
    Returns:
    --------
    valid_mask : array, shape (N,), dtype=bool
        Boolean mask indicating valid receivers
    valid_Rxyz : array, shape (M, 3)
        Valid receiver positions
    """
    from common.myfuncs import ind2sub3d
    
    Nx, Ny, Nz = len(xv), len(yv), len(zv)
    
    # Convert boundary indices to (ix, iy, iz)
    bn_set = set(bn_ixyz.flat[:])
    
    # Also mark neighbors within boundary_margin (convert meters to grid cells)
    margin_cells = int(np.ceil(boundary_margin / h))
    neighbor_offsets = []
    for dx in range(-margin_cells, margin_cells + 1):
        for dy in range(-margin_cells, margin_cells + 1):
            for dz in range(-margin_cells, margin_cells + 1):
                if dx*dx + dy*dy + dz*dz <= margin_cells*margin_cells:
                    neighbor_offsets.append((dx, dy, dz))
    
    # Expand boundary set to include neighbors
    expanded_bn_set = set(bn_set)
    for bn_idx in bn_set:
        ix, iy, iz = ind2sub3d(bn_idx, Nx, Ny, Nz)
        for dx, dy, dz in neighbor_offsets:
            ix_n = ix + dx
            iy_n = iy + dy
            iz_n = iz + dz
            if 0 <= ix_n < Nx and 0 <= iy_n < Ny and 0 <= iz_n < Nz:
                idx_n = ix_n * Ny * Nz + iy_n * Nz + iz_n
                expanded_bn_set.add(idx_n)
    
    # Check each receiver
    valid_mask = np.ones(Rxyz.shape[0], dtype=bool)
    
    for i in range(Rxyz.shape[0]):
        # Find nearest grid point
        ix = np.flatnonzero(xv >= Rxyz[i, 0])[0] if np.any(xv >= Rxyz[i, 0]) else Nx - 1
        iy = np.flatnonzero(yv >= Rxyz[i, 1])[0] if np.any(yv >= Rxyz[i, 1]) else Ny - 1
        iz = np.flatnonzero(zv >= Rxyz[i, 2])[0] if np.any(zv >= Rxyz[i, 2]) else Nz - 1
        
        ix = np.clip(ix, 0, Nx - 1)
        iy = np.clip(iy, 0, Ny - 1)
        iz = np.clip(iz, 0, Nz - 1)
        
        # Convert to 1D index
        ixyz = ix * Ny * Nz + iy * Nz + iz
        
        # Check if this or any of its 8 interpolation neighbors clash
        # (receivers use trilinear interpolation with 8 neighbors)
        neighbors = [
            (ix, iy, iz),
            (ix-1, iy, iz), (ix, iy-1, iz), (ix, iy, iz-1),
            (ix-1, iy-1, iz), (ix-1, iy, iz-1), (ix, iy-1, iz-1),
            (ix-1, iy-1, iz-1)
        ]
        
        clashes = False
        for nx, ny, nz in neighbors:
            if 0 <= nx < Nx and 0 <= ny < Ny and 0 <= nz < Nz:
                n_idx = nx * Ny * Nz + ny * Nz + nz
                if n_idx in expanded_bn_set:
                    clashes = True
                    break
        
        if clashes:
            valid_mask[i] = False
    
    valid_Rxyz = Rxyz[valid_mask]
    
    return valid_mask, valid_Rxyz
