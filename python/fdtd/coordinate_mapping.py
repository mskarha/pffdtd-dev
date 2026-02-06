##############################################################################
# This file is a part of PFFDTD.
#
# PFFTD is released under the MIT License.
# For details see the LICENSE file.
#
# Copyright 2021 Brian Hamilton.
#
# File name: coordinate_mapping.py
#
# Description: Utility to document and understand coordinate transformations
# between real space (xyz), voxel space (ix,iy,iz), and 1D linear indices
#
##############################################################################

import numpy as np
from numpy import array as npa
from pathlib import Path
import h5py
import csv
from common.myfuncs import ind2sub3d

def xyz_to_voxel_indices(xyz, xv, yv, zv, h):
    """
    Convert real-space coordinates to voxel indices.
    
    Parameters:
    -----------
    xyz : array-like, shape (3,) or (N, 3)
        Real-space coordinates in meters
    xv, yv, zv : array-like
        Grid coordinate vectors (from cart_grid)
    h : float
        Grid spacing
        
    Returns:
    --------
    ix, iy, iz : int or array
        Voxel indices
    """
    xyz = np.atleast_2d(xyz)
    Nx, Ny, Nz = len(xv), len(yv), len(zv)
    
    ix = np.empty(xyz.shape[0], dtype=np.int64)
    iy = np.empty(xyz.shape[0], dtype=np.int64)
    iz = np.empty(xyz.shape[0], dtype=np.int64)
    
    for i in range(xyz.shape[0]):
        # Find first grid point >= position (same logic as sim_comms)
        ix[i] = np.flatnonzero(xv >= xyz[i, 0])[0] if np.any(xv >= xyz[i, 0]) else Nx - 1
        iy[i] = np.flatnonzero(yv >= xyz[i, 1])[0] if np.any(yv >= xyz[i, 1]) else Ny - 1
        iz[i] = np.flatnonzero(zv >= xyz[i, 2])[0] if np.any(zv >= xyz[i, 2]) else Nz - 1
        
        # Clamp to valid range
        ix[i] = np.clip(ix[i], 0, Nx - 1)
        iy[i] = np.clip(iy[i], 0, Ny - 1)
        iz[i] = np.clip(iz[i], 0, Nz - 1)
    
    if xyz.shape[0] == 1:
        return ix[0], iy[0], iz[0]
    return ix, iy, iz

def voxel_to_1d_index(ix, iy, iz, Ny, Nz):
    """
    Convert voxel indices (ix, iy, iz) to 1D linear index.
    
    Formula: ixyz = ix*Ny*Nz + iy*Nz + iz
    (z is contiguous dimension)
    
    Parameters:
    -----------
    ix, iy, iz : int or array
        Voxel indices
    Ny, Nz : int
        Grid dimensions
        
    Returns:
    --------
    ixyz : int or array
        1D linear index
    """
    return ix * Ny * Nz + iy * Nz + iz

def write_coordinate_mapping_csv(save_folder, output_file='coordinate_mapping.csv'):
    """
    Write a CSV file documenting coordinate transformations for all grid points.
    
    Parameters:
    -----------
    save_folder : Path or str
        Folder containing cart_grid.h5
    output_file : str
        Output CSV filename
    """
    save_folder = Path(save_folder)
    
    # Load grid data
    h5f = h5py.File(save_folder / Path('cart_grid.h5'), 'r')
    xv = h5f['xv'][...]
    yv = h5f['yv'][...]
    zv = h5f['zv'][...]
    h = h5f['h'][()]
    h5f.close()
    
    Nx, Ny, Nz = len(xv), len(yv), len(zv)
    
    # Generate all grid points
    print(f'Writing coordinate mapping for {Nx*Ny*Nz} grid points...')
    
    with open(save_folder / output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            'ix', 'iy', 'iz',  # Voxel indices
            'x_real', 'y_real', 'z_real',  # Real-space coordinates (meters)
            'ixyz_1d',  # 1D linear index
            'ixyz_1d_verify'  # Verification (recomputed from ix,iy,iz)
        ])
        
        for ix in range(Nx):
            for iy in range(Ny):
                for iz in range(Nz):
                    x_real = xv[ix]
                    y_real = yv[iy]
                    z_real = zv[iz]
                    
                    # Compute 1D index
                    ixyz_1d = voxel_to_1d_index(ix, iy, iz, Ny, Nz)
                    
                    # Verify by converting back
                    ix_verify, iy_verify, iz_verify = ind2sub3d(ixyz_1d, Nx, Ny, Nz)
                    ixyz_1d_verify = voxel_to_1d_index(ix_verify, iy_verify, iz_verify, Ny, Nz)
                    
                    writer.writerow([
                        ix, iy, iz,
                        f'{x_real:.6f}', f'{y_real:.6f}', f'{z_real:.6f}',
                        ixyz_1d,
                        ixyz_1d_verify
                    ])
    
    print(f'Coordinate mapping written to {save_folder / output_file}')
    print(f'Total grid points: {Nx*Ny*Nz}')
    print(f'Grid dimensions: Nx={Nx}, Ny={Ny}, Nz={Nz}')
    print(f'Grid spacing: h={h:.6f} m')
    print(f'1D index formula: ixyz = ix*Ny*Nz + iy*Nz + iz')
    print(f'Inverse formula: iz = ixyz % Nz, iy = (ixyz - iz)//Nz % Ny, ix = ((ixyz - iz)//Nz - iy)//Ny')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generate coordinate mapping CSV')
    parser.add_argument('--data_dir', type=str, required=True, help='Simulation data directory')
    parser.add_argument('--output', type=str, default='coordinate_mapping.csv', help='Output CSV filename')
    args = parser.parse_args()
    
    write_coordinate_mapping_csv(args.data_dir, args.output)
