##############################################################################
# This file is a part of PFFDTD.
#
# PFFTD is released under the MIT License.
# For details see the LICENSE file.
#
# Copyright 2021 Brian Hamilton.
#
# File name: sim_setup.py
#
# Description: Function to set up a PFFDTD simulation with single source and multiple receivers
#
##############################################################################

import numpy as np
import json as json
from pathlib import Path
from numpy import array as npa
import h5py
from common.room_geo import RoomGeo
from voxelizer.cart_grid import CartGrid
from voxelizer.vox_grid import VoxGrid
from voxelizer.vox_scene import VoxScene
from fdtd.sim_comms import SimComms
from fdtd.sim_consts import SimConsts
from fdtd.sim_mats import SimMats
from fdtd.rotate_sim_data import rotate_sim_data,sort_sim_data,copy_sim_data,fold_fcc_sim_data
from fdtd.receiver_grid import generate_receiver_grid, filter_receivers_by_boundaries
from air_abs.get_air_absorption import get_air_absorption

def sim_setup(
                #the following are required but using None default so not positional
                insig_type=None, #sig type (see sig_comms.py)
                fmax=None, #fmax for simulation (to set grid spacing)
                PPW=None, #points per wavelength (also to set grid spacing)
                save_folder=None, #where to save .h5 files
                model_json_file=None, #json export of model
                mat_folder=None, #folder where to find .h5 DEF coefficients for wal impedances
                mat_files_dict=None, #dict to link up materials to .h5 mat files
                duration=None, #duration to simulate, in seconds
                #the following are not required
                Tc=20, #temperature in deg C (sets sound speed)
                rh=50, #relative humidity of air (configures air absorption post processing)
                source_num=1, #1-based indexing, source to simulate (in sources.csv)
                save_folder_gpu=None, #folder to save gpu-prepared .h5 data (sorted and rotated and FCC-folded)
                #save_folder_cpu=None,
                draw_vox=False, #draw voxelization 
                draw_backend='mayavi', #default, 'polyscope' better for larger grids
                diff_source=False, #use this for single precision runs
                fcc_flag=False, #to use FCC scheme
                bmin=None, #to set custom scene bounds (useful for open scenes)
                bmax=None, #to set custom scene bounds (useful for open scenes)
                Nvox_est=None, #to manually set number of voxels (for ray-tri intersections) for voxelization
                Nh=None, #to set voxel size in grid pacing (for ray-tri intersections)
                Nprocs=None, #number of processes for multiprocessing, defaults to 80% of cores
                compress=None, #GZIP compress for HDF5, 0 to 9 (fast to slow)
                rot_az_el=[0.,0.], #to rotate the whole scene (including sources/receivers) -- to test robustness of scheme
                use_receiver_grid=False, #generate 3D grid of receivers instead of using JSON receivers
                receiver_grid_spacing=0.1, #receiver grid spacing in meters (default: 0.1m)
                receiver_grid_boundary_margin=0.1, #minimum distance from boundaries in meters (default: 0.1m)
              ):
    assert Tc is not None
    assert rh is not None
    assert source_num > 0
    assert insig_type is not None
    assert fmax is not None
    assert PPW is not None
    assert save_folder is not None
    assert model_json_file is not None
    assert mat_folder is not None
    assert mat_files_dict is not None
    assert duration is not None

    if (bmin is not None) and (bmax is not None):
        #custom bmin/bmax (for open scenes)
        bmin = npa(bmin,dtype=np.float64)
        bmax = npa(bmax,dtype=np.float64)

    #set up room geometry (reads in JSON export, rotates scene)
    room_geo = RoomGeo(model_json_file,az_el=rot_az_el,bmin=bmin,bmax=bmax)
    room_geo.print_stats()

    #sources have to be specified in advance (edit JSON if necessary)
    Sxyz = room_geo.Sxyz[source_num-1] #one source (one-based indexing)
    
    #set up receivers: either from JSON or generate grid
    if use_receiver_grid:
        # Generate 3D grid of receivers matching simulation grid spacing
        # We need grid spacing first, so generate receivers after sim_consts
        Rxyz = None  # Will be set after sim_consts
        receiver_grid_metadata = {'use_grid': True, 'boundary_margin': receiver_grid_boundary_margin}
    else:
        Rxyz = room_geo.Rxyz #many receivers from JSON
        receiver_grid_metadata = {'use_grid': False}

    #some constants for the simulation, in one place
    sim_consts = SimConsts(Tc=Tc,rh=rh,fmax=fmax,PPW=PPW,fcc=fcc_flag)
    sim_consts.save(save_folder)

    #link up the wall materials to impedance datasets
    sim_mats = SimMats(save_folder=save_folder)
    sim_mats.package(mat_files_dict=mat_files_dict,mat_list=room_geo.mat_str,read_folder=mat_folder)

    #set the cartesian grid (also for FCC)
    cart_grid = CartGrid(h=sim_consts.h,offset=3.5,bmin=room_geo.bmin,bmax=room_geo.bmax,fcc=fcc_flag)
    cart_grid.print_stats()
    cart_grid.save(save_folder)

    #set up the voxel grid (volume hierarchy for ray-triangle intersections)
    vox_grid = VoxGrid(room_geo,cart_grid,Nvox_est=Nvox_est,Nh=Nh)
    vox_grid.fill(Nprocs=Nprocs)
    vox_grid.print_stats()

    #'voxelize' the scene (calculate FDTD mesh adjacencies and identify/correct boundary surfaces)
    vox_scene = VoxScene(room_geo,cart_grid,vox_grid,fcc=fcc_flag)
    vox_scene.calc_adj(Nprocs=Nprocs)
    vox_scene.check_adj_full()
    vox_scene.save(save_folder,compress=compress)

    #generate receiver grid if requested (after voxelization so we can filter)
    if use_receiver_grid:
        print(f'--SIM_SETUP: Generating 3D receiver grid with spacing {receiver_grid_spacing}m...')
        Rxyz, grid_shape, grid_indices = generate_receiver_grid(
            bmin=room_geo.bmin,
            bmax=room_geo.bmax,
            spacing=receiver_grid_spacing,
            boundary_margin=receiver_grid_boundary_margin
        )
        receiver_grid_metadata.update({
            'grid_shape': tuple(int(x) for x in grid_shape),  # Convert to tuple of ints
            'grid_indices': grid_indices.tolist(),  # Convert to list for JSON
            'N_receivers_total': int(len(Rxyz)),
            'Rxyz': Rxyz.tolist(),  # Store original positions
            'receiver_grid_spacing': float(receiver_grid_spacing),
            'receiver_grid_boundary_margin': float(receiver_grid_boundary_margin)
        })
        print(f'--SIM_SETUP: Generated {len(Rxyz)} receiver positions in grid shape {grid_shape}')
        print(f'--SIM_SETUP: Receiver grid spacing: {receiver_grid_spacing}m, boundary margin: {receiver_grid_boundary_margin}m')
        
        # Filter receivers by boundaries before preparing
        h5f = h5py.File(save_folder / Path('cart_grid.h5'), 'r')
        xv = h5f['xv'][...]
        yv = h5f['yv'][...]
        zv = h5f['zv'][...]
        h5f.close()
        
        valid_mask, valid_Rxyz = filter_receivers_by_boundaries(
            Rxyz, vox_scene.bn_ixyz, xv, yv, zv, sim_consts.h,
            boundary_margin=receiver_grid_boundary_margin
        )
        
        # Update metadata (convert numpy types to native Python types for JSON)
        receiver_grid_metadata['N_receivers_valid'] = int(np.sum(valid_mask))
        receiver_grid_metadata['N_receivers_filtered'] = int(len(Rxyz) - np.sum(valid_mask))
        receiver_grid_metadata['valid_mask'] = valid_mask.tolist()
        receiver_grid_metadata['valid_Rxyz'] = valid_Rxyz.tolist()
        receiver_grid_metadata['valid_grid_indices'] = grid_indices[valid_mask].tolist()
        
        print(f'--SIM_SETUP: After filtering: {receiver_grid_metadata["N_receivers_valid"]} valid receivers')
        print(f'--SIM_SETUP: Filtered out: {receiver_grid_metadata["N_receivers_filtered"]} receivers')
        
        # Use filtered receivers
        Rxyz = valid_Rxyz

    #set up source/receiver positions and input signals
    sim_comms = SimComms(save_folder=save_folder) #reads from cart_grid
    sim_comms.prepare_source_pts(Sxyz)
    sim_comms.prepare_receiver_pts(Rxyz)
    sim_comms.prepare_source_signals(duration,sig_type=insig_type)
    if diff_source:
        sim_comms.diff_source()
    
    #check that source/receivers don't intersect with boundaries
    # For grid-based receivers, use filter_mode=True to filter interpolation points that clash
    # (prepare_receiver_pts creates interpolation points that might clash even if receiver positions were filtered)
    # For JSON receivers, use filter_mode=False to assert if there are clashes
    if use_receiver_grid:
        receiver_valid_mask = sim_comms.check_for_clashes(vox_scene.bn_ixyz, filter_mode=True)
        # If receivers were filtered, update metadata
        if receiver_valid_mask is not None:
            n_filtered_after = np.sum(~receiver_valid_mask)
            if n_filtered_after > 0:
                print(f'--SIM_SETUP: Additional filtering after prepare_receiver_pts: {n_filtered_after} receivers removed')
                # Update metadata to reflect final valid receivers
                if 'valid_mask' in receiver_grid_metadata:
                    # Combine with previous filtering
                    original_valid = np.array(receiver_grid_metadata['valid_mask'])
                    final_valid = original_valid.copy()
                    # Map receiver_valid_mask back to original grid indices
                    # receiver_valid_mask corresponds to the filtered Rxyz we passed to prepare_receiver_pts
                    valid_indices = np.where(original_valid)[0]
                    if len(valid_indices) == len(receiver_valid_mask):
                        final_valid[valid_indices] = receiver_valid_mask
                        receiver_grid_metadata['valid_mask'] = final_valid.tolist()
                        receiver_grid_metadata['N_receivers_valid'] = int(np.sum(final_valid))
                        receiver_grid_metadata['N_receivers_filtered'] = int(len(original_valid) - np.sum(final_valid))
                        
                        # CRITICAL: Update valid_Rxyz and valid_grid_indices to match the final filtered receivers
                        # This ensures the metadata matches the actual receiver data in r_out
                        valid_Rxyz_array = np.array(receiver_grid_metadata['valid_Rxyz'])
                        valid_grid_indices_array = np.array(receiver_grid_metadata['valid_grid_indices'])
                        
                        # receiver_valid_mask filters the receivers that were passed to prepare_receiver_pts
                        # which corresponds to valid_Rxyz and valid_grid_indices
                        receiver_grid_metadata['valid_Rxyz'] = valid_Rxyz_array[receiver_valid_mask].tolist()
                        receiver_grid_metadata['valid_grid_indices'] = valid_grid_indices_array[receiver_valid_mask].tolist()
                        
                        print(f'--SIM_SETUP: Updated metadata: {len(receiver_grid_metadata["valid_Rxyz"])} final valid receivers')
    else:
        sim_comms.check_for_clashes(vox_scene.bn_ixyz, filter_mode=False)
    
    sim_comms.save(compress=compress)
    
    # Save receiver grid metadata for later VTKHDF export
    if use_receiver_grid:
        metadata_file = Path(save_folder) / 'receiver_grid_metadata.json'
        # Ensure all numpy types are converted to native Python types for JSON
        def convert_to_native(obj):
            """Recursively convert numpy types to native Python types."""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_to_native(value) for key, value in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_native(item) for item in obj]
            else:
                return obj
        
        metadata_json = convert_to_native(receiver_grid_metadata)
        with open(metadata_file, 'w') as f:
            json.dump(metadata_json, f, indent=2)
        print(f'--SIM_SETUP: Saved receiver grid metadata to {metadata_file}')

    #make copy for sorting/rotation for gpu 
    if save_folder_gpu is not None and Path(save_folder_gpu) != Path(save_folder):
        copy_sim_data(save_folder,save_folder_gpu)
    if save_folder_gpu is not None:
        rotate_sim_data(save_folder_gpu)
        if fcc_flag:
            fold_fcc_sim_data(save_folder_gpu)
        sort_sim_data(save_folder_gpu)

    #extra folder for cpu version (if needed for testing)
    #if save_folder_cpu is not None and Path(save_folder_cpu) != Path(save_folder):
        #copy_sim_data(save_folder,save_folder_cpu)
    #if save_folder_cpu is not None:
        #rotate_sim_data(save_folder_cpu)
        #if fcc_flag:
            #fold_fcc_sim_data(save_folder_gpu)
        #sort_sim_data(save_folder_cpu)

    #draw the voxelisation (use polyscope for dense grids)
    if draw_vox:
        room_geo.draw(wireframe=False,backend=draw_backend)
        vox_scene.draw(backend=draw_backend)
        room_geo.show(backend=draw_backend)
