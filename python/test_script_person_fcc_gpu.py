##############################################################################
# This file is a part of PFFDTD.
#
# PFFTD is released under the MIT License.
# For details see the LICENSE file.
#
# Copyright 2021 Brian Hamilton.
#
# File name: test_script_MV_fcc_viz.py
#
# Description: this shows a simple setup with FCC scheme, for a larger single-precision GPU run (<12GB VRAM)
#
##############################################################################
from sim_setup import sim_setup

sim_setup(
    model_json_file='../data/models/person/model_export.json',
    draw_backend='mayavi',
    mat_folder='../data/materials',
    source_num=1,
    insig_type='impulse',
    diff_source=True,
    mat_files_dict={
                    'rp_mei_posed_001_30k_mat': 'mv_plasterboard.h5',
                    }, #see build_mats.py to set these material impedances from absorption data
    duration=0.03,
    Tc=20,
    rh=50,
    fcc_flag=True,
    PPW=7.7, #for 1% phase velocity error at fmax
    fmax=1000.0,
    save_folder='../data/sim_data/person/gpu',
    save_folder_gpu='../data/sim_data/person/gpu',
    compress=3, #apply level-3 GZIP compression to larger h5 files
    use_receiver_grid=True, #generate 3D grid of receivers for visualization
    receiver_grid_spacing=0.1, #receiver grid spacing in meters (0.2m gives ~10k receivers, 0.1m gives ~500k - too many!)
    receiver_grid_boundary_margin=0.01, #minimum distance from boundaries in meters (default: 0.1m)
)

#then from '../data/sim_data/person/gpu' folder, run (relative path for default folder structure):
#   ../../../../c_cuda/fdtd_main_gpu_single.x

#then post-process with something like:
#   python -m fdtd.process_outputs --data_dir='../data/sim_data/person/gpu/' --fcut_lowpass 2500.0 --N_order_lowpass=8 --symmetric --fcut_lowcut 10.0 --N_order_lowcut=4 --air_abs_filter='stokes' --save_wav --plot
