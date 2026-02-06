#!/usr/bin/env python3
"""
Calculate memory requirements for PFFDTD simulation to help select AWS EC2 instances.

Usage:
    python calculate_memory_requirements.py --spacing 0.2 --duration 0.03 --fmax 1700 --PPW 7.7
"""

import argparse

def calculate_receiver_count(room_size, spacing, boundary_margin=0.1):
    """
    Estimate number of receivers in a 3D grid.
    Uses the same logic as receiver_grid.py:generate_receiver_grid()
    
    Args:
        room_size: Tuple (width, length, height) in meters
        spacing: Receiver grid spacing in meters
        boundary_margin: Minimum distance from boundaries in meters (default: 0.1)
    
    Returns:
        Approximate number of receivers
    """
    w, l, h = room_size
    bmin = [0, 0, 0]
    bmax = [w, l, h]
    
    # Apply boundary margin (matches receiver_grid.py)
    bmin_grid = [bmin[i] + boundary_margin for i in range(3)]
    bmax_grid = [bmax[i] - boundary_margin for i in range(3)]
    
    # Generate grid positions (matches receiver_grid.py lines 56-67)
    import math
    x_start = math.ceil(bmin_grid[0] / spacing) * spacing
    y_start = math.ceil(bmin_grid[1] / spacing) * spacing
    z_start = math.ceil(bmin_grid[2] / spacing) * spacing
    
    x_end = math.floor(bmax_grid[0] / spacing) * spacing
    y_end = math.floor(bmax_grid[1] / spacing) * spacing
    z_end = math.floor(bmax_grid[2] / spacing) * spacing
    
    # Count grid points (matches receiver_grid.py line 65-67)
    n_x = int((x_end - x_start) / spacing) + 1
    n_y = int((y_end - y_start) / spacing) + 1
    n_z = int((z_end - z_start) / spacing) + 1
    
    return n_x * n_y * n_z

def calculate_time_steps(duration, fmax, PPW, Tc=20, fcc=False):
    """
    Calculate number of time steps using the same formula as sim_consts.py.
    
    Args:
        duration: Simulation duration in seconds
        fmax: Maximum frequency in Hz
        PPW: Points per wavelength
        Tc: Temperature in Celsius (default: 20)
        fcc: Whether FCC scheme is used (default: False)
    
    Returns:
        Number of time steps (Nt)
    """
    # Speed of sound (matches sim_consts.py)
    c = 343.2 * (Tc / 20.0) ** 0.5
    
    # Grid spacing based on PPW (matches sim_consts.py line 49)
    h = c / (fmax * PPW)
    
    # CFL number (matches sim_consts.py lines 29-40)
    if fcc:
        l = 1.0
    else:
        l = (1.0 / 3.0) ** 0.5
    
    # Back off to remove nyquist mode (matches sim_consts.py line 39)
    l *= 0.999
    l2 = l * l
    
    # Time step (matches sim_consts.py line 50)
    Ts = h / c * l
    
    # Number of time steps (matches sim_comms.py line 67)
    import math
    Nt = int(math.ceil(duration / Ts))
    
    return Nt

def calculate_memory_requirements(Nr, Nt, precision='double'):
    """
    Calculate memory requirements for simulation.
    
    Args:
        Nr: Number of receivers
        Nt: Number of time steps
        precision: 'single' or 'double' (default: 'double')
    
    Returns:
        Dictionary with memory breakdown
    """
    bytes_per_sample = 8 if precision == 'double' else 4
    
    # Main u_out array (receiver outputs)
    u_out_size_bytes = Nr * Nt * bytes_per_sample
    u_out_size_gb = u_out_size_bytes / (1024**3)
    
    # GPU memory (per GPU, approximate)
    # Based on typical simulation: ~60 MB per GPU for grid arrays
    gpu_mem_per_device_mb = 60
    
    # Host memory for other arrays (approximate)
    # Includes: in_sigs, boundary arrays, material arrays, etc.
    # Rough estimate: 20-30% of u_out size
    host_overhead_gb = u_out_size_gb * 0.25
    
    # Total host memory needed
    total_host_mem_gb = u_out_size_gb + host_overhead_gb
    
    return {
        'Nr': Nr,
        'Nt': Nt,
        'u_out_size_gb': u_out_size_gb,
        'host_overhead_gb': host_overhead_gb,
        'total_host_mem_gb': total_host_mem_gb,
        'gpu_mem_per_device_mb': gpu_mem_per_device_mb,
        'precision': precision,
        'bytes_per_sample': bytes_per_sample
    }

def recommend_instance(total_mem_gb):
    """
    Recommend AWS EC2 instance based on memory requirements.
    
    Args:
        total_mem_gb: Total host memory needed in GB
    
    Returns:
        List of recommended instances
    """
    recommendations = []
    
    if total_mem_gb <= 16:
        recommendations.append({
            'instance': 'g5.xlarge',
            'ram_gb': 16,
            'gpus': 1,
            'on_demand_hr': 0.51,
            'spot_hr': 0.15,
            'suitable': total_mem_gb <= 12
        })
    elif total_mem_gb <= 32:
        recommendations.append({
            'instance': 'g5.2xlarge',
            'ram_gb': 32,
            'gpus': 1,
            'on_demand_hr': 1.01,
            'spot_hr': 0.30,
            'suitable': total_mem_gb <= 28
        })
    elif total_mem_gb <= 192:
        recommendations.append({
            'instance': 'g5.12xlarge',
            'ram_gb': 192,
            'gpus': 4,
            'on_demand_hr': 5.67,
            'spot_hr': 1.70,
            'suitable': total_mem_gb <= 160
        })
    else:
        recommendations.append({
            'instance': 'g5.24xlarge',
            'ram_gb': 384,
            'gpus': 4,
            'on_demand_hr': 10.14,
            'spot_hr': 3.04,
            'suitable': total_mem_gb <= 320
        })
    
    return recommendations

def main():
    parser = argparse.ArgumentParser(
        description='Calculate memory requirements for PFFDTD simulation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Calculate for default room (6x4x3.5m) with 0.2m spacing
  python calculate_memory_requirements.py --spacing 0.2 --duration 0.03 --fmax 1700 --PPW 7.7
  
  # Calculate for custom room size
  python calculate_memory_requirements.py --spacing 0.1 --duration 0.03 --fmax 1700 --PPW 7.7 --room-size 10 8 4
  
  # Use single precision
  python calculate_memory_requirements.py --spacing 0.2 --duration 0.03 --fmax 1700 --PPW 7.7 --precision single
        """
    )
    
    parser.add_argument('--spacing', type=float, default=0.2,
                       help='Receiver grid spacing in meters (default: 0.2)')
    parser.add_argument('--boundary-margin', type=float, default=0.1,
                       help='Boundary margin in meters (default: 0.1)')
    parser.add_argument('--duration', type=float, default=None,
                       help='Simulation duration in seconds (required if Nt not provided)')
    parser.add_argument('--fmax', type=float, default=None,
                       help='Maximum frequency in Hz (required if Nt not provided)')
    parser.add_argument('--PPW', type=float, default=None,
                       help='Points per wavelength (required if Nt not provided)')
    parser.add_argument('--room-size', type=float, nargs=3, default=[6.0, 4.0, 3.5],
                       metavar=('WIDTH', 'LENGTH', 'HEIGHT'),
                       help='Room dimensions in meters: width length height (default: 6 4 3.5)')
    parser.add_argument('--precision', choices=['single', 'double'], default='double',
                       help='Floating point precision (default: double)')
    parser.add_argument('--Tc', type=float, default=20.0,
                       help='Temperature in Celsius (default: 20)')
    parser.add_argument('--fcc', action='store_true',
                       help='Use FCC scheme (affects CFL number and time step)')
    parser.add_argument('--Nr', type=int, default=None,
                       help='Override: specify exact number of receivers (skips calculation)')
    parser.add_argument('--Nt', type=int, default=None,
                       help='Override: specify exact number of time steps (skips calculation)')
    parser.add_argument('--data-dir', type=str, default=None,
                       help='Read actual Nr and Nt from simulation data directory (comms_out.h5)')
    
    args = parser.parse_args()
    
    # Try to read from data directory if provided
    if args.data_dir is not None:
        try:
            import h5py
            from pathlib import Path
            data_path = Path(args.data_dir)
            comms_file = data_path / 'comms_out.h5'
            if comms_file.exists():
                with h5py.File(comms_file, 'r') as h5f:
                    Nr_from_file = int(h5f['Nr'][()])
                    Nt_from_file = int(h5f['Nt'][()])
                if args.Nr is None:
                    args.Nr = Nr_from_file
                    print(f"Read Nr from {comms_file}: {Nr_from_file:,}")
                if args.Nt is None:
                    args.Nt = Nt_from_file
                    print(f"Read Nt from {comms_file}: {Nt_from_file:,}")
            else:
                print(f"Warning: {comms_file} not found, using calculated values")
        except Exception as e:
            print(f"Warning: Could not read from data directory: {e}")
    
    # Calculate or use provided Nr
    if args.Nr is not None:
        Nr = args.Nr
        print(f"Using provided Nr: {Nr:,}")
    else:
        if args.spacing is None:
            parser.error("--spacing required when --Nr not provided")
        Nr = calculate_receiver_count(args.room_size, args.spacing, args.boundary_margin)
        print(f"Estimated Nr from room size {args.room_size}m, spacing {args.spacing}m, margin {args.boundary_margin}m: {Nr:,}")
        print(f"  NOTE: Actual Nr may differ due to boundary filtering. Use --Nr to specify exact value.")
    
    # Calculate or use provided Nt
    if args.Nt is not None:
        Nt = args.Nt
        print(f"Using provided Nt: {Nt:,}")
    else:
        if args.duration is None or args.fmax is None or args.PPW is None:
            parser.error("--duration, --fmax, and --PPW required when --Nt not provided")
        Nt = calculate_time_steps(args.duration, args.fmax, args.PPW, args.Tc, args.fcc)
        scheme = "FCC" if args.fcc else "Cartesian"
        print(f"Calculated Nt: {Nt:,} time steps (scheme: {scheme}, Tc: {args.Tc}°C)")
    print()
    
    # Calculate memory requirements
    mem_req = calculate_memory_requirements(Nr, Nt, args.precision)
    
    print("=" * 60)
    print("MEMORY REQUIREMENTS")
    print("=" * 60)
    print(f"Receivers (Nr):           {mem_req['Nr']:,}")
    print(f"Time steps (Nt):          {mem_req['Nt']:,}")
    print(f"Precision:                {mem_req['precision']}")
    print(f"Bytes per sample:         {mem_req['bytes_per_sample']}")
    print()
    print(f"u_out array:              {mem_req['u_out_size_gb']:.2f} GB")
    print(f"Host overhead (est.):     {mem_req['host_overhead_gb']:.2f} GB")
    print(f"Total host memory:        {mem_req['total_host_mem_gb']:.2f} GB")
    print(f"GPU memory (per device):   {mem_req['gpu_mem_per_device_mb']:.2f} MB")
    print()
    print("⚠  IMPORTANT: If actual memory differs, check:")
    print("   - Actual Nr from simulation output (use --Nr to override)")
    print("   - Actual Nt from simulation output (use --Nt to override)")
    print("   - Room bounds (bmin/bmax) may differ from room_size")
    print("   - Boundary filtering may remove fewer receivers than expected")
    print()
    
    # Get recommendations
    recommendations = recommend_instance(mem_req['total_host_mem_gb'])
    
    print("=" * 60)
    print("RECOMMENDED AWS EC2 INSTANCES")
    print("=" * 60)
    for rec in recommendations:
        status = "✓ SUITABLE" if rec['suitable'] else "⚠ TIGHT FIT"
        print(f"\n{status}: {rec['instance']}")
        print(f"  RAM:           {rec['ram_gb']} GB")
        print(f"  GPUs:          {rec['gpus']}")
        print(f"  On-demand:     ${rec['on_demand_hr']:.2f}/hour")
        print(f"  Spot (est.):   ${rec['spot_hr']:.2f}/hour (~70% savings)")
        if not rec['suitable']:
            print(f"  ⚠ Warning: Memory requirement ({mem_req['total_host_mem_gb']:.2f} GB) is close to instance limit ({rec['ram_gb']} GB)")
    
    print()
    print("=" * 60)
    print("COST OPTIMIZATION TIPS")
    print("=" * 60)
    print("1. Use Spot instances for 70-90% cost savings")
    print("2. Increase receiver_grid_spacing to reduce memory:")
    print(f"   - Current spacing: {args.spacing}m → {Nr:,} receivers")
    if args.spacing >= 0.1:
        new_spacing = args.spacing * 2
        new_Nr = calculate_receiver_count(args.room_size, new_spacing)
        new_mem = calculate_memory_requirements(new_Nr, Nt, args.precision)
        print(f"   - Try {new_spacing}m → {new_Nr:,} receivers → {new_mem['total_host_mem_gb']:.2f} GB")
    print("3. Consider single precision if acceptable:")
    if args.precision == 'double':
        single_mem = calculate_memory_requirements(Nr, Nt, 'single')
        print(f"   - Single precision: {single_mem['total_host_mem_gb']:.2f} GB (50% reduction)")
    print("4. Monitor actual memory usage and right-size accordingly")

if __name__ == '__main__':
    main()
