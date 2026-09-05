"""Convert a transient VTKHDF UnstructuredGrid (point cloud) to transient
VTKHDF ImageData.

Use case
--------
PFFDTD's existing receiver-grid exporter (fdtd.export_vtkhdf) produces a
single .vtkhdf of Type=UnstructuredGrid: a point cloud of receivers on
(usually) a regular Cartesian sub-grid, with vertex cells, plus a
flattened time-major Pressure array indexed by Steps/PointDataOffsets.

ParaView can render that as glyphs/points but it cannot do volume
rendering, FlyingEdges3D, or smooth contouring on a UnstructuredGrid of
vertex cells.  For that we need vtkImageData.

This script infers the underlying regular grid from the point coordinates,
re-pours each timestep's pressure into a dense (nz, ny, nx) voxel array
(NaN for missing cells, optionally hole-filled), and writes a single
transient VTKHDF ImageData containing the raw signed pressure plus a
family of precomputed dynamic-range-compressed `Alpha_*` fields suitable
for direct volume rendering.

Output arrays (all 4D: (NSteps, Nz, Ny, Nx), float32)
-----------------------------------------------------
  Pressure    raw signed pressure, NaN where no source point covered the cell
  Alpha_p2    p^2 / max(p^2) -- legacy default; heavily compressed dynamic range
  Alpha_lin   |p| / max(|p|) -- linear in amplitude
  Alpha_gamma (|p| / max(|p|))^gamma -- power-law expansion of low values
              gamma defaults to 0.4 (good general-purpose mid-range expansion)
  Alpha_dB    (20*log10(|p|/max(|p|)) + dB_floor) / dB_floor, clipped to [0, 1]
              Logarithmic; dB_floor defaults to 60 (i.e. shows 60 dB of range)

The global max(|p|) is computed once across the entire spacetime volume,
so peak amplitudes change over time the way they should (a quiet frame
won't be amplified to look like a loud one).

For all Alpha_* arrays, NaN-filled cells are converted to 0 so missing
cells render fully transparent under default opacity ramps.

Usage
-----
    python3 -m fdtd.vtkhdf_pointcloud_to_imagedata \
        --src   wave_propagation.vtkhdf \
        --out   wave_propagation_image.vtkhdf

Useful flags
------------
    --densify N      passes of 6-neighbour averaging to fill NaN cells
                     (1-3 helps FCC sublattices)
    --alpha-modes    comma-separated subset of: p2,lin,gamma,db
                     default 'lin,gamma,db' (all three useful variants)
    --gamma          exponent for Alpha_gamma (default 0.4; smaller = more
                     low-end expansion, larger = compressed)
    --db-floor       dB range covered by Alpha_dB (default 60.0)
    --gzip 0..9      compression level (default 3)
    --version        VTKHDF version (default 2.3)
    --no-alpha       skip all Alpha_* arrays (write only Pressure)
    --no-pressure    skip raw Pressure (Alpha_* only)

Recommended workflow in ParaView:
  - Try Alpha_dB first for the widest visible dynamic range
  - Try Alpha_gamma for a softer mid-range emphasis
  - Use Alpha_lin if you want strict linear-in-amplitude behavior
  - Switch to Pressure for signed contour/slice
"""
from __future__ import annotations

import argparse
import os
import sys

import h5py
import numpy as np


# ---------------------------------------------------------------------------
# Grid inference
# ---------------------------------------------------------------------------

def _unique_with_tolerance(values, tol):
    """Return (sorted unique values to within tol, integer index per input).

    A simple O(N log N) bucketization: sort, then walk merging consecutive
    values that differ by less than tol.  Integer index maps each input value
    to its bucket id in the unique array.
    """
    order = np.argsort(values, kind='stable')
    sorted_v = values[order]
    bucket_id = np.empty(len(values), dtype=np.int64)
    uniques = [sorted_v[0]]
    cur = 0
    bucket_id[order[0]] = cur
    for i in range(1, len(values)):
        if sorted_v[i] - uniques[cur] > tol:
            cur += 1
            uniques.append(sorted_v[i])
        bucket_id[order[i]] = cur
    return np.asarray(uniques, dtype=values.dtype), bucket_id


def infer_grid(points, tol=None):
    """Infer (origin, spacing, dims, ix, iy, iz) for a regular Cartesian grid.

    Parameters
    ----------
    points : (N, 3) float array of point coordinates
    tol    : tolerance for unique coordinate detection.  If None, set to
             min(diff)/4 along each axis.

    Returns
    -------
    dict with keys:
      origin   : (3,) float
      spacing  : (3,) float
      dims     : (Nx, Ny, Nz)
      ix, iy, iz : (N,) int arrays mapping each input point to its voxel
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f'points must be (N, 3), got shape {points.shape}')
    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    if tol is None:
        # Per-axis tolerance: 1/4 of the smallest non-zero diff in that axis
        tol_per_axis = []
        for ax in (x, y, z):
            sv = np.sort(ax)
            d = np.diff(sv)
            d = d[d > 0]
            if len(d) == 0:
                tol_per_axis.append(1e-9)
            else:
                tol_per_axis.append(0.25 * float(d.min()))
        tol_x, tol_y, tol_z = tol_per_axis
    else:
        tol_x = tol_y = tol_z = tol

    xv, ix = _unique_with_tolerance(x, tol_x)
    yv, iy = _unique_with_tolerance(y, tol_y)
    zv, iz = _unique_with_tolerance(z, tol_z)

    Nx, Ny, Nz = len(xv), len(yv), len(zv)
    if Nx < 2 or Ny < 2 or Nz < 2:
        raise ValueError(
            f'Inferred grid is degenerate: Nx={Nx} Ny={Ny} Nz={Nz}. '
            'Point cloud does not look like a 3D regular grid.'
        )

    # Spacing: take median of consecutive diffs in each axis
    dx = float(np.median(np.diff(xv)))
    dy = float(np.median(np.diff(yv)))
    dz = float(np.median(np.diff(zv)))

    # Sanity: all axis diffs should be very close to dx/dy/dz
    for axis_name, vals, sp in (('x', xv, dx), ('y', yv, dy), ('z', zv, dz)):
        diffs = np.diff(vals)
        rel_err = np.max(np.abs(diffs - sp)) / max(sp, 1e-30)
        if rel_err > 1e-2:
            print(f'WARN: {axis_name}-axis diffs vary by {rel_err*100:.2f}%; '
                  f'grid may not be perfectly uniform', file=sys.stderr)

    return {
        'origin':  (float(xv[0]), float(yv[0]), float(zv[0])),
        'spacing': (dx, dy, dz),
        'dims':    (Nx, Ny, Nz),
        'ix': ix.astype(np.int64),
        'iy': iy.astype(np.int64),
        'iz': iz.astype(np.int64),
    }


# ---------------------------------------------------------------------------
# Hole filling (NaN -> averaged neighbour)
# ---------------------------------------------------------------------------

def _fill_nan_neighbors(vol, passes=1):
    """One or more passes of 6-neighbour averaging to fill NaN cells.

    vol shape: (nz, ny, nx).  Modifies a copy and returns it.
    """
    out = vol.copy()
    for _ in range(passes):
        nan_mask = np.isnan(out)
        if not nan_mask.any():
            break
        # Build 6-neighbour sum + count of non-NaN neighbours
        filled = np.where(nan_mask, 0.0, out)
        count  = (~nan_mask).astype(np.float32)
        nbr_sum = np.zeros_like(filled)
        nbr_cnt = np.zeros_like(count)
        for axis in (0, 1, 2):
            for shift in (+1, -1):
                nbr_sum += np.roll(filled, shift, axis=axis)
                nbr_cnt += np.roll(count,  shift, axis=axis)
        with np.errstate(invalid='ignore', divide='ignore'):
            avg = nbr_sum / nbr_cnt
        # Only fill cells that were NaN AND have at least one non-NaN neighbour
        fillable = nan_mask & (nbr_cnt > 0)
        out = np.where(fillable, avg, out)
    return out


# ---------------------------------------------------------------------------
# Reader for the source UG .vtkhdf
# ---------------------------------------------------------------------------

def _read_ug_pressure_series(src_path):
    """Read points + per-step pressure arrays from a transient UG VTKHDF.

    Returns
    -------
    points       : (Npoints, 3) float64
    pressure_2d  : (NSteps, Npoints) float32  (NaN where the source had NaN)
    times        : (NSteps,) float
    """
    with h5py.File(src_path, 'r') as f:
        if 'VTKHDF' not in f:
            raise RuntimeError(f'{src_path}: missing /VTKHDF group')
        root = f['VTKHDF']
        type_attr = root.attrs.get('Type', b'')
        if isinstance(type_attr, bytes):
            type_attr = type_attr.decode('ascii', errors='replace')
        if type_attr not in ('UnstructuredGrid', 'PolyData'):
            print(f'WARN: source Type={type_attr!r} (expected UnstructuredGrid)',
                  file=sys.stderr)

        points = root['Points'][...]
        if points.ndim != 2 or points.shape[1] != 3:
            raise RuntimeError(
                f'Expected Points shape (N, 3), got {points.shape}')

        if 'Steps' not in root:
            raise RuntimeError(
                f'{src_path}: no Steps group -- not a transient file')
        steps = root['Steps']
        nsteps = int(steps.attrs.get('NSteps', len(steps['Values'])))
        times = np.asarray(steps['Values'][...], dtype=np.float32)

        pressure_flat = root['PointData/Pressure'][...]

        # Source layout: pressure can be either
        #   (a) flat 1D of length NSteps * Npoints, indexed by
        #       Steps/PointDataOffsets/Pressure
        #   (b) 2D (NSteps, Npoints)
        # Detect both.
        if pressure_flat.ndim == 1:
            offsets_path = 'PointDataOffsets/Pressure'
            if offsets_path in steps:
                offsets = np.asarray(steps[offsets_path][...], dtype=np.int64)
            else:
                # Assume contiguous Npoints per step
                if pressure_flat.size % nsteps != 0:
                    raise RuntimeError(
                        f'Pressure size {pressure_flat.size} not divisible by '
                        f'NSteps={nsteps}')
                npts = pressure_flat.size // nsteps
                offsets = np.arange(nsteps, dtype=np.int64) * npts

            # Try to find Npoints from the offset stride
            if len(offsets) >= 2:
                npts = int(offsets[1] - offsets[0])
            elif points.shape[0] > 0:
                npts = int(points.shape[0])
            else:
                raise RuntimeError('cannot determine Npoints')

            if npts * nsteps != pressure_flat.size:
                # Sometimes the last block is a different length; try
                # to handle by using offsets directly.
                pressure_2d = np.empty((nsteps, npts), dtype=np.float32)
                for t in range(nsteps):
                    start = int(offsets[t])
                    end = (int(offsets[t+1])
                           if t + 1 < nsteps else pressure_flat.size)
                    seg = pressure_flat[start:end]
                    if seg.size != npts:
                        raise RuntimeError(
                            f'step {t}: expected {npts} values, got {seg.size}')
                    pressure_2d[t] = seg
            else:
                pressure_2d = pressure_flat.reshape(nsteps, npts).astype(
                    np.float32, copy=False)
        elif pressure_flat.ndim == 2:
            if pressure_flat.shape[0] == nsteps:
                pressure_2d = pressure_flat.astype(np.float32, copy=False)
            elif pressure_flat.shape[1] == nsteps:
                pressure_2d = pressure_flat.T.astype(np.float32, copy=False)
            else:
                raise RuntimeError(
                    f'Pressure 2D shape {pressure_flat.shape} matches neither '
                    f'NSteps={nsteps} dimension')
        else:
            raise RuntimeError(
                f'Pressure has unexpected ndim={pressure_flat.ndim}')

    return points, pressure_2d, times


# ---------------------------------------------------------------------------
# Writer for the destination ImageData .vtkhdf
# ---------------------------------------------------------------------------

def _write_transient_imagedata(out_path, dims, origin, spacing, times,
                               channels, channel_iter,
                               gzip_level=3, version=(2, 3)):
    """Stream-write one transient ImageData file.

    `channels` is an ordered list of dataset names that will be created
    under PointData/.  `channel_iter` is a generator that yields a dict
    {name: (Nz, Ny, Nx) float32 volume} per timestep, with one entry per
    name in `channels`.
    """
    Nx, Ny, Nz = dims
    nt = len(times)

    print(f'writing -> {out_path}')
    print(f'  dims (Nx, Ny, Nz) = {Nx}, {Ny}, {Nz}')
    print(f'  origin  = {origin}')
    print(f'  spacing = {spacing}')
    print(f'  arrays  = {channels}')

    with h5py.File(out_path, 'w') as f:
        root = f.create_group('VTKHDF')
        root.attrs.create('Version', np.asarray(version, dtype=np.int64))
        type_str = b'ImageData'
        root.attrs.create('Type', type_str,
                          dtype=h5py.string_dtype('ascii', len(type_str)))
        root.attrs.create('WholeExtent',
                          np.asarray((0, Nx-1, 0, Ny-1, 0, Nz-1),
                                     dtype=np.int64))
        root.attrs.create('Origin',    np.asarray(origin,    dtype=np.float64))
        root.attrs.create('Spacing',   np.asarray(spacing,   dtype=np.float64))
        root.attrs.create('Direction',
                          np.asarray((1, 0, 0, 0, 1, 0, 0, 0, 1),
                                     dtype=np.float64))

        pd = root.create_group('PointData')

        kw = dict(shape=(nt, Nz, Ny, Nx), dtype=np.float32,
                  chunks=(1, Nz, Ny, Nx))
        if gzip_level and gzip_level > 0:
            kw.update(compression='gzip', compression_opts=int(gzip_level),
                      shuffle=True)

        datasets = {name: pd.create_dataset(name, **kw) for name in channels}

        steps = root.create_group('Steps')
        steps.attrs.create('NSteps', np.int64(nt))
        steps.create_dataset('Values',
                             data=np.asarray(times, dtype=np.float32))

        for t in range(nt):
            vols = next(channel_iter)
            for name in channels:
                datasets[name][t] = vols[name].astype(np.float32, copy=False)
            if (t + 1) % 5 == 0 or (t + 1) == nt:
                print(f'  wrote step {t+1}/{nt}  t={times[t]:.6g}s')


# ---------------------------------------------------------------------------
# Alpha variants
# ---------------------------------------------------------------------------

def _alpha_dB(absp_norm, db_floor):
    """absp_norm in [0, 1] -> dB-compressed [0, 1] over a `db_floor` dB range."""
    floor_lin = 10.0 ** (-db_floor / 20.0)
    ratio = np.maximum(absp_norm, floor_lin)
    db_value = 20.0 * np.log10(ratio) + db_floor
    return np.clip(db_value / db_floor, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

_ALPHA_MODE_TO_NAME = {
    'p2':    'Alpha_p2',
    'lin':   'Alpha_lin',
    'gamma': 'Alpha_gamma',
    'db':    'Alpha_dB',
}
_VALID_ALPHA_MODES = tuple(_ALPHA_MODE_TO_NAME.keys())


def convert(src, out,
            alpha_modes=('lin', 'gamma', 'db'),
            gamma=0.4, db_floor=60.0,
            write_pressure=True,
            densify_passes=0, gzip_level=3, version=(2, 3)):
    print(f'reading point cloud series from {src}')
    points, pressure_2d, times = _read_ug_pressure_series(src)
    nt, npts = pressure_2d.shape
    print(f'  Npoints = {npts}, NSteps = {nt}')
    print(f'  pressure range: [{np.nanmin(pressure_2d):.6g}, '
          f'{np.nanmax(pressure_2d):.6g}]')

    print('inferring regular grid from point coordinates ...')
    grid = infer_grid(points)
    Nx, Ny, Nz = grid['dims']
    ix, iy, iz = grid['ix'], grid['iy'], grid['iz']
    origin, spacing = grid['origin'], grid['spacing']
    print(f'  inferred dims (Nx, Ny, Nz) = {Nx}, {Ny}, {Nz}')
    print(f'  origin = {origin}')
    print(f'  spacing = {spacing}')
    fill_ratio = npts / float(Nx * Ny * Nz)
    print(f'  fill ratio = {fill_ratio*100:.1f}% '
          f'({npts} of {Nx*Ny*Nz} cells covered)')
    if densify_passes > 0:
        print(f'  hole-fill: {densify_passes} pass(es) of 6-neighbour averaging')

    # Pre-build (z, y, x) flat indices for scatter
    flat_idx = (iz.astype(np.int64) * (Ny * Nx)
                + iy.astype(np.int64) * Nx
                + ix.astype(np.int64))

    # Resolve which channels we'll write
    channels = []
    if write_pressure:
        channels.append('Pressure')
    for m in alpha_modes:
        if m not in _ALPHA_MODE_TO_NAME:
            raise ValueError(f'unknown alpha mode {m!r}; '
                             f'choose from {_VALID_ALPHA_MODES}')
        channels.append(_ALPHA_MODE_TO_NAME[m])
    if not channels:
        raise ValueError('No channels to write -- '
                         'pass at least one of pressure / alpha modes')

    # Pass 1: compute global max(|p|) once.  All Alpha_* variants are
    # derived from |p|/max_abs, so a single scan suffices.
    max_abs = 0.0
    if alpha_modes:
        print('pass 1/2: scanning for global max(|p|) ...')
        for t in range(nt):
            m = float(np.nanmax(np.abs(pressure_2d[t])))
            if np.isfinite(m) and m > max_abs:
                max_abs = m
            if (t + 1) % 5 == 0 or (t + 1) == nt:
                print(f'  [{t+1}/{nt}] running max(|p|) = {max_abs:.6g}')
        if max_abs == 0.0:
            max_abs = 1.0
        print(f'  global max(|p|) = {max_abs:.6g}')

    def _scatter(vec):
        """Scatter a length-Npoints vector into a (Nz, Ny, Nx) volume.

        NaN-fills cells without a corresponding source point.
        """
        vol = np.full((Nz, Ny, Nx), np.nan, dtype=np.float32)
        vol.reshape(-1)[flat_idx] = vec
        if densify_passes > 0:
            vol = _fill_nan_neighbors(vol, passes=densify_passes)
        return vol

    def make_channel_iter():
        for t in range(nt):
            p = pressure_2d[t]
            absp = np.abs(p)
            out_step = {}
            if write_pressure:
                # Keep NaN in Pressure so Threshold / Contour can mask
                out_step['Pressure'] = _scatter(p)
            if alpha_modes:
                # Build |p|/max_abs once on the volume, then derive variants
                absp_vol = _scatter(absp)
                # NaN -> 0 so missing cells are transparent under default ramp
                absp_vol = np.nan_to_num(absp_vol, nan=0.0, copy=False)
                norm_vol = (absp_vol / max_abs) if max_abs > 0 else absp_vol
                # Clip to [0, 1] -- densify averaging can push tiny values
                # slightly above 1 due to float roundoff
                norm_vol = np.clip(norm_vol, 0.0, 1.0).astype(np.float32,
                                                              copy=False)
                for m in alpha_modes:
                    name = _ALPHA_MODE_TO_NAME[m]
                    out_step[name] = _alpha_from_norm(
                        norm_vol, m, gamma=gamma, db_floor=db_floor)
            yield out_step

    print('pass 2/2: writing transient ImageData ...')
    _write_transient_imagedata(
        out, dims=(Nx, Ny, Nz), origin=origin, spacing=spacing,
        times=times,
        channels=channels, channel_iter=make_channel_iter(),
        gzip_level=gzip_level, version=version,
    )

    print()
    print(f'wrote {out}')
    if alpha_modes:
        print(f'  Alpha_* arrays normalized against max(|p|) = {max_abs:.6g}')
        if 'gamma' in alpha_modes:
            print(f'  Alpha_gamma exponent = {gamma}')
        if 'db' in alpha_modes:
            print(f'  Alpha_dB range = {db_floor} dB')
    print()
    print('In ParaView:')
    print(f'  1. File -> Open -> {out}')
    print('  2. Apply')
    print('  3. Active array dropdown -- pick the variant that gives the')
    print('     dynamic range you want:')
    print('       Alpha_dB    : 60 dB log scale (widest visible range)')
    print('       Alpha_gamma : power-law expansion of low values')
    print('       Alpha_lin   : linear in amplitude')
    print('       Alpha_p2    : energy (heavy compression of low values)')
    print('       Pressure    : signed raw, for Contour / Slice')


def _alpha_from_norm(norm_vol, mode, gamma=0.4, db_floor=60.0):
    """Apply a dynamic-range transform to |p|/max(|p|) in [0, 1]."""
    if mode == 'p2':
        return (norm_vol * norm_vol).astype(np.float32, copy=False)
    if mode == 'lin':
        return norm_vol
    if mode == 'gamma':
        return np.power(norm_vol, gamma).astype(np.float32, copy=False)
    if mode == 'db':
        return _alpha_dB(norm_vol, db_floor).astype(np.float32, copy=False)
    raise ValueError(f'unknown alpha mode {mode!r}')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--src',          required=True,
                   help='input transient UG VTKHDF (point cloud)')
    p.add_argument('--out',          default=None,
                   help='output transient ImageData VTKHDF (default <src>_image.vtkhdf)')
    p.add_argument('--alpha-modes',  default='lin,gamma,db',
                   help='comma-separated subset of {p2,lin,gamma,db} '
                        '(default: lin,gamma,db)')
    p.add_argument('--gamma',        type=float, default=0.4,
                   help='exponent for Alpha_gamma (default 0.4)')
    p.add_argument('--db-floor',     type=float, default=60.0,
                   help='dB range covered by Alpha_dB (default 60.0)')
    # Back-compat / convenience flags
    p.add_argument('--no-alpha',     action='store_true',
                   help='write only Pressure (skip all Alpha_* arrays)')
    p.add_argument('--no-pressure',  action='store_true',
                   help='write only Alpha_* (skip raw Pressure)')
    p.add_argument('--alpha-mode',   default=None,
                   choices=['pressure_squared', 'abs_pressure'],
                   help='[deprecated] legacy single-mode selector; '
                        'maps to --alpha-modes p2 or --alpha-modes lin')
    p.add_argument('--densify',      type=int, default=0,
                   help='passes of 6-neighbour averaging to fill NaN cells')
    p.add_argument('--gzip',         type=int, default=3)
    p.add_argument('--version',      default='2.3')
    args = p.parse_args(argv)

    out = args.out
    if out is None:
        base, ext = os.path.splitext(args.src)
        out = f'{base}_image{ext or ".vtkhdf"}'

    try:
        major, minor = (int(x) for x in args.version.split('.'))
    except ValueError:
        print(f'invalid --version {args.version!r}', file=sys.stderr)
        return 2

    # Resolve alpha modes
    if args.no_alpha:
        alpha_modes = ()
    elif args.alpha_mode is not None:
        print('WARN: --alpha-mode is deprecated; use --alpha-modes',
              file=sys.stderr)
        alpha_modes = ('p2',) if args.alpha_mode == 'pressure_squared' \
                      else ('lin',)
    else:
        alpha_modes = tuple(
            m.strip() for m in args.alpha_modes.split(',') if m.strip()
        )
        for m in alpha_modes:
            if m not in _VALID_ALPHA_MODES:
                print(f'invalid alpha mode {m!r}; '
                      f'choose from {_VALID_ALPHA_MODES}', file=sys.stderr)
                return 2

    convert(args.src, out,
            alpha_modes=alpha_modes,
            gamma=args.gamma, db_floor=args.db_floor,
            write_pressure=not args.no_pressure,
            densify_passes=max(0, args.densify),
            gzip_level=args.gzip, version=(major, minor))
    return 0


if __name__ == '__main__':
    sys.exit(main())
