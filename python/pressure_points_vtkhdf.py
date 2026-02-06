import math
import numpy as np
import h5py as h5

# VTK cell type for a single vertex (a point cell)
VTK_VERTEX = 1

def append_dataset(ds: h5.Dataset, arr: np.ndarray):
    """Append arr along axis 0 to a chunked dataset ds."""
    arr = np.asarray(arr)
    old = ds.shape[0]
    ds.resize(old + arr.shape[0], axis=0)
    ds[old:] = arr

def make_points(nx=20, ny=20, nz=10, spacing=(0.2, 0.2, 0.2)):
    """Regular grid of points."""
    sx, sy, sz = spacing
    xs = np.arange(nx) * sx
    ys = np.arange(ny) * sy
    zs = np.arange(nz) * sz
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1).astype(np.float32)
    return pts

def gaussian_pressure(points, t, center0, velocity, sigma=0.35, base=0.0, amp=1.0):
    """
    Moving 3D Gaussian blob:
      p(x,t) = base + amp * exp(-||x - c(t)||^2 / (2*sigma^2)) * sin(2*pi*f*t)
    """
    f = 1.0  # Hz, just for visual interest
    cx, cy, cz = (center0 + velocity * t)
    dx = points[:, 0] - cx
    dy = points[:, 1] - cy
    dz = points[:, 2] - cz
    r2 = dx*dx + dy*dy + dz*dz
    gauss = np.exp(-r2 / (2.0 * sigma * sigma)).astype(np.float32)
    osc = np.float32(math.sin(2.0 * math.pi * f * t))
    return (base + amp * gauss * osc).astype(np.float32)

def write_vtkhdf_pressure_points(
    out_path="pressure_points.vtkhdf",
    nsteps=60,
    dt=0.05
):
    points = make_points(nx=22, ny=22, nz=12, spacing=(0.15, 0.15, 0.15))
    npts = points.shape[0]

    # Build "vertex cells": one cell per point
    # Connectivity for vertices is just [0, 1, 2, ..., npts-1]
    connectivity = np.arange(npts, dtype=np.int64)

    # Types: one VTK_VERTEX per cell
    types = np.full((npts,), VTK_VERTEX, dtype=np.uint8)

    # Offsets: for vertex cells, each cell uses 1 connectivity id
    # Offsets length must be (#cells + 1): [0,1,2,3,...,npts]
    offsets = np.arange(npts + 1, dtype=np.int64)

    with h5.File(out_path, "w") as f:
        root = f.create_group("VTKHDF")

        # Header metadata
        root.attrs["Version"] = (2, 3)
        root.attrs["Type"] = "UnstructuredGrid"

        # ---- Static geometry (written ONCE) ----
        # In the tutorial, these can be per-part; we keep one part total.
        root.create_dataset("NumberOfPoints", data=(npts,), dtype="i8")
        root.create_dataset("Points", data=points, dtype="f")

        root.create_dataset("NumberOfCells", data=(npts,), dtype="i8")
        root.create_dataset("Types", data=types, dtype="uint8")

        root.create_dataset("NumberOfConnectivityIds", data=(npts,), dtype="i8")
        root.create_dataset("Connectivity", data=connectivity, dtype="i8")
        root.create_dataset("Offsets", data=offsets, dtype="i8")

        # ---- Time + offsets bookkeeping ----
        steps = root.create_group("Steps")
        steps.attrs["NSteps"] = nsteps

        # Time values (one per step)
        steps.create_dataset("Values", shape=(0,), maxshape=(None,), dtype="f")

        # These offsets tell ParaView where each step starts in flattened arrays.
        # Since geometry is STATIC, these are always 0 for every step.
        steps.create_dataset("PartOffsets", shape=(0,), maxshape=(None,), dtype="i8")
        steps.create_dataset("NumberOfParts", shape=(0,), maxshape=(None,), dtype="i8")
        steps.create_dataset("PointOffsets", shape=(0,), maxshape=(None,), dtype="i8")
        steps.create_dataset("CellOffsets", shape=(0,), maxshape=(None,), dtype="i8")
        steps.create_dataset("ConnectivityIdOffsets", shape=(0,), maxshape=(None,), dtype="i8")

        # ---- PointData: Pressure over time ----
        point_data = root.create_group("PointData")
        # Flattened: [t0 all points][t1 all points]...
        point_data.create_dataset("Pressure", shape=(0,), maxshape=(None,), dtype="f")

        # Offsets for point-data fields live under Steps/PointDataOffsets/<FieldName>
        pdo = steps.create_group("PointDataOffsets")
        pdo.create_dataset("Pressure", shape=(0,), maxshape=(None,), dtype="i8")

        # Dummy motion parameters for pressure field
        center0 = np.array([0.6, 0.6, 0.6], dtype=np.float32)
        velocity = np.array([0.15, 0.10, 0.05], dtype=np.float32)

        # ---- Write each timestep ----
        for k in range(nsteps):
            t = float(k * dt)

            # Record time
            append_dataset(steps["Values"], np.array([t], dtype=np.float32))

            # Static mesh offsets: always 0
            append_dataset(steps["PartOffsets"], np.array([0], dtype=np.int64))
            append_dataset(steps["NumberOfParts"], np.array([1], dtype=np.int64))
            append_dataset(steps["PointOffsets"], np.array([0], dtype=np.int64))
            append_dataset(steps["CellOffsets"], np.array([0], dtype=np.int64))
            append_dataset(steps["ConnectivityIdOffsets"], np.array([0], dtype=np.int64))

            # PointData offset: where THIS timestep’s pressure block starts
            pressure_ds = root["PointData/Pressure"]
            start = pressure_ds.shape[0]
            append_dataset(steps["PointDataOffsets/Pressure"], np.array([start], dtype=np.int64))

            # Append pressure values for all points at this time
            p = gaussian_pressure(points, t, center0, velocity, sigma=0.30, base=0.0, amp=1.0)
            append_dataset(pressure_ds, p)

    print(f"Wrote {out_path} with {npts} points and {nsteps} timesteps.")

if __name__ == "__main__":
    write_vtkhdf_pressure_points()
