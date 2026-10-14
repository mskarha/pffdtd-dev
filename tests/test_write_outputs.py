"""
Unit test for write_outputs() in c_cuda/fdtd_data.h (the sim_outs.h5 writer).

Compiles tests/c/test_write_outputs.c against the real headers in several
block-size variants, then runs it over a range of (Nr, Nt) shapes so that the
single-block, multi-block, partial-last-block and row-larger-than-block paths
are all exercised on small data.  See tests/c/test_write_outputs.c.

Requires gcc and libhdf5-dev; HDF5 paths are read from c_cuda/Makefile.
"""
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
C_DIR = REPO / 'c_cuda'
SRC = REPO / 'tests' / 'c' / 'test_write_outputs.c'

# Block-size variants to build: name -> extra compiler flags.
# default: 256 MB blocks (production).  Small variants force many/partial blocks.
VARIANTS = {
    'default': [],
    'blk4k': ['-DWRITE_BLOCK_BYTES=4096'],
    'blk64k': ['-DWRITE_BLOCK_BYTES=65536'],
}

# (Nr, Nt, seed).  With 4 KiB blocks: Nt=37 -> 13 rows/block, Nt=1000 -> 1 row/block.
CASES = [
    (1, 1, 1),          # trivial
    (7, 37, 2),         # tiny, partial block with blk4k
    (1000, 1000, 3),    # 8 MB; single block (default), 1 row/block (blk4k), 8 rows/block (blk64k)
    (3001, 50, 4),      # many blocks + partial tail with small variants
    (13, 2000, 5),      # row (16 KB) > 4 KiB block -> clamped to 1 row/block
]


def _hdf5_paths():
    """First uncommented HDF5_INC / HDF5_LIB assignments in the Makefile."""
    text = (C_DIR / 'Makefile').read_text()
    inc = re.search(r'^HDF5_INC\s*\??=\s*(\S+)', text, re.M)
    lib = re.search(r'^HDF5_LIB\s*\??=\s*(\S+)', text, re.M)
    assert inc and lib, 'HDF5_INC/HDF5_LIB not found in c_cuda/Makefile'
    return inc.group(1), lib.group(1)


@pytest.fixture(scope='module', params=list(VARIANTS), ids=list(VARIANTS))
def binary(request, tmp_path_factory):
    """Compile one variant of the harness; yield the executable path."""
    name = request.param
    inc, lib = _hdf5_paths()
    exe = tmp_path_factory.mktemp('bin') / f'test_write_outputs_{name}.x'
    cmd = ['gcc', f'-I{C_DIR}', f'-I{inc}',
           '-Wall', '-Wextra', '-Wundef', '-Wpedantic', '-std=c99', '-O2', '-fopenmp',
           '-DPRECISION=2', '-DUSING_CUDA=false', *VARIANTS[name],
           str(SRC), '-o', str(exe), f'-L{lib}', '-lhdf5', '-lgomp', '-lm']
    res = subprocess.run(cmd, text=True, capture_output=True)
    assert res.returncode == 0, f'compile failed:\n{" ".join(cmd)}\n{res.stderr}'
    # Warnings originating in the header under test are a failure too.
    assert 'fdtd_data.h' not in res.stderr, f'warnings in fdtd_data.h:\n{res.stderr}'
    return exe


@pytest.mark.parametrize('Nr,Nt,seed', CASES, ids=[f'Nr{c[0]}_Nt{c[1]}' for c in CASES])
def test_write_outputs_roundtrip(binary, tmp_path, Nr, Nt, seed):
    res = subprocess.run([str(binary), str(Nr), str(Nt), str(seed)],
                         cwd=str(tmp_path), text=True, capture_output=True)
    assert res.returncode == 0, f'harness failed:\n{res.stdout[-3000:]}\n{res.stderr[-1000:]}'
    assert f'VERIFY Nr={Nr} Nt={Nt}: OK' in res.stdout, res.stdout[-3000:]
    assert (tmp_path / 'sim_outs.h5').exists()


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
