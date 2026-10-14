"""
End-to-end regression test for PFFDTD on a tiny shoebox room.

Pipeline exercised (same commands a user runs, via subprocess):
    sim_setup(...)  ->  fdtd_main_cpu_double.x  ->  python -m fdtd.process_outputs

The processed receiver signals (r_out_f) are compared against a committed
golden file.  The CPU double-precision engine is deterministic, so the
tolerance is tight.  Run time is well under a minute on a laptop.

Usage:
    cd <repo>; source python/venv_pffdtd/bin/activate
    pytest tests/                       # compare against golden
    PFFDTD_UPDATE_GOLDEN=1 pytest tests/   # (re)generate golden -- only when a
                                           # change is MEANT to alter results

The CPU engine is rebuilt with `make cpu_double` on every run.
Requires gcc, make and libhdf5-dev (see c_cuda/Makefile).
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
PY_DIR = REPO / 'python'
C_DIR = REPO / 'c_cuda'
FIXTURE = REPO / 'tests' / 'fixtures' / 'box_room' / 'model_export.json'
GOLDEN = REPO / 'tests' / 'golden' / 'box_room_r_out_f.h5'
OUT = REPO / 'tests' / '_out' / 'box_room'          # gitignored scratch folder
ENGINE = C_DIR / 'fdtd_main_cpu_double.x'
UPDATE_GOLDEN = os.environ.get('PFFDTD_UPDATE_GOLDEN', '0') == '1'

# Fixed simulation parameters.  Changing any of these invalidates the golden.
SETUP_KW = dict(
    model_json_file=str(FIXTURE),
    mat_folder=str(REPO / 'data' / 'materials'),
    mat_files_dict={'Plasterboard': 'mv_plasterboard.h5'},
    source_num=1,
    insig_type='impulse',
    diff_source=False,
    duration=0.02,
    Tc=20,
    rh=50,
    fcc_flag=False,
    PPW=7.5,
    fmax=1000.0,
    save_folder=str(OUT),
    compress=0,
    draw_vox=False,
    Nprocs=2,
)
PROCESS_ARGS = [
    '--fcut_lowpass', '1000', '--N_order_lowpass', '8', '--symmetric_lowpass',
    '--fcut_lowcut', '10.0', '--N_order_lowcut', '4',
]
# Relative tolerance on max |diff| / max |golden|.  CPU double is deterministic;
# this only needs to absorb libm / compiler differences between machines.
RTOL = 1e-9


def _run(cmd, cwd):
    """Run a subprocess, raising with its output on failure."""
    res = subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True)
    if res.returncode != 0:
        raise RuntimeError(
            f'command failed ({res.returncode}): {" ".join(map(str, cmd))}\n'
            f'--- stdout (tail) ---\n{res.stdout[-4000:]}\n'
            f'--- stderr (tail) ---\n{res.stderr[-4000:]}'
        )
    return res


def _git_commit():
    try:
        return subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=str(REPO), text=True,
                              capture_output=True, check=True).stdout.strip()
    except Exception:
        return 'unknown'


@pytest.fixture(scope='module')
def pipeline():
    """Run setup -> engine -> process_outputs once; yield the run folder."""
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    # 1. sim_setup (in its own interpreter, cwd=python/, like the test scripts)
    _run([sys.executable, '-c',
          'import json, sys; from sim_setup import sim_setup; sim_setup(**json.loads(sys.argv[1]))',
          json.dumps(SETUP_KW)], cwd=PY_DIR)

    # 2. CPU double-precision engine.  Always rebuild (a few seconds) so a stale
    #    binary can never mask a change in c_cuda/*.h.
    _run(['make', 'cpu_double'], cwd=C_DIR)
    assert ENGINE.exists(), f'engine not built: {ENGINE}'
    _run([str(ENGINE)], cwd=OUT)

    # 3. post-processing CLI
    _run([sys.executable, '-m', 'fdtd.process_outputs', '--data_dir', str(OUT)] + PROCESS_ARGS,
         cwd=PY_DIR)
    return OUT


def test_fixture_is_sane():
    with open(FIXTURE) as f:
        d = json.load(f)
    assert len(d['sources']) == 1
    assert len(d['receivers']) == 4
    assert set(d['mats_hash']) == {'Plasterboard', '_RIGID'}


def test_raw_output_structure(pipeline):
    """sim_outs.h5 layout must match what comms_out.h5 promised."""
    with h5py.File(pipeline / 'comms_out.h5', 'r') as f:
        Nr = int(f['Nr'][()])
        Nt = int(f['Nt'][()])
        out_alpha = f['out_alpha'][...]
    assert out_alpha.shape == (4, 8)           # 4 receivers x 8 trilinear points
    assert Nr == out_alpha.size
    with h5py.File(pipeline / 'sim_outs.h5', 'r') as f:
        u = f['u_out']
        assert u.shape == (Nr, Nt)
        assert u.dtype == np.float64
        u = u[...]
    assert np.all(np.isfinite(u))
    assert np.max(np.abs(u)) > 0.0


def test_processed_matches_golden(pipeline):
    with h5py.File(pipeline / 'sim_outs_processed.h5', 'r') as f:
        r_out_f = f['r_out_f'][...]
        Fs_f = float(f['Fs_f'][()])
    assert r_out_f.shape[0] == 4
    assert np.all(np.isfinite(r_out_f))

    if UPDATE_GOLDEN:
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(GOLDEN, 'w') as f:
            f.create_dataset('r_out_f', data=r_out_f)
            f.create_dataset('Fs_f', data=Fs_f)
            f.attrs['setup_kw'] = json.dumps(SETUP_KW)
            f.attrs['process_args'] = ' '.join(PROCESS_ARGS)
            f.attrs['git_commit'] = _git_commit()
            f.attrs['engine'] = ENGINE.name
        pytest.skip(f'golden regenerated at {GOLDEN} (commit {_git_commit()[:8]})')

    assert GOLDEN.exists(), (
        f'no golden at {GOLDEN}; generate once with PFFDTD_UPDATE_GOLDEN=1 on known-good code')
    with h5py.File(GOLDEN, 'r') as f:
        g = f['r_out_f'][...]
        g_Fs = float(f['Fs_f'][()])
    assert r_out_f.shape == g.shape, f'shape {r_out_f.shape} != golden {g.shape}'
    assert Fs_f == pytest.approx(g_Fs, rel=1e-12)
    scale = np.max(np.abs(g))
    err = np.max(np.abs(r_out_f - g)) / scale
    assert err <= RTOL, f'max rel error {err:.3e} > {RTOL:.0e} vs golden'


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
