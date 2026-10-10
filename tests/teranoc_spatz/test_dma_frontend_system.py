"""Build the bare-metal DMA ABI probe and run installed 4x4 and 8x8 models.

Usage: python3 test_dma_frontend_system.py [--work-dir /tmp/dma-abi]
Set RISCV_CC to override the neighboring RTL repo's GCC installation.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import tempfile

here = Path(__file__).resolve().parent
root = here.parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--work-dir', type=Path)
args = parser.parse_args()
work = (args.work_dir or Path(tempfile.mkdtemp(prefix='teranoc-dma-abi-'))).resolve()
work.mkdir(parents=True, exist_ok=True)
cc = os.environ.get('RISCV_CC', str(root.parents[1] /
    'TeraNoC/install/riscv-gcc/bin/riscv32-unknown-elf-gcc'))
elf = work / 'dma_frontend_smoke.elf'
subprocess.run([cc, '-march=rv32ima', '-mabi=ilp32', '-nostdlib',
                '-Wl,-Ttext=0x80000000,-e,_start,--no-relax',
                str(here / 'dma_frontend_smoke.S'), '-o', str(elf)], check=True)
env = {k: v for k, v in os.environ.items()
       if not k.startswith('TERANOC_') and k != 'QWEN_L2_BYTES'}
env.update(PATH=str(root / 'install/bin') + ':/tmp/teranoc-gvsoc-py312:' + env['PATH'],
           PYTHONPATH=str(root / 'install/python'), LD_LIBRARY_PATH=str(root / 'install/lib'))


def run(profile):
    run_dir = work / profile
    run_dir.mkdir(exist_ok=True)
    command = ['/usr/bin/python3.12', '-m', 'gvrun',
               '--target=teranoc_spatz:system_config=' + profile,
               '--work-dir=' + str(run_dir), '--parameter', 'binary=' + str(elf), 'run']
    with (run_dir / 'run.log').open('w') as log:
        process = subprocess.Popen(command, cwd=root, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            exit_code = process.wait(timeout=600)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            exit_code = 124
    output = (run_dir / 'run.log').read_text(errors='replace')
    eoc = re.search(r'EOC register return value: (0x[0-9a-f]+)', output)
    record = dict(profile=profile, command=command, exit_code=exit_code,
                  eoc=int(eoc[1], 16) if eoc else None)
    record['passed'] = exit_code == 0 and record['eoc'] == 0
    print(json.dumps(record), flush=True)
    return record


profiles = ['terapool_spatz4_fpu', 'terapool_spatz4_fpu_8x8']
with ThreadPoolExecutor(max_workers=2) as pool:
    results = list(pool.map(run, profiles))
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


record = dict(elf_sha256=sha(elf), results=results,
              frontend_source_sha256=sha(root / 'pulp/pulp/teranoc_spatz/dma/teranoc_dma_frontend.hpp'),
              dma_model_sha256={str(p.relative_to(root)): sha(p)
                               for p in (root / 'install/models').glob('*teranoc_dma_cpp*.so')})
(work / 'results.json').write_text(json.dumps(record, indent=2) + '\n')
assert all(r['passed'] for r in results), f'DMA ABI regression failed; see {work}'
print(f'PASS installed 4x4/8x8 DMA reset, launch, sticky completion and copied data: {work}')
