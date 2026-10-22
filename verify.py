"""Verify saved results and inspect additional saved artifacts without training."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', help='Fresh directory under this repository\'s runs/; defaults to a unique directory.')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    output = (Path(args.output).expanduser().resolve() if args.output else
              root/'runs'/f'verify_{time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())}_{os.getpid()}')
    if output.exists():
        parser.error('Output directory already exists; choose a fresh directory.')
    runs = root/'runs'
    if output == runs or runs not in output.parents:
        parser.error('Choose a fresh subdirectory of this repository\'s runs/ directory.')
    output.mkdir(parents=True)
    started = time.monotonic()
    commands = [
        ('core_saved_results', [sys.executable, '-B', str(root/'scripts/verify_saved_results.py'),
                                '--root', str(root), '--output', str(output/'core')]),
        ('additional_saved_inspection', [sys.executable, '-B', str(root/'additional_analyses/portable/reproduce.py'),
                                         'inspect-saved', '--snapshot-root', str(root)]),
        ('e1_saved_rescore', [sys.executable, '-B', str(root/'additional_analyses/portable/reproduce.py'),
                              'rescore-e1', '--snapshot-root', str(root), '--output-dir', str(output/'e1')])
    ]
    results = []
    for name, command in commands:
        before = time.monotonic()
        with (output/f'{name}.stdout.txt').open('w', encoding='utf-8') as stdout, \
             (output/f'{name}.stderr.txt').open('w', encoding='utf-8') as stderr:
            completed = subprocess.run(command, stdout=stdout, stderr=stderr, check=False)
        results.append({'check':name, 'exit_code':completed.returncode,
                        'elapsed_seconds':round(time.monotonic()-before,3)})
        print(f'{name}: exit {completed.returncode}', flush=True)
    receipt = {'status':'PASS' if all(r['exit_code']==0 for r in results) else 'FAIL',
               'elapsed_seconds':round(time.monotonic()-started,3), 'checks':results,
               'scope':'Four-group saved-result verification including canonical CPU R2/margins, E1 full saved-prediction rescore, and selected E2 saved-output inspection.',
               'training_performed':False,
               'limits':'No full historical training reconstruction, E1 refitting or E2 decomposition rerun.'}
    (output/'verification_status.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(receipt, indent=2), flush=True)
    print('Output: ' + str(output), flush=True)
    return 0 if receipt['status']=='PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
