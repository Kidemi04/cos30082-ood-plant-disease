"""Create an isolated environment once, then run dataset preparation."""
from pathlib import Path
import subprocess
import sys
import venv


def main():
    project = Path(__file__).resolve().parents[1]
    environment = project / '.venv-dataset'
    python = environment / ('Scripts/python.exe' if sys.platform == 'win32' else 'bin/python')
    if not python.is_file():
        print('Creating the dataset environment...', flush=True)
        venv.EnvBuilder(with_pip=True).create(environment)
    requirements = project / 'requirements-dataset.txt'
    check = 'from importlib.metadata import version; ' + '; '.join(
        f'assert version({name!r}) == {version!r}'
        for name, version in (
            line.strip().split('==') for line in requirements.read_text().splitlines() if line.strip()
        )
    )
    if subprocess.run([str(python), '-c', check], capture_output=True).returncode:
        print('Installing dataset dependencies (Internet needed on the first run)...', flush=True)
        subprocess.run([str(python), '-m', 'pip', 'install', '-r', str(requirements)], check=True)
    return subprocess.call([str(python), '-u', str(project / 'scripts/prepare_dataset.py'), *sys.argv[1:]])


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, subprocess.CalledProcessError) as error:
        print(f'Setup failed: {error}', file=sys.stderr)
        sys.exit(1)
