"""Publication hygiene checks: no generated cohorts or training binaries in Git."""
from pathlib import Path
import subprocess
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_git_contains_no_large_training_artifacts():
    if not (ROOT / '.git').exists():
        pytest.skip('Source archive has no Git index')
    tracked = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0')
    violations = []
    for name in filter(None, tracked):
        path = ROOT / name
        data = name.startswith(('data/raw/', 'data/processed/', 'data/sft/', 'full/', 'artifacts/')) and path.name != '.gitkeep'
        binary = path.suffix in {'.safetensors', '.pt', '.pth', '.ckpt', '.bin'}
        checkpoint = any(part.startswith('checkpoint-') for part in path.relative_to(ROOT).parts)
        large = path.is_file() and path.stat().st_size > 10 * 1024**2
        if data or binary or checkpoint or large:
            violations.append(name)
    assert not violations, f'Tracked generated data/weights/checkpoints/large files: {violations}'


def test_ignore_rules_cover_training_outputs_in_other_locations():
    if not (ROOT / '.git').exists():
        pytest.skip('Source archive has no Git index')
    names = ['data/raw/model/config.json', 'data/processed/train.parquet',
             'data/sft/train.parquet', 'full/metadata.json', 'elsewhere/adapter_model.safetensors',
             'elsewhere/checkpoint-1/trainer_state.json', 'elsewhere/optimizer.pt']
    result = subprocess.run(['git', 'check-ignore', '--no-index', '--stdin'],
                            cwd=ROOT, input='\n'.join(names)+'\n', text=True, capture_output=True)
    assert set(result.stdout.splitlines()) == set(names)
