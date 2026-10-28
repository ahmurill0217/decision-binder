"""The documented numbered sequence must have no hidden split prerequisite."""
from pathlib import Path
import runpy
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_prepare_entrypoint_creates_splits(monkeypatch):
    import decision_binder.data as data
    import decision_binder.splits as splits
    calls = []
    monkeypatch.setattr(data, 'run', lambda config: calls.append(('prepare', config)))
    monkeypatch.setattr(splits, 'run', lambda config, mode: calls.append(('split', config, mode)))
    monkeypatch.setattr(sys, 'argv', ['01_prepare_data.py', '--config', 'configs/example.yaml'])
    runpy.run_path(str(ROOT / 'scripts/01_prepare_data.py'), run_name='__main__')
    assert calls == [('prepare', 'configs/example.yaml'), ('split', 'configs/example.yaml', 'primary')]


@pytest.mark.parametrize('script', ['01_prepare_data.py', '02_build_sft_dataset.py', '03_evaluate_base.py',
                                  '04_train_lora.py', '05_evaluate_finetuned.py', '06_compare_results.py'])
def test_numbered_cli_help(script, monkeypatch, capsys):
    monkeypatch.setattr(sys, 'argv', [script, '--help'])
    with pytest.raises(SystemExit) as exit:
        runpy.run_path(str(ROOT / 'scripts' / script), run_name='__main__')
    assert exit.value.code == 0
    assert '--config' in capsys.readouterr().out
