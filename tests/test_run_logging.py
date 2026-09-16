import json
import logging
import sys

import pytest

from satvision_pix4d.run_logging import RunLogging


def test_rank_traceback_output_and_stream_restoration(tmp_path, monkeypatch):
    monkeypatch.setenv('LOCAL_RANK', '5')
    monkeypatch.delenv('RANK', raising=False)
    monkeypatch.setenv('NODE_RANK', '0')
    stdout, stderr = sys.stdout, sys.stderr
    logger = logging.getLogger('satvision-test-nonpropagating')
    handler = logging.StreamHandler(stderr)
    logger.addHandler(handler)
    logger.propagate = False
    logger.setLevel(logging.INFO)
    try:
        with pytest.raises(RuntimeError, match='worker failed'):
            with RunLogging(tmp_path, command=['example']) as run:
                print('stdout message')
                print('stderr message', file=sys.stderr)
                logger.info('existing logger message')
                raise RuntimeError('worker failed')
        content = run.log_path.read_text()
        for expected in ['stdout message', 'stderr message', 'existing logger message',
                         'Traceback', 'RuntimeError: worker failed', 'test_run_logging.py']:
            assert expected in content
        status = json.loads(run.status_path.read_text())
        assert status['rank'] == '5' and status['status'] == 'failed'
        assert status['exit_code'] == 1 and 'worker failed' in status['traceback']
        assert sys.stdout is stdout and sys.stderr is stderr
        assert handler.stream is stderr
    finally:
        logger.removeHandler(handler)
        logger.propagate = True
        handler.close()


def test_retry_logs_are_preserved_and_nonzero_completion_is_distinct(tmp_path):
    with RunLogging(tmp_path) as first:
        print('first attempt')
        first.exit_code = 2
    with RunLogging(tmp_path) as second:
        print('second attempt')
    assert first.log_path != second.log_path
    assert 'first attempt' in first.log_path.read_text()
    assert json.loads(first.status_path.read_text())['status'] == 'completed_nonzero'
    assert json.loads(second.status_path.read_text())['status'] == 'completed'


def test_benchmark_logs_failure_before_trainer_creation(tmp_path, monkeypatch):
    import satvision_pix4d.benchmark as benchmark

    def fail_config(args):
        raise RuntimeError('configuration setup failed')

    monkeypatch.setattr(benchmark, 'prepare_config', fail_config)
    with pytest.raises(RuntimeError, match='configuration setup failed'):
        benchmark.main(['--output', str(tmp_path)])
    files = list((tmp_path / 'logs').glob('*.status.json'))
    assert len(files) == 1
    status = json.loads(files[0].read_text())
    assert status['status'] == 'failed'
    assert 'fail_config' in status['traceback']


def test_pretraining_logs_failure_before_trainer_creation(tmp_path, monkeypatch):
    import satvision_pix4d.satvision_pix4d_cli as training

    def fail_setup(config, output_dir):
        raise ValueError('pipeline setup failed')

    monkeypatch.setattr(training, '_train', fail_setup)
    with pytest.raises(ValueError, match='pipeline setup failed'):
        training.main(None, str(tmp_path))
    status = json.loads(next((tmp_path / 'logs').glob('*.status.json')).read_text())
    assert status['status'] == 'failed' and 'fail_setup' in status['traceback']
