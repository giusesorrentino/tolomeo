import logging

import pytest

from tests.common import RUN_YAML, write_run
from tolomeo._cli import main


@pytest.fixture(autouse=True)
def restore_log_level():
    # `-v` and `-q` change the level of the logger of the package: put it back.
    logger = logging.getLogger("tolomeo")
    level = logger.level
    yield
    logger.setLevel(level)


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------


def test_main_writes_the_map(tmp_path, capsys):
    path = write_run(tmp_path)

    code = main([str(path)])

    out, err = capsys.readouterr()
    assert code == 0
    assert (tmp_path / "out" / "map.fits").is_file()
    assert "[OUTPUT] Writing the map" in out
    assert err == ""


def test_main_quiet(tmp_path, capsys):
    code = main(["-q", str(write_run(tmp_path))])

    out, _ = capsys.readouterr()
    assert code == 0
    assert out == ""


def test_main_verbose(tmp_path, capsys):
    # `-v` adds the detailed messages, for example the time of every cleaning step.
    code = main(["-v", str(write_run(tmp_path))])

    out, _ = capsys.readouterr()
    assert code == 0
    assert "Completed baseline in" in out


def test_main_invalid_configuration(tmp_path, capsys):
    code = main([str(write_run(tmp_path, ""))])

    _, err = capsys.readouterr()
    assert code == 2
    assert err.startswith("error: ")
    assert "is empty" in err
    assert "Traceback" not in err


def test_main_missing_configuration(tmp_path, capsys):
    code = main([str(tmp_path / "missing.yaml")])

    _, err = capsys.readouterr()
    assert code == 2
    assert err.startswith("error: ")


def test_main_verbose_shows_the_traceback(tmp_path, capsys):
    code = main(["-v", str(write_run(tmp_path, ""))])

    _, err = capsys.readouterr()
    assert code == 2
    assert "Traceback" in err
    assert err.rstrip().splitlines()[-1].startswith("error: ")


def test_main_failed_run(tmp_path, capsys):
    path = write_run(tmp_path)
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "map.fits").touch()

    code = main([str(path)])

    _, err = capsys.readouterr()
    assert code == 1
    assert "error: " in err
    assert "already exists" in err


@pytest.mark.filterwarnings("always")
def test_main_prints_warnings_on_one_line(tmp_path, capsys):
    # A filter without a window step warns: the warning is printed as
    # "warning: <message>", without the file and the line that issued it.
    text = RUN_YAML + ("filtering:\n  steps:\n    - type: highpass\n      fc: 1\n")

    code = main([str(write_run(tmp_path, text))])

    _, err = capsys.readouterr()
    assert code == 0
    assert "warning: Filtering without windowing" in err
    assert ".py" not in err


def test_main_verbose_and_quiet_exclusive(tmp_path):
    with pytest.raises(SystemExit) as exc:
        main(["-v", "-q", str(tmp_path / "run.yaml")])

    assert exc.value.code == 2
