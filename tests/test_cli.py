""" Tests for command line interface """
from click.testing import CliRunner
from reemission.cli.cli import main


def test_main():
    runner = CliRunner()
    result_ok = runner.invoke(main, [])
    result_notok = runner.invoke(main, ["dd"])
    assert result_ok.exit_code == 0
    assert result_notok.exit_code == 2
