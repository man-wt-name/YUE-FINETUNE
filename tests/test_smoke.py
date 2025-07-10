import subprocess, sys


def test_trainer_help():
    """`python -m finetune.new_yue_trainer --help` завершается с кодом 0."""
    result = subprocess.run(
        [sys.executable, "-m", "finetune.new_yue_trainer", "--help"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "YuE Trainer" in result.stdout


def test_infer_module_import():
    """Модуль inference.infer должен импортироваться без ошибок."""
    __import__("inference.infer") 