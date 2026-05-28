from pathlib import Path

from quad_pendulum.policy import default_model_path


def test_default_model_path_prefers_explicit_model():
    explicit_path = Path("custom.zip")

    assert default_model_path("sac", 1, explicit_path) == explicit_path


def test_default_model_path_prefers_best_checkpoint(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    best_path = tmp_path / "models" / "sac_links1_best" / "best_model.zip"
    final_path = tmp_path / "models" / "sac_links1.zip"
    best_path.parent.mkdir(parents=True)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    best_path.touch()
    final_path.touch()

    assert default_model_path("sac", 1) == Path("models/sac_links1_best/best_model.zip")


def test_default_model_path_falls_back_to_final_model(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    assert default_model_path("ppo", 4) == Path("models/ppo_links4.zip")
