import json

from linear_models import cli


def test_all_writes_figures_and_json(tmp_path, capsys):
    out = tmp_path / "img"
    js = tmp_path / "results.json"
    assert cli.main(["all", "--out-dir", str(out), "--json", str(js)]) == 0
    assert (out / "scatter.png").stat().st_size > 10_000
    assert (out / "residuals.png").stat().st_size > 10_000
    results = json.loads(js.read_text())
    assert set(results) == {"replicate", "loglog"}
    assert "rho(Y, population) = +0.4557" in capsys.readouterr().out
