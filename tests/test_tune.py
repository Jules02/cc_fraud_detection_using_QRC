"""Tests for the Optuna hyperparameter search, on the smallest simulation preset."""

import csv
import json

import numpy as np
import optuna
import pytest

import experiments.cli as cli
from experiments import tune


def test_suggested_params_are_valid_qrc_configurations():
    """Every sampled point respects the encoding's qubit range and the qubit budget."""
    n_features, max_qubits = 7, 10
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    for _ in range(200):
        params = tune.suggest_params(
            study.ask(),
            n_features=n_features,
            encodings=list(tune.ENCODINGS),
            hamiltonians=list(tune.HAMILTONIANS),
            topologies=list(tune.TOPOLOGIES),
            max_qubits=max_qubits,
        )
        n_input = params["n_input_qubits"]
        assert n_input + params["n_mem_qubits"] <= max_qubits
        if params["encoding"] == "dense":
            assert n_features / 2 <= n_input < n_features
        else:
            assert n_input >= n_features
        assert (params["encoding_layers"] > 1) == (params["encoding"] == "reupload")
        assert ("depth" in params) == (params["hamiltonian"] == "random")
        assert ("evolution_time" in params) == (params["hamiltonian"] != "random")

        # and the qrc CLI accepts it
        args = cli.build_parser().parse_args(["qrc", *tune.cli_flags(params)])
        assert args.encoding == params["encoding"]


def test_infeasible_encodings_are_dropped():
    # 7 features: angle needs 7 input + 1 memory qubit, dense only 4 + 1
    assert tune.feasible_encodings(tune.ENCODINGS, 7, 6) == ["dense"]
    assert tune.feasible_encodings(tune.ENCODINGS, 7, 8) == list(tune.ENCODINGS)


def test_study_runs_saves_and_resumes(tmp_path):
    argv = [
        "--n-trials",
        "2",
        "--max-qubits",
        "7",
        "--encodings",
        "dense",
        "--hamiltonians",
        "random",
        "--output-dir",
        str(tmp_path),
        "--size",
        "small",
    ]
    study = tune.main(argv)
    study = tune.main([*argv[:1], "1", *argv[2:]])

    assert len(study.trials) == 3
    params = [trial.params for trial in study.trials]
    assert params[2] not in params[:2]  # the resumed sampler does not repeat itself

    study_dir = tmp_path / "qrc"
    with open(study_dir / "runs" / "summary.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3
    assert {row["encoding"] for row in rows} == {"dense"}

    # exact backend: every cost but shots is an objective
    assert len(study.directions) == 1 + 4
    pareto = json.loads((study_dir / "pareto.json").read_text())
    assert {entry["trial"] for entry in pareto} == {t.number for t in study.best_trials}
    for entry in pareto:
        assert list(entry["objectives"]) == [
            "validation_pr_auc",
            "qubits",
            "depth",
            "bases",
            "wall_time",
        ]
        assert entry["qubits"] <= 7 and entry["depth"] > 0 and entry["shots"] == 0
        assert entry["command"].startswith("python -m experiments.cli qrc")
    assert (study_dir / "trials.csv").exists()


def test_shots_objective_searches_precision(tmp_path):
    study = tune.main(
        [
            "--n-trials",
            "1",
            "--objectives",
            "shots",
            "--max-qubits",
            "7",
            "--encodings",
            "dense",
            "--hamiltonians",
            "random",
            "--output-dir",
            str(tmp_path),
            "--size",
            "small",
            "--backend",
            "gaussian",
        ]
    )
    (trial,) = study.trials
    assert trial.params["precision"] in tune.PRECISIONS
    assert trial.values[1] == trial.user_attrs["shots"] == round(trial.params["precision"] ** -2)


def test_shots_objective_needs_a_sampled_backend(tmp_path):
    with pytest.raises(SystemExit, match="sampled backend"):
        tune.main(["--output-dir", str(tmp_path), "--objectives", "shots"])
    with pytest.raises(SystemExit, match="precision"):
        tune.main(["--output-dir", str(tmp_path), "--backend", "shots", "--precision", "0.1"])


def test_no_objectives_is_a_single_objective_search(tmp_path):
    study = tune.main(
        ["--n-trials", "0", "--objectives", "--output-dir", str(tmp_path), "--size", "small"]
    )
    assert len(study.directions) == 1


def test_circuit_cost_counts_window_and_routing():
    def cost(window_length, topology):
        args = cli.build_parser().parse_args(
            ["qrc", "--hamiltonian", "tfim", "--topology", topology, "--n-mem-qubits", "2"]
        )
        protocol, _ = cli.build_protocol(args)
        return tune.circuit_cost(protocol, window_length, n_features=7, reset_inputs=False)

    linear = cost(3, "linear")
    assert cost(6, "linear")["two_qubit_gates"] == 2 * linear["two_qubit_gates"]
    # fully-connected couplings on a linear chain need SWAPs
    assert cost(3, "fully-connected")["two_qubit_gates"] > 3 * linear["two_qubit_gates"]


def test_resuming_with_another_setup_is_refused(tmp_path):
    argv = ["--n-trials", "0", "--output-dir", str(tmp_path), "--size", "small"]
    tune.main(argv)
    with pytest.raises(SystemExit, match="different setup"):
        tune.main([*argv[:-1], "medium"])


def test_searched_options_cannot_be_passed_through(tmp_path):
    with pytest.raises(SystemExit, match="n_mem_qubits"):
        tune.main(["--output-dir", str(tmp_path), "--n-mem-qubits", "3"])


@pytest.mark.parametrize(
    "flags, encoding",
    [
        ([], "angle"),
        (["--n-input-qubits", "4"], "dense"),
        (["--encoding-layers", "2"], "reupload"),
        (["--encoding", "reupload", "--encoding-layers", "1"], "reupload"),
    ],
)
def test_qrc_cli_resolves_encoding(monkeypatch, flags, encoding):
    args = cli.build_parser().parse_args(
        ["qrc", "--hamiltonian", "tfim", "--evolution-time", "0.5", *flags]
    )
    captured = []

    class RecordingProtocol:
        def __init__(self, encoder, reservoir, observables, backend):
            captured.append((encoder, reservoir))

        def run(self, windows):
            return np.zeros((len(windows), 1))

    monkeypatch.setattr(cli, "QRCProtocol", RecordingProtocol)
    monkeypatch.setattr(cli, "make_backend", lambda *args: args)
    data = {
        f"X_{split}": np.zeros((1, 1, 7)) for split in ("train", "validation", "test")
    }

    _, reservoir_info = cli.qrc_features(data, args)

    assert reservoir_info["encoding"] == encoding
    assert reservoir_info["evolution_time"] == 0.5
    assert captured[0][1].time == 0.5
