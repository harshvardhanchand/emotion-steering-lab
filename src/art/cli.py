"""Typer CLI for steering-only framework."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer

from art.artifacts.paths import create_run_dir, make_run_id
from art.artifacts.read import read_json, read_jsonl
from art.artifacts.write import write_json, write_jsonl, write_text
from art.constants import (
    DEFAULT_ALPHA_MAX,
    DEFAULT_ALPHA_MIN,
    DEFAULT_ALPHA_STEP,
    DEFAULT_HIDDEN_SIZE,
    DEFAULT_MAX_LENGTH,
    DEFAULT_NUM_LAYERS,
    PAPER_MODE_NAME,
    project_root,
)
from art.data.generate import DataGenConfig, generate_probe_data
from art.errors import ArtError, SchemaValidationError
from art.probes.diagnose import run_diagnosis
from art.probes.dataset import load_probe_data
from art.probes.train import train_probe_artifact
from art.repro import hash_file, hash_object, meta_path_for, utc_now_iso
from art.steering.profile import build_profile
from art.steering.run import run_steering
from art.steering.sweep import run_alpha_sweep
from art.schemas.validator import validate_document, validate_documents

app = typer.Typer(help="Inference-time steering framework", no_args_is_help=True)
data_app = typer.Typer(help="Synthetic data generation")
probe_app = typer.Typer(help="Probe training")
steer_app = typer.Typer(help="Steering runs")

app.add_typer(data_app, name="data")
app.add_typer(probe_app, name="probe")
app.add_typer(steer_app, name="steer")


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else (project_root() / path)


def _exit_with_error(exc: Exception) -> None:
    typer.echo(f"Error: {exc}", err=True)
    raise typer.Exit(code=1) from exc


def _write_meta(
    *,
    primary_output: Path,
    command: str,
    config: dict[str, Any],
    inputs: dict[str, Path],
    outputs: dict[str, Path],
) -> None:
    input_hashes: dict[str, str] = {}
    for key, path in inputs.items():
        if path.exists():
            input_hashes[key] = hash_file(path)

    output_hashes: dict[str, str] = {}
    for key, path in outputs.items():
        if path.exists():
            output_hashes[key] = hash_file(path)

    payload = {
        "schema_version": "1.0",
        "created_at": utc_now_iso(),
        "command": command,
        "config": config,
        "config_hash": hash_object(config),
        "inputs": {k: str(v) for k, v in inputs.items()},
        "input_hashes": input_hashes,
        "outputs": {k: str(v) for k, v in outputs.items()},
        "output_hashes": output_hashes,
    }
    meta_path = meta_path_for(primary_output)
    write_json(meta_path, payload)


def _profile_steer_settings(
    *,
    profile_payload: dict[str, Any],
    artifact_model_id: str,
    default_scope: str,
) -> tuple[str, float, list[int], str]:
    if bool(profile_payload.get("enabled", True)) is not True:
        raise ArtError("Profile is disabled (`enabled=false`)")

    profile_type = str(profile_payload.get("profile_type", "")).strip()
    if profile_type == "slider_blend_v1":
        profile_model = str(profile_payload.get("model_id", "")).strip()
        if profile_model and profile_model != artifact_model_id:
            raise ArtError(
                f"Profile model_id ({profile_model}) does not match artifact model_id ({artifact_model_id})"
            )
        weights = profile_payload.get("weights")
        if not isinstance(weights, dict):
            raise ArtError("Invalid slider profile: missing `weights` object")
        non_zero = [(str(k), float(v)) for k, v in weights.items() if abs(float(v)) > 1e-9]
        if len(non_zero) == 0:
            raise ArtError("Slider profile has no active probe weight")
        if len(non_zero) > 1:
            raise ArtError(
                "Slider profile has multiple active probes; `steer run` supports one probe. "
                "Use a single active weight or run from UI chat mode."
            )
        probe_name, alpha = non_zero[0]
        selected_layer = int(profile_payload.get("selected_layer", 0))
        return probe_name, alpha, [selected_layer], default_scope

    validate_document(profile_payload, "steering_profile.schema.json", context="steering_profile")
    profile_model = str(profile_payload.get("model_id", "")).strip()
    if profile_model and profile_model != artifact_model_id:
        raise ArtError(
            f"Profile model_id ({profile_model}) does not match artifact model_id ({artifact_model_id})"
        )
    probe_name = str(profile_payload["probe_name"]).strip()
    alpha = float(profile_payload["alpha"])
    layers = [int(x) for x in list(profile_payload.get("layers", []))]
    if not layers:
        raise ArtError("Profile `layers` must contain at least one layer")
    scope = str(profile_payload.get("scope", default_scope))
    return probe_name, alpha, layers, scope


@app.command("ui")
def launch_ui() -> None:
    """Launch Streamlit UI."""
    try:
        from art.ui_app import launch

        launch()
    except Exception as exc:
        _exit_with_error(exc)


@data_app.command("generate")
def data_generate(
    out: Path = typer.Option(Path("data/probe_data.jsonl"), "--out"),
    backend: str = typer.Option("transformers", "--backend"),
    model_id: str = typer.Option("Qwen/Qwen2.5-0.5B-Instruct", "--model-id"),
    tokenizer_id: str = typer.Option("", "--tokenizer-id"),
    device: str = typer.Option("auto", "--device"),
    dtype: str = typer.Option("auto", "--dtype"),
    max_length: int = typer.Option(DEFAULT_MAX_LENGTH, "--max-length"),
    max_new_tokens: int = typer.Option(768, "--max-new-tokens"),
    temperature: float = typer.Option(0.0, "--temperature"),
    generation_batch_size: int = typer.Option(8, "--generation-batch-size"),
    generation_cache: bool = typer.Option(True, "--generation-cache/--no-generation-cache"),
    generation_cache_dir: str = typer.Option("cache/generation", "--generation-cache-dir"),
    emotion: list[str] = typer.Option([], "--emotion", help="Repeat flag to select one or more emotions."),
    topic: list[str] = typer.Option([], "--topic", help="Optional override for topics (default uses full paper topic list)."),
    seed: int = typer.Option(42, "--seed"),
    paper_mode: str = typer.Option(PAPER_MODE_NAME, "--paper-mode"),
) -> None:
    """Generate synthetic probe_data records."""

    try:
        if paper_mode != PAPER_MODE_NAME:
            raise ArtError(f"Unsupported paper mode: {paper_mode}")

        cfg = DataGenConfig(
            emotions=emotion or None,
            topics=topic or None,
            seed=seed,
            backend_name=backend,
            model_id=model_id,
            tokenizer_id=tokenizer_id or model_id,
            device=device,
            dtype=dtype,
            max_length=max_length,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            generation_batch_size=generation_batch_size,
            use_generation_cache=generation_cache,
            generation_cache_dir=generation_cache_dir,
        )
        rows = generate_probe_data(cfg)
        validate_documents(rows, "probe_data.schema.json", context_prefix="probe_data")

        out_path = _resolve(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.exists():
            raise ArtError(f"Output already exists (immutable): {out_path}")
        write_jsonl(out_path, rows)
        _write_meta(
            primary_output=out_path,
            command="data.generate",
            config={
                "paper_mode": paper_mode,
                "backend": backend,
                "model_id": model_id,
                "tokenizer_id": tokenizer_id or model_id,
                "device": device,
                "dtype": dtype,
                "max_length": max_length,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "generation_batch_size": generation_batch_size,
                "generation_cache": generation_cache,
                "generation_cache_dir": generation_cache_dir,
                "emotion": emotion,
                "topic": topic,
                "seed": seed,
            },
            inputs={},
            outputs={"probe_data": out_path},
        )
        typer.echo(f"rows={len(rows)}")
        typer.echo(f"probe_data={out_path}")
    except (ArtError, SchemaValidationError, OSError, ValueError) as exc:
        _exit_with_error(exc)


@probe_app.command("train")
def probe_train(
    probe_data: Path = typer.Option(Path("data/probe_data.jsonl"), "--probe-data"),
    model_id: str = typer.Option("Qwen/Qwen2.5-0.5B-Instruct", "--model-id"),
    tokenizer_id: str = typer.Option("", "--tokenizer-id"),
    backend: str = typer.Option("transformers", "--backend"),
    device: str = typer.Option("auto", "--device"),
    dtype: str = typer.Option("auto", "--dtype"),
    max_length: int = typer.Option(DEFAULT_MAX_LENGTH, "--max-length"),
    num_layers: int = typer.Option(DEFAULT_NUM_LAYERS, "--num-layers"),
    hidden_size: int = typer.Option(DEFAULT_HIDDEN_SIZE, "--hidden-size"),
    activation_cache: bool = typer.Option(True, "--activation-cache/--no-activation-cache"),
    activation_cache_dir: str = typer.Option("cache/activations", "--activation-cache-dir"),
    out: str = typer.Option("", "--out"),
) -> None:
    """Train probe vectors and write probe_artifact.json."""

    try:
        probe_path = _resolve(probe_data)
        records = load_probe_data(probe_path)
        validate_documents(records, "probe_data.schema.json", context_prefix="probe_data")

        artifact = train_probe_artifact(
            records=records,
            model_id=model_id,
            tokenizer_id=tokenizer_id or model_id,
            num_layers=num_layers,
            hidden_size=hidden_size,
            backend_name=backend,
            device=device,
            dtype=dtype,
            max_length=max_length,
            use_activation_cache=activation_cache,
            activation_cache_dir=activation_cache_dir,
        )
        validate_document(artifact, "probe_artifact.schema.json", context="probe_artifact")

        if out:
            out_path = _resolve(Path(out))
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if out_path.exists():
                raise ArtError(f"Output already exists (immutable): {out_path}")
        else:
            run_id = make_run_id()
            run_dir = create_run_dir(run_id)
            out_path = run_dir / "probe_artifact.json"

        write_json(out_path, artifact)
        _write_meta(
            primary_output=out_path,
            command="probe.train",
            config={
                "backend": backend,
                "model_id": model_id,
                "tokenizer_id": tokenizer_id or model_id,
                "device": device,
                "dtype": dtype,
                "max_length": max_length,
                "num_layers": num_layers,
                "hidden_size": hidden_size,
                "activation_cache": activation_cache,
                "activation_cache_dir": activation_cache_dir,
            },
            inputs={"probe_data": probe_path},
            outputs={"probe_artifact": out_path},
        )
        typer.echo(f"probe_artifact={out_path}")
        typer.echo(f"probes={len(artifact.get('probes', []))}")
    except (ArtError, SchemaValidationError, OSError, ValueError) as exc:
        _exit_with_error(exc)


@steer_app.command("run")
def steer_run(
    probe_artifact: Path = typer.Option(..., "--probe-artifact"),
    probe_name: str = typer.Option("", "--probe-name"),
    cases: Path = typer.Option(Path("data/probe_data.jsonl"), "--cases"),
    backend: str = typer.Option("transformers", "--backend"),
    model_id: str = typer.Option("", "--model-id"),
    tokenizer_id: str = typer.Option("", "--tokenizer-id"),
    device: str = typer.Option("auto", "--device"),
    dtype: str = typer.Option("auto", "--dtype"),
    max_length: int = typer.Option(DEFAULT_MAX_LENGTH, "--max-length"),
    max_new_tokens: int = typer.Option(96, "--max-new-tokens"),
    temperature: float = typer.Option(0.0, "--temperature"),
    alpha: float = typer.Option(0.05, "--alpha"),
    scope: str = typer.Option("full_suite", "--scope"),
    base_run_id: str = typer.Option("baseline", "--base-run-id"),
    out: str = typer.Option("", "--out"),
    profile: str = typer.Option("", "--profile"),
    save_profile: str = typer.Option("", "--save-profile"),
) -> None:
    """Run causal inference-time steering and write steering results."""

    try:
        artifact_path = _resolve(probe_artifact)
        cases_path = _resolve(cases)
        artifact = read_json(artifact_path)
        validate_document(artifact, "probe_artifact.schema.json", context="probe_artifact")

        case_rows = read_jsonl(cases_path)
        validate_documents(case_rows, "probe_data.schema.json", context_prefix="probe_data")

        resolved_probe_name = str(probe_name).strip()
        resolved_alpha = float(alpha)
        resolved_scope = str(scope)
        resolved_layers: list[int] | None = None
        profile_input_path: Path | None = None

        if profile:
            profile_input_path = _resolve(Path(profile))
            profile_payload = read_json(profile_input_path)
            if not isinstance(profile_payload, dict):
                raise ArtError(f"Invalid profile document: {profile_input_path}")
            (
                resolved_probe_name,
                resolved_alpha,
                resolved_layers,
                resolved_scope,
            ) = _profile_steer_settings(
                profile_payload=profile_payload,
                artifact_model_id=str(artifact["model"]["model_id"]),
                default_scope=resolved_scope,
            )

        if not resolved_probe_name:
            raise ArtError("Provide --probe-name or --profile")
        if resolved_scope not in {"full_suite", "failures_only"}:
            raise ArtError("--scope must be one of: full_suite, failures_only")

        steer_run_id = make_run_id()
        if out:
            out_path = _resolve(Path(out))
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if out_path.exists():
                raise ArtError(f"Output already exists (immutable): {out_path}")
        else:
            run_dir = create_run_dir(steer_run_id)
            out_path = run_dir / "steering_results.jsonl"

        rows = run_steering(
            probe_artifact=artifact,
            probe_name=resolved_probe_name,
            alpha=resolved_alpha,
            cases=case_rows,
            base_run_id=base_run_id,
            steer_run_id=steer_run_id,
            scope=resolved_scope,
            backend_name=backend,
            model_id=model_id,
            tokenizer_id=tokenizer_id,
            device=device,
            dtype=dtype,
            max_length=max_length,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            steered_layers=resolved_layers,
        )
        validate_documents(rows, "steering_results.schema.json", context_prefix="steering")
        write_jsonl(out_path, rows)

        profile_path: Path | None = None
        if save_profile:
            profile_path = _resolve(Path(save_profile))
            profile_path.parent.mkdir(parents=True, exist_ok=True)
            if profile_path.exists():
                raise ArtError(f"Profile already exists (immutable): {profile_path}")
            save_model_id = str(artifact["model"]["model_id"])
            selected_layer = (
                int(resolved_layers[0])
                if resolved_layers and len(resolved_layers) > 0
                else int(artifact["layer_selection"]["selected_layer"])
            )
            profile = build_profile(
                profile_id=f"profile_{steer_run_id}",
                model_id=save_model_id,
                probe_name=resolved_probe_name,
                layers=[selected_layer],
                alpha=resolved_alpha,
                scope=resolved_scope,
            )
            validate_document(profile, "steering_profile.schema.json", context="steering_profile")
            write_json(profile_path, profile)

        outputs = {"steering_results": out_path}
        if profile_path:
            outputs["steering_profile"] = profile_path
        _write_meta(
            primary_output=out_path,
            command="steer.run",
            config={
                "backend": backend,
                "model_id": model_id or str(artifact["model"]["model_id"]),
                "tokenizer_id": tokenizer_id or str(artifact["model"]["tokenizer_id"]),
                "device": device,
                "dtype": dtype,
                "max_length": max_length,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "alpha": resolved_alpha,
                "scope": resolved_scope,
                "base_run_id": base_run_id,
                "probe_name": resolved_probe_name,
                "steered_layers": resolved_layers,
                "profile": str(profile_input_path) if profile_input_path is not None else "",
            },
            inputs={
                **{"probe_artifact": artifact_path, "cases": cases_path},
                **({"profile": profile_input_path} if profile_input_path is not None else {}),
            },
            outputs=outputs,
        )

        typer.echo(f"steering_results={out_path}")
        typer.echo(f"records={len(rows)}")
        if profile_path:
            typer.echo(f"steering_profile={profile_path}")
    except (ArtError, SchemaValidationError, OSError, ValueError) as exc:
        _exit_with_error(exc)


@steer_app.command("alpha-grid")
def steer_alpha_grid(
    alpha_min: float = typer.Option(DEFAULT_ALPHA_MIN, "--alpha-min"),
    alpha_max: float = typer.Option(DEFAULT_ALPHA_MAX, "--alpha-max"),
    alpha_step: float = typer.Option(DEFAULT_ALPHA_STEP, "--alpha-step"),
) -> None:
    """Print default alpha sweep grid."""

    if alpha_step <= 0:
        _exit_with_error(ArtError("--alpha-step must be > 0"))
    values: list[float] = []
    x = alpha_min
    while x <= alpha_max + 1e-9:
        values.append(round(x, 6))
        x += alpha_step
    typer.echo(",".join(str(v) for v in values))


@probe_app.command("diagnose")
def probe_diagnose(
    probe_artifact: Path = typer.Option(..., "--probe-artifact"),
    cases: Path = typer.Option(Path("data/probe_data.jsonl"), "--cases"),
    backend: str = typer.Option("transformers", "--backend"),
    model_id: str = typer.Option("", "--model-id"),
    tokenizer_id: str = typer.Option("", "--tokenizer-id"),
    device: str = typer.Option("auto", "--device"),
    dtype: str = typer.Option("auto", "--dtype"),
    max_length: int = typer.Option(DEFAULT_MAX_LENGTH, "--max-length"),
    max_new_tokens: int = typer.Option(96, "--max-new-tokens"),
    temperature: float = typer.Option(0.0, "--temperature"),
    include_layer_matrix: bool = typer.Option(False, "--include-layer-matrix"),
    out: str = typer.Option("", "--out"),
) -> None:
    """Run probe diagnosis and write diagnosis.jsonl."""

    try:
        artifact_path = _resolve(probe_artifact)
        cases_path = _resolve(cases)
        artifact = read_json(artifact_path)
        validate_document(artifact, "probe_artifact.schema.json", context="probe_artifact")

        case_rows = read_jsonl(cases_path)
        validate_documents(case_rows, "probe_data.schema.json", context_prefix="probe_data")

        run_id = make_run_id()
        if out:
            out_path = _resolve(Path(out))
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if out_path.exists():
                raise ArtError(f"Output already exists (immutable): {out_path}")
        else:
            run_dir = create_run_dir(run_id)
            out_path = run_dir / "diagnosis.jsonl"

        rows = run_diagnosis(
            probe_artifact=artifact,
            cases=case_rows,
            run_id=run_id,
            backend_name=backend,
            model_id=model_id,
            tokenizer_id=tokenizer_id,
            device=device,
            dtype=dtype,
            max_length=max_length,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            include_layer_matrix=include_layer_matrix,
        )
        validate_documents(rows, "diagnosis.schema.json", context_prefix="diagnosis")
        write_jsonl(out_path, rows)
        _write_meta(
            primary_output=out_path,
            command="probe.diagnose",
            config={
                "backend": backend,
                "model_id": model_id or str(artifact["model"]["model_id"]),
                "tokenizer_id": tokenizer_id or str(artifact["model"]["tokenizer_id"]),
                "device": device,
                "dtype": dtype,
                "max_length": max_length,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "include_layer_matrix": include_layer_matrix,
            },
            inputs={"probe_artifact": artifact_path, "cases": cases_path},
            outputs={"diagnosis": out_path},
        )
        typer.echo(f"diagnosis={out_path}")
        typer.echo(f"records={len(rows)}")
    except (ArtError, SchemaValidationError, OSError, ValueError) as exc:
        _exit_with_error(exc)


@steer_app.command("sweep")
def steer_sweep(
    probe_artifact: Path = typer.Option(..., "--probe-artifact"),
    probe_name: str = typer.Option(..., "--probe-name"),
    cases: Path = typer.Option(Path("data/probe_data.jsonl"), "--cases"),
    backend: str = typer.Option("transformers", "--backend"),
    model_id: str = typer.Option("", "--model-id"),
    tokenizer_id: str = typer.Option("", "--tokenizer-id"),
    device: str = typer.Option("auto", "--device"),
    dtype: str = typer.Option("auto", "--dtype"),
    max_length: int = typer.Option(DEFAULT_MAX_LENGTH, "--max-length"),
    max_new_tokens: int = typer.Option(96, "--max-new-tokens"),
    temperature: float = typer.Option(0.0, "--temperature"),
    alpha_min: float = typer.Option(DEFAULT_ALPHA_MIN, "--alpha-min"),
    alpha_max: float = typer.Option(DEFAULT_ALPHA_MAX, "--alpha-max"),
    alpha_step: float = typer.Option(DEFAULT_ALPHA_STEP, "--alpha-step"),
    scope: str = typer.Option("full_suite", "--scope"),
    base_run_id: str = typer.Option("baseline", "--base-run-id"),
    out: str = typer.Option("", "--out"),
    report_md: str = typer.Option("", "--report-md"),
    report_html: str = typer.Option("", "--report-html"),
) -> None:
    """Run alpha sweep and generate summary markdown/html reports."""

    try:
        if scope not in {"full_suite", "failures_only"}:
            raise ArtError("--scope must be one of: full_suite, failures_only")

        artifact_path = _resolve(probe_artifact)
        cases_path = _resolve(cases)
        artifact = read_json(artifact_path)
        validate_document(artifact, "probe_artifact.schema.json", context="probe_artifact")
        case_rows = read_jsonl(cases_path)
        validate_documents(case_rows, "probe_data.schema.json", context_prefix="probe_data")

        sweep_id = make_run_id()
        if out:
            out_path = _resolve(Path(out))
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if out_path.exists():
                raise ArtError(f"Output already exists (immutable): {out_path}")
            base_dir = out_path.parent
        else:
            run_dir = create_run_dir(sweep_id)
            out_path = run_dir / "steering_results.jsonl"
            base_dir = run_dir

        md_path = _resolve(Path(report_md)) if report_md else (base_dir / "steering_summary.md")
        html_path = _resolve(Path(report_html)) if report_html else (base_dir / "steering_summary.html")
        if md_path.exists():
            raise ArtError(f"Output already exists (immutable): {md_path}")
        if html_path.exists():
            raise ArtError(f"Output already exists (immutable): {html_path}")

        rows, md, html = run_alpha_sweep(
            probe_artifact=artifact,
            probe_name=probe_name,
            cases=case_rows,
            base_run_id=base_run_id,
            sweep_run_id=sweep_id,
            scope=scope,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            alpha_step=alpha_step,
            backend_name=backend,
            model_id=model_id,
            tokenizer_id=tokenizer_id,
            device=device,
            dtype=dtype,
            max_length=max_length,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        validate_documents(rows, "steering_results.schema.json", context_prefix="steering")
        write_jsonl(out_path, rows)
        write_text(md_path, md)
        write_text(html_path, html)
        _write_meta(
            primary_output=out_path,
            command="steer.sweep",
            config={
                "backend": backend,
                "model_id": model_id or str(artifact["model"]["model_id"]),
                "tokenizer_id": tokenizer_id or str(artifact["model"]["tokenizer_id"]),
                "device": device,
                "dtype": dtype,
                "max_length": max_length,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "alpha_min": alpha_min,
                "alpha_max": alpha_max,
                "alpha_step": alpha_step,
                "scope": scope,
                "base_run_id": base_run_id,
                "probe_name": probe_name,
            },
            inputs={"probe_artifact": artifact_path, "cases": cases_path},
            outputs={
                "steering_results": out_path,
                "report_md": md_path,
                "report_html": html_path,
            },
        )
        typer.echo(f"steering_results={out_path}")
        typer.echo(f"report_md={md_path}")
        typer.echo(f"report_html={html_path}")
        typer.echo(f"records={len(rows)}")
    except (ArtError, SchemaValidationError, OSError, ValueError) as exc:
        _exit_with_error(exc)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
