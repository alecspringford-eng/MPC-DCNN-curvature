from __future__ import annotations
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
import yaml
from dcnn_curvature.training.schedules import BetaSchedule, CurvatureDecay

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_yaml(path: str | Path) -> dict[str, Any]:
    with resolve_path(path).open("r") as f:
        data = yaml.safe_load(f)
    return data or {}


@dataclass
class DataConfig:
    raw_dir: str = "data/dbs_recordings"
    runs: dict[str, dict] = field(
        default_factory=lambda: {
            "system_identification": {"glob": "system_id_*.csv"},
            "controller_rollouts": {"glob": "simple_controllers_*.csv"},
        }
    )
    input_column: str = "inputs"
    output_column: str = "outputs"
    state_columns: list[str] | None = None
    control_columns: list[str] | None = None
    target_column: str | None = None
    target_offset: int = 1
    n_state_history: int = 20
    n_control_history: int = 20
    horizon: int = 1
    split_mode: Literal["by_run", "by_time_block"] = "by_run"
    val_fraction: float = 0.2
    test_fraction: float = 0.0
    split_seed: int = 42
    standardize: bool = True

    def resolved_state_columns(self) -> list[str]:
        return self.state_columns if self.state_columns is not None else [self.output_column]

    def resolved_control_columns(self) -> list[str]:
        return self.control_columns if self.control_columns is not None else [self.input_column]

    def resolved_target_column(self) -> str:
        return self.target_column if self.target_column is not None else self.output_column


@dataclass
class ICNNConfig:
    hidden_width: int = 32
    depth: int = 1


@dataclass
class ModelConfig:
    g: ICNNConfig = field(default_factory=ICNNConfig)
    h: ICNNConfig = field(default_factory=ICNNConfig)


@dataclass
class ActivationConfig:
    kind: Literal["relu", "softplus", "centered_softplus"] = "relu"
    beta_schedule: BetaSchedule = field(default_factory=BetaSchedule)
    curvature_decay: CurvatureDecay = field(default_factory=CurvatureDecay)
    learnable_beta: bool = False
    beta_min: float = 0.5
    lambda_beta: float = 0.0
    swap_to_relu_after_training: bool = False
    finetune_epochs_fraction: float = 0.08
    finetune_lr: float = 1.0e-4


@dataclass
class CurvatureConfig:
    mode: Literal[
        "none",
        "tangent_gap",
        "smoothhess_trace",
        "ccp_directional",
        "hybrid",
        "exact_hessian",
        "jacobian_variation",
        "smoothhess_directional",
    ] = "tangent_gap"
    target: Literal["g", "h", "both"] = "h"
    lambda_h: float = 0.1
    lambda_g: float = 0.0
    sigma: list[float] = field(default_factory=lambda: [0.05, 0.15, 0.30])
    multi_scale: bool = False
    n_pairs: int = 4
    distribution: Literal["gaussian", "rademacher"] = "gaussian"
    antithetic: bool = True
    direction_refresh_every_n_epochs: int = 0
    direction_refresh_drift_threshold: float = 0.0
    direction_n_solves: int = 8
    regularization_sampling_mode: Literal["random", "trajectory"] = "random"
    n_samples_per_batch: int = 256
    tail_weight: float = 0.0
    tail_alpha: float = 0.9
    location_realism_frac: float = 0.0
    lambda_h_adapt_step: float = 0.0
    lambda_h_adapt_min: float = 1e-6
    lambda_h_adapt_max: float = 10.0
    direction_weight_by_difficulty: bool = False
    direction_async_refresh: bool = False


@dataclass
class TrainingConfig:
    epochs: int = 100
    batch_size: int = 512
    lr: float = 1e-3
    warmup_epochs: int = 5
    patience: int = 10
    optimizer: Literal["adam", "adamw", "sgd"] = "adam"
    weight_decay: float = 0.0
    device: str = "auto"
    log_every_n_epochs: int = 1


@dataclass
class CCPConfig:
    max_iters: int = 50
    tol: float = 1e-6
    solver: str = "parametrized"
    delta_J_min: float | None = None
    tracking_prox_init_scale: float = 1e-3


@dataclass
class ClosedLoopConfig:
    plant: str = "placeholder"
    plant_kwargs: dict = field(default_factory=dict)
    Q: float = 1.0
    R: float = 0.01
    R_delta: float = 0.01
    u_min: float | list[float] = -3.0
    u_max: float | list[float] = 3.0
    du_max: float | list[float] = 1.0
    eval_delta_J_min: float | None = None


@dataclass
class LoggingConfig:
    results_dir: str = "results"
    checkpoint_dir: str = "experiments"
    run_name: str | None = None
    save_curvature_diagnostics: bool = True


@dataclass
class Config:
    seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    activation: ActivationConfig = field(default_factory=ActivationConfig)
    curvature: CurvatureConfig = field(default_factory=CurvatureConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    ccp: CCPConfig = field(default_factory=CCPConfig)
    closed_loop: ClosedLoopConfig = field(default_factory=ClosedLoopConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def _dataclass_from_dict(cls: type, data: dict[str, Any]) -> Any:
    hints = typing.get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for f in cls.__dataclass_fields__.values():  # type: ignore[attr-defined]
        if f.name not in data:
            continue
        value = data[f.name]
        field_type = hints.get(f.name)
        if (
            isinstance(field_type, type)
            and hasattr(field_type, "__dataclass_fields__")
            and isinstance(value, dict)
        ):
            kwargs[f.name] = _dataclass_from_dict(field_type, value)
        else:
            kwargs[f.name] = value
    return cls(**kwargs)


def validate_curvature_mode(mode: str) -> None:
    from dcnn_curvature.curvature.losses import CURVATURE_MODES

    if mode not in CURVATURE_MODES:
        raise ValueError(
            f"curvature.mode={mode!r} is not implemented; valid modes: {sorted(CURVATURE_MODES)}"
        )


def validate_activation(config: Config) -> None:
    if config.activation.swap_to_relu_after_training and config.activation.kind == "relu":
        raise ValueError(
            "activation.swap_to_relu_after_training=True has no effect when "
            "activation.kind is already 'relu'; there is no softplus model to swap"
        )
    if config.curvature.mode == "exact_hessian" and config.activation.kind == "relu":
        raise ValueError(
            "curvature.mode='exact_hessian' under activation.kind='relu' is a silent "
            "no-op: ReLU's autograd Hessian is ~0 almost everywhere. "
            "Set activation.kind='softplus'."
        )


def validate_closed_loop_plant(plant: str) -> None:
    from dcnn_curvature.mpc.plants import PLANT_REGISTRY

    if plant not in PLANT_REGISTRY:
        raise ValueError(
            f"closed_loop.plant={plant!r} is not implemented; valid plants: {sorted(PLANT_REGISTRY)}"
        )


def validate_ccp_solver(solver: str) -> None:
    valid = ("naive", "parametrized")
    if solver not in valid:
        raise ValueError(f"ccp.solver={solver!r} is not implemented; valid solvers: {valid}")


def validate_search_space(path: str | Path) -> None:
    with open(path) as fh:
        raw = yaml.safe_load(fh)

    def _resolve(cls: type, parts: list[str], dotted_key: str) -> None:
        hints = typing.get_type_hints(cls)
        head, *rest = parts
        if head not in hints:
            raise ValueError(
                f"search_space key {dotted_key!r} has no matching Config field ({head!r} not found)"
            )
        if not rest:
            return
        field_type = hints[head]
        if not (isinstance(field_type, type) and hasattr(field_type, "__dataclass_fields__")):
            raise ValueError(f"search_space key {dotted_key!r} continues past leaf field {head!r}")
        _resolve(field_type, rest, dotted_key)

    for dotted_key in raw.get("search_space", {}):
        _resolve(Config, dotted_key.split("."), dotted_key)


DEFAULT_CONFIG_PATH = "configs/default.yaml"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path) -> Config:
    path = resolve_path(path)
    raw = load_yaml(path)
    default_path = resolve_path(DEFAULT_CONFIG_PATH)
    if path != default_path and default_path.exists():
        raw = _deep_merge(load_yaml(default_path), raw)
    config = _dataclass_from_dict(Config, raw)
    validate_curvature_mode(config.curvature.mode)
    validate_activation(config)
    validate_closed_loop_plant(config.closed_loop.plant)
    validate_ccp_solver(config.ccp.solver)
    return config
