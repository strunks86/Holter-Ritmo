"""Carga y validación de configuración YAML.

Entradas: rutas a YAML de hardware/criterios.
Salidas: modelos pydantic validados.
Unidades: las de cada campo (ms, lpm, µV, %).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, cast

from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field

QtcFormula = Literal["bazett", "fridericia", "framingham", "hodges"]
_DEFAULT_QTC: tuple[QtcFormula, ...] = ("bazett", "fridericia", "framingham", "hodges")

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HARDWARE = _REPO_ROOT / "configs" / "hardware" / "cpu.yaml"
DEFAULT_CRITERIA = _REPO_ROOT / "configs" / "criteria" / "hscribe_default.yaml"


class HardwareConfig(BaseModel):
    """Perfil de hardware para entrenamiento/inferencia."""

    model_config = ConfigDict(extra="forbid")

    device: Literal["cpu", "cuda", "rocm", "auto"] = "cpu"
    batch_size: int = Field(default=32, gt=0)
    precision: Literal["float32", "float16", "bfloat16"] = "float32"
    num_workers: int = Field(default=0, ge=0)


class CriteriaConfig(BaseModel):
    """Umbrales clínicos por defecto (se sobreescriben con ExamSettings.xml)."""

    model_config = ConfigDict(extra="forbid")

    sv_prematurity_pct: float = Field(default=25.0, ge=0.0, le=100.0)
    pause_ms: int = Field(default=2000, ge=0)
    pause_scope: Literal["all_beats", "normal_to_normal"] = "all_beats"
    st_depression_uv: int = Field(default=100, ge=0)
    st_elevation_uv: int = Field(default=200, ge=0)
    st_j_ms: Literal[60, 80] = 60
    st_min_duration_s: int = Field(default=60, ge=0)
    tachycardia_bpm: int = Field(default=120, gt=0)
    bradycardia_bpm: int = Field(default=50, gt=0)
    min_tachy_brady_duration_s: int = Field(default=180, ge=0)
    vt_bpm: int = Field(default=100, gt=0)
    vt_min_beats: int = Field(default=3, ge=1)
    svt_bpm: int = Field(default=100, gt=0)
    svt_min_beats: int = Field(default=3, ge=1)
    af_auto_detect: bool = True
    hr_scope: Literal["all_beats", "normal_only"] = "all_beats"
    exclude_pause_from_hr: bool = False
    pacemaker_analysis: bool = False
    pacemaker_rate_bpm: int = Field(default=60, gt=0)
    # Ventanas día/noche de la HRV, en hora local (extremo final excluido).
    day_window_start_h: int = Field(default=8, ge=0, le=23)
    day_window_end_h: int = Field(default=20, ge=0, le=23)
    night_window_start_h: int = Field(default=23, ge=0, le=23)
    night_window_end_h: int = Field(default=7, ge=0, le=23)
    qtc_formulas: list[QtcFormula] = Field(default_factory=lambda: list(_DEFAULT_QTC))
    qtc_formula: QtcFormula = "bazett"
    source: str = "hscribe_default"


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Carga un YAML y devuelve un diccionario plano."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"no existe el YAML: {p}")
    raw = OmegaConf.to_container(OmegaConf.load(str(p)), resolve=True)
    if not isinstance(raw, dict):
        raise ValueError(f"el YAML no es un mapeo: {p}")
    return cast(dict[str, Any], raw)


def load_hardware(path: str | Path = DEFAULT_HARDWARE) -> HardwareConfig:
    """Carga y valida un perfil de hardware."""
    return HardwareConfig.model_validate(load_yaml(path))


def load_criteria(path: str | Path = DEFAULT_CRITERIA) -> CriteriaConfig:
    """Carga y valida los criterios clínicos por defecto."""
    return CriteriaConfig.model_validate(load_yaml(path))
