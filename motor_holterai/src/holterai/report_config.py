"""Configuración local de la cabecera/firma del informe."""

from __future__ import annotations

import base64
import json
import mimetypes
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class ReportConfig:
    clinic: str = ""
    doctor: str = ""
    signature: str = ""
    footer: str = ""
    logo: str = ""

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def config_path(path: str | Path | None = None) -> Path:
    return Path(path) if path else Path("data/report_config.json")


def load_report_config(path: str | Path | None = None) -> ReportConfig:
    p = config_path(path)
    if not p.is_file():
        return ReportConfig()
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return ReportConfig()
    if not isinstance(raw, dict):
        return ReportConfig()
    return ReportConfig(
        clinic=str(raw.get("clinic", "")),
        doctor=str(raw.get("doctor", "")),
        signature=str(raw.get("signature", "")),
        footer=str(raw.get("footer", "")),
        logo=str(raw.get("logo", "")),
    )


def save_report_config(cfg: ReportConfig, path: str | Path | None = None) -> Path:
    p = config_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cfg.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    return p


def logo_data_uri(logo: str) -> str:
    """Convierte el logo a data URI para un HTML autocontenido."""
    if not logo:
        return ""
    p = Path(logo)
    if not p.is_file():
        return ""
    mime = mimetypes.guess_type(p.name)[0] or "image/png"
    data = base64.b64encode(p.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"
