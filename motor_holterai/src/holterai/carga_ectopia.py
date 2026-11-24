"""Carga de ESV y EV por examen, según la práctica chilena (T120).

Los umbrales están en ``configs/criteria/carga_ectopia.yaml`` (fuente:
``/root/holter/tratados/CRITERIOS-TRATADOS.md``, sección B). Aquí solo hay
reglas: no se entrena nada.

Entrada: latidos ``(t_s, clase)``, con ``clase == "S"`` (ESV), ``"V"`` (EV) u
otra cosa (latido válido que no es ectopia), y tramos no analizables
``(inicio_s, fin_s)`` (ilegible, artefacto o corte). Los latidos dentro de esos
tramos no cuentan ni entran en el denominador, y las horas cuentan solo tiempo
analizable: la duración menos la unión de los tramos no analizables.
"""

from __future__ import annotations

import math
from bisect import bisect_left, bisect_right
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from holterai.config import load_yaml

DEFAULT_CARGA = (
    Path(__file__).resolve().parents[2] / "configs" / "criteria" / "carga_ectopia.yaml"
)
CLASE_ESV = "S"
CLASE_EV = "V"

Latido = tuple[float, str]
Tramo = tuple[float, float]


class _Esv(BaseModel):
    model_config = ConfigDict(extra="forbid")

    escasa_menor_que: float = Field(default=10, ge=0)
    frecuente_desde: float = Field(default=100, ge=0)
    muy_frecuente_desde: float = Field(default=500, ge=0)
    excesiva_hora_mayor_que: float = Field(default=30, ge=0)
    excesiva_24h_mayor_que: float = Field(default=218, ge=0)
    excesiva_salva_desde: int = Field(default=20, ge=1)


class _Ev(BaseModel):
    model_config = ConfigDict(extra="forbid")

    escasa_menor_que: float = Field(default=10, ge=0)
    frecuente_h: float = Field(default=30, ge=0)
    frecuente_pct: float = Field(default=3, ge=0)
    muy_frecuente_24h: float = Field(default=10000, ge=0)
    muy_frecuente_pct: float = Field(default=10, ge=0)


class _Horas(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hora_max_min_analizable_s: float = Field(default=1800, ge=0)


class CriteriosCarga(BaseModel):
    """Umbrales de carga de ESV y EV (ver el YAML para la fuente de cada uno)."""

    model_config = ConfigDict(extra="forbid")

    esv: _Esv = Field(default_factory=_Esv)
    ev: _Ev = Field(default_factory=_Ev)
    horas: _Horas = Field(default_factory=_Horas)
    source: str = "carga_ectopia_chile_2022"


@dataclass(frozen=True)
class ResumenCarga:
    horas_analizables: float
    horas_no_analizables: float
    latidos_validos: int
    esv: int
    esv_24h: float | None
    esv_h_max: float | None
    salva_max: int
    categoria_esv: str | None
    esv_excesiva: bool
    motivos_esv_excesiva: tuple[str, ...]
    ev: int
    ev_24h: float | None
    ev_h: float | None
    ev_pct: float | None
    categoria_ev: str | None

    def as_dict(self) -> dict[str, Any]:
        """Valores redondeados para informe y CSV (sin datos de paciente)."""

        def r(x: float | None, nd: int) -> float | None:
            return None if x is None else round(x, nd)

        return {
            "horas_analizables": r(self.horas_analizables, 2),
            "horas_no_analizables": r(self.horas_no_analizables, 2),
            "latidos_validos": self.latidos_validos,
            "esv": self.esv,
            "esv_24h": r(self.esv_24h, 1),
            "esv_h_max": r(self.esv_h_max, 1),
            "salva_max": self.salva_max,
            "categoria_esv": self.categoria_esv,
            "esv_excesiva": self.esv_excesiva,
            "motivos_esv_excesiva": list(self.motivos_esv_excesiva),
            "ev": self.ev,
            "ev_24h": r(self.ev_24h, 1),
            "ev_h": r(self.ev_h, 2),
            "ev_pct": r(self.ev_pct, 2),
            "categoria_ev": self.categoria_ev,
        }


def cargar_criterios(path: str | Path = DEFAULT_CARGA) -> CriteriosCarga:
    """Carga y valida los umbrales de carga; rechaza claves desconocidas."""
    return CriteriosCarga.model_validate(load_yaml(path))


def _fusionar(tramos: Sequence[Tramo], duracion_s: float) -> list[Tramo]:
    """Recorta a [0, duración), ordena y une tramos solapados o contiguos."""
    out: list[list[float]] = []
    for ini, fin in sorted((max(0.0, a), min(duracion_s, b)) for a, b in tramos):
        if fin <= ini:
            continue
        if out and ini <= out[-1][1]:
            out[-1][1] = max(out[-1][1], fin)
        else:
            out.append([ini, fin])
    return [(a, b) for a, b in out]


def _solape(a: float, b: float, tramos: Sequence[Tramo]) -> float:
    """Segundos de [a, b) cubiertos por tramos fusionados (ordenados, disjuntos)."""
    return sum(max(0.0, min(b, fin) - max(a, ini)) for ini, fin in tramos)


def _dentro(t: float, inicios: list[float], tramos: Sequence[Tramo]) -> bool:
    i = bisect_right(inicios, t) - 1
    return i >= 0 and t < tramos[i][1]


def _categoria_esv(esv24: float, c: _Esv) -> str:
    if esv24 >= c.muy_frecuente_desde:
        return "muy frecuente"
    if esv24 >= c.frecuente_desde:
        return "frecuente"
    if esv24 >= c.escasa_menor_que:
        return "ocasional"
    return "escasa"


def _categoria_ev(ev24: float, evh: float, pct: float | None, c: _Ev) -> str:
    pct_v = 0.0 if pct is None else pct
    if ev24 >= c.muy_frecuente_24h or pct_v >= c.muy_frecuente_pct:
        return "muy frecuente"
    if evh >= c.frecuente_h or pct_v >= c.frecuente_pct:
        return "frecuente"
    if ev24 < c.escasa_menor_que:
        return "escasa"
    return "ocasional"


def resumen_carga(
    latidos: Sequence[Latido],
    duracion_s: float,
    no_analizables: Sequence[Tramo] = (),
    criterios: CriteriosCarga | None = None,
) -> ResumenCarga:
    """Calcula la carga de ESV y EV de un examen con los umbrales dados.

    ``latidos`` con ``t_s`` fuera de [0, duracion_s) o dentro de un tramo no
    analizable se descartan. Si no queda tiempo analizable, las tasas y
    categorías son ``None``.
    """
    c = criterios or CriteriosCarga()
    dur = float(duracion_s)
    tramos = _fusionar(no_analizables, dur)
    inicios = [a for a, _ in tramos]
    t_no = sum(b - a for a, b in tramos)
    h_an = (dur - t_no) / 3600.0

    validos = sorted(
        (float(t), clase)
        for t, clase in latidos
        if 0.0 <= t < dur and not _dentro(float(t), inicios, tramos)
    )
    n_val = len(validos)
    esv_t = [t for t, k in validos if k == CLASE_ESV]
    n_esv = len(esv_t)
    n_ev = sum(1 for _, k in validos if k == CLASE_EV)

    # Salva: S seguidas sin N, V ni tramo no analizable entre ellas.
    salva_max = 0
    run = 0
    prev_t: float | None = None
    for t, k in validos:
        cortada = prev_t is not None and _hay_tramo(prev_t, t, inicios, tramos)
        if k == CLASE_ESV and not cortada:
            run += 1
        else:
            run = 1 if k == CLASE_ESV else 0
        salva_max = max(salva_max, run)
        prev_t = t

    if h_an <= 0:
        return ResumenCarga(
            horas_analizables=0.0, horas_no_analizables=t_no / 3600.0,
            latidos_validos=n_val, esv=n_esv, esv_24h=None, esv_h_max=None,
            salva_max=salva_max, categoria_esv=None, esv_excesiva=False,
            motivos_esv_excesiva=(), ev=n_ev, ev_24h=None, ev_h=None,
            ev_pct=None, categoria_ev=None,
        )

    esv24 = n_esv * 24.0 / h_an
    evh = n_ev / h_an
    ev24 = evh * 24.0
    pct = None if n_val == 0 else 100.0 * n_ev / n_val

    esv_h_max = _esv_hora_max(esv_t, dur, tramos, c.horas.hora_max_min_analizable_s)
    motivos: list[str] = []
    if esv_h_max is not None and esv_h_max > c.esv.excesiva_hora_mayor_que:
        motivos.append(f"ESV/h máxima {esv_h_max:.1f} > {c.esv.excesiva_hora_mayor_que:g}")
    if esv24 > c.esv.excesiva_24h_mayor_que:
        motivos.append(f"ESV/24 h {esv24:.1f} > {c.esv.excesiva_24h_mayor_que:g}")
    if salva_max >= c.esv.excesiva_salva_desde:
        motivos.append(f"salva de {salva_max} S (>= {c.esv.excesiva_salva_desde})")

    return ResumenCarga(
        horas_analizables=h_an,
        horas_no_analizables=t_no / 3600.0,
        latidos_validos=n_val,
        esv=n_esv,
        esv_24h=esv24,
        esv_h_max=esv_h_max,
        salva_max=salva_max,
        categoria_esv=_categoria_esv(esv24, c.esv),
        esv_excesiva=bool(motivos),
        motivos_esv_excesiva=tuple(motivos),
        ev=n_ev,
        ev_24h=ev24,
        ev_h=evh,
        ev_pct=pct,
        categoria_ev=_categoria_ev(ev24, evh, pct, c.ev),
    )


def _hay_tramo(prev_t: float, t: float, inicios: list[float],
               tramos: Sequence[Tramo]) -> bool:
    """¿Algún tramo no analizable cae entre dos latidos válidos consecutivos?"""
    i = bisect_left(inicios, t) - 1
    return i >= 0 and tramos[i][1] > prev_t


def _esv_hora_max(esv_t: list[float], dur: float, tramos: Sequence[Tramo],
                  min_analizable_s: float) -> float | None:
    """ESV/h máxima en horas de reloj desde el inicio, con tiempo analizable suficiente."""
    mejor: float | None = None
    for i in range(math.ceil(dur / 3600.0)):
        a, b = i * 3600.0, min((i + 1) * 3600.0, dur)
        an = (b - a) - _solape(a, b, tramos)
        if an < min_analizable_s or an <= 0:
            continue
        n = bisect_left(esv_t, b) - bisect_left(esv_t, a)
        tasa = n / (an / 3600.0)
        mejor = tasa if mejor is None else max(mejor, tasa)
    return mejor
