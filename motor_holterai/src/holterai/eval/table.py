"""Tablas Markdown para los informes de evaluación."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any


def fmt(value: Any, digits: int = 3) -> str:
    """Formatea números para tabla: enteros sin decimales, floats redondeados."""
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "sí" if value else "no"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value != value:  # NaN
            return "—"
        return f"{value:.{digits}f}"
    return str(value)


def _cell(value: Any) -> str:
    return "" if value is None else str(value).replace("|", "\\|")


def markdown_table(
    headers: Sequence[str],
    rows: Iterable[Sequence[Any]],
    align: str | None = None,
) -> str:
    """Tabla Markdown con encabezado y filas ya formateadas.

    ``align`` admite ``"l"``, ``"c"`` o ``"r"`` (por defecto texto a la
    izquierda y números a la derecha se dejan a cargo del llamador).
    """
    head = "| " + " | ".join(_cell(h) for h in headers) + " |"
    if align is None:
        sep = "|" + "|".join("---" for _ in headers) + "|"
    else:
        marks = {"l": ":---", "c": ":---:", "r": "---:"}
        try:
            sep = "|" + "|".join(marks[a] for a in align) + "|"
        except KeyError as exc:
            raise ValueError(f"alineación no válida: {exc.args[0]!r}") from exc
    body = ["| " + " | ".join(_cell(c) for c in row) + " |" for row in rows]
    return "\n".join([head, sep, *body])
