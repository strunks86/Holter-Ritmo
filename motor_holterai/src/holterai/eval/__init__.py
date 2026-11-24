"""Evaluación de latidos, rachas y episodios al estilo EC57.

Tres bloques independientes:

* ``bxb``    — emparejamiento latido a latido (ventana 150 ms) con matriz
  AAMI N/S/V/F/Q + no detectado/sobrante y Se/+P/FPR.
* ``rxr``    — rachas de V y S: dupletas, 3–5 y 6 o más.
* ``epicmp`` — comparación de episodios (p. ej. FA) por episodio y duración.

``mitdb`` e ``holter`` añaden los cargadores que se usan desde la CLI; no se
importan aquí para que ``holterai.eval`` no arrastre numpy ni el modelo.
"""

from holterai.eval.bxb import (
    AAMI_CLASSES,
    NO_DETECTADO,
    SOBRANTE,
    WINDOW_S,
    BeatMatches,
    BxbResult,
    QualityStrata,
    RecordBeats,
    assign_stratum,
    combine_results,
    confusion_matrix,
    match_beats,
    mean_metrics,
    normalize_class,
    score_bxb,
    score_bxb_by_stratum,
    score_bxb_records,
)
from holterai.eval.epicmp import (
    Episode,
    compare_episodes,
    merge_episodes,
    parse_episodes,
)
from holterai.eval.rxr import (
    RunSummary,
    pool_runs,
    scan_runs,
    score_rxr,
)
from holterai.eval.table import markdown_table

__all__ = [
    "AAMI_CLASSES",
    "NO_DETECTADO",
    "SOBRANTE",
    "WINDOW_S",
    "BeatMatches",
    "BxbResult",
    "Episode",
    "QualityStrata",
    "RecordBeats",
    "RunSummary",
    "assign_stratum",
    "combine_results",
    "compare_episodes",
    "confusion_matrix",
    "markdown_table",
    "match_beats",
    "mean_metrics",
    "merge_episodes",
    "normalize_class",
    "parse_episodes",
    "pool_runs",
    "scan_runs",
    "score_bxb",
    "score_bxb_by_stratum",
    "score_bxb_records",
    "score_rxr",
]
