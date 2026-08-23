"""Statement-block chunking.

Migrated from the source pipeline's ``pipeline/chunk.py``. Splits a
subprogram's execution part into numbered blocks at one statement or one whole
construct each, so that every later stage -- translation, instrumentation,
read/write-set analysis -- addresses the same piece of code by the same id.

``chunk_subprogram`` was already the most-imported symbol in the pipeline it
came from, for exactly that reason: block boundaries are a shared vocabulary,
and stages that disagree about them cannot have their results lined up.
"""

from __future__ import annotations

from typing import Any

from recast.fortran._parse import f03
from recast.fortran.interface import node_span

BLOCK_TYPES: tuple[type, ...] = (
    f03.Assignment_Stmt,
    f03.Block_Nonlabel_Do_Construct,
    f03.If_Construct,
    f03.If_Stmt,
    f03.Case_Construct,
    f03.Where_Construct,
    f03.Where_Stmt,
    f03.Call_Stmt,
    f03.Return_Stmt,
    f03.Write_Stmt,
)
"""Statement and construct types this frontend claims to understand.

A block whose type is outside this tuple still gets an id and a span -- it is
reported, not dropped. ``known_type`` is how a Transform decides whether to
translate it or put it on the agent queue.
"""


def chunk_subprogram(sub: Any) -> list[tuple[str, Any, tuple[int | None, int | None]]]:
    """``[(block_id, node, (line_lo, line_hi)), ...]`` for one subprogram node.

    Ids are scoped per subprogram and assigned in source order: ``B001``,
    ``B002``. They are positional by construction, so inserting a statement
    renumbers everything after it -- ids address a revision, not a lineage.
    """
    exec_part = next((c for c in sub.children if isinstance(c, f03.Execution_Part)), None)
    if exec_part is None:
        return []
    blocks: list[tuple[str, Any, tuple[int | None, int | None]]] = []
    for stmt in exec_part.children:
        blocks.append((f"B{len(blocks) + 1:03d}", stmt, node_span(stmt)))
    return blocks


def blocks_of(sub: Any, source_lines: list[str]) -> list[dict[str, Any]]:
    """``chunk_subprogram`` as records, with the verbatim source of each block."""
    out: list[dict[str, Any]] = []
    for bid, stmt, (lo, hi) in chunk_subprogram(sub):
        out.append(
            {
                "id": bid,
                "type": type(stmt).__name__,
                "known_type": isinstance(stmt, BLOCK_TYPES),
                "span": [lo, hi],
                "fortran": "\n".join(source_lines[lo - 1 : hi]) if lo else str(stmt),
            }
        )
    return out
