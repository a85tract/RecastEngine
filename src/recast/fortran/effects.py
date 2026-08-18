"""Side channels: what a subprogram does besides compute.

``interface.extract`` already reports read/write sets over module state. What
it does not report is the part of a routine's behaviour that has no return
value -- writing to a unit, halting the program, entering MPI. That is exactly
the part a translation cannot reproduce by matching numbers, and the part a
Verifier has to know about before it calls a kernel pure.

Small on purpose. This answers "is there a side channel here, and of what
kind", not "what does it print". A routine with an empty record here is one a
differential check can drive without capturing anything but its arguments.
"""

from __future__ import annotations

from typing import Any

from recast.fortran._parse import f03, walk

IO_STMTS: dict[str, type] = {
    "read": f03.Read_Stmt,
    "write": f03.Write_Stmt,
    "print": f03.Print_Stmt,
    "open": f03.Open_Stmt,
    "close": f03.Close_Stmt,
    "inquire": f03.Inquire_Stmt,
    "rewind": f03.Rewind_Stmt,
    "flush": f03.Flush_Stmt,
}
"""Statement kinds that touch a unit. Keyed by the word a report should use."""

MPI_PREFIX = "mpi_"


def side_channels(sub: Any) -> dict[str, Any]:
    """What ``sub`` does that a numeric comparison would not see.

    ``io`` and ``halts`` are the two that make a routine untranslatable as a
    pure function; ``mpi`` and ``allocates`` are reported because they change
    what an oracle has to stand up around the call, not because they are wrong.
    """
    exec_part = next((c for c in sub.children if isinstance(c, f03.Execution_Part)), None)
    if exec_part is None:
        return {"io": [], "halts": False, "mpi": [], "allocates": False}

    io = sorted(kind for kind, cls in IO_STMTS.items() if walk(exec_part, cls))
    mpi = sorted(
        {
            name
            for call in walk(exec_part, f03.Call_Stmt)
            if (name := str(call.children[0]).lower()).startswith(MPI_PREFIX)
        }
    )
    allocates = bool(walk(exec_part, (f03.Allocate_Stmt, f03.Deallocate_Stmt)))
    return {
        "io": io,
        "halts": bool(walk(exec_part, f03.Stop_Stmt)),
        "mpi": mpi,
        "allocates": allocates,
    }
