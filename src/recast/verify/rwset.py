"""``static.rwset``: cross-check a translation's dataflow against the source's.

Migrated from CESM-language-translator ``pipeline/rwset.py``, target half. For
every block, the source's read and write sets are compared against the sets of
the code the Transform emitted for it. Any inequality fails the block, which
keeps it out of differential testing and sends it to the agent queue instead.

This is the cheapest gate there is -- milliseconds, no compiler, no oracle --
and it catches the transform bug that a differential test is worst at
catching: a translation that runs, produces plausible numbers, and reads or
writes the wrong variable. A numerical comparison only notices that if the
inputs happen to make the difference visible.

Nothing here knows Fortran. The source side arrives in ``Candidate.notes``
already reduced to sets of names, put there by the Transform out of ``Facts``.
Any frontend that reports per-block read/write sets, and any Transform that
says which output lines it emitted for each block, can be gated by this
verifier -- the two halves only ever meet through a set equality.

The comparison is symmetric on purpose. An extra read on the target side is as
much a failure as a missing one: it means the translation depends on something
the source does not, which is how a supposedly pure kernel picks up a
dependency on state nobody intended it to see.
"""

from __future__ import annotations

import ast
import keyword
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from recast.model import Candidate, Confidence, Unit, Verdict
from recast.plugins.executor import Executor
from recast.plugins.verifier import StaticVerifier

LITERALS = frozenset({"None", "True", "False"})
"""The only emitted names that can never stand for a source symbol.

Deliberately this short. ``sum``, ``min``, ``len`` and ``int`` look like Python
builtins and are perfectly ordinary Fortran variable names -- CESM has a local
called ``sum`` -- so a verifier that skipped them by default would silently
drop a real dataflow edge on exactly the code most likely to have one.
Everything a particular backend emits is backend knowledge and arrives in the
protocol below instead.
"""

HOISTED_LITERAL = re.compile(r"[FI]_[0-9EMP]+")
"""``F_273P15``, ``I_5``. A hoisted literal is a constant, not a variable read."""

DISCARD = re.compile(r"_wm\d*|_|_g")
"""Scaffolding targets: a discarded value, a where-mask, a region label."""

PRESENT_SENTINEL = re.compile(r"want_(\w+)")
"""``want_x`` is how an optional output argument is spelled on the target side;
it corresponds to ``present(x)`` on the source side, which counts as a read of
``x``. A gate that saw one and not the other would fail every optional
argument."""


@dataclass
class Protocol:
    """What the Transform has to say for its output to be checkable.

    A read/write cross-check is not something a Verifier can do to an opaque
    artifact: only the Transform knows which output lines it produced for which
    input block, and only it knows what it renamed. Requiring it to say so is
    not overhead -- a Transform that cannot answer either question has not
    recorded enough to explain its own output to anyone.
    """

    blocks: list[dict[str, Any]] = field(default_factory=list)
    """``[{"subprogram", "block", "reads", "writes", "lines": [lo, hi]}, ...]``.

    ``reads``/``writes`` are the source's, copied from ``Facts.effects``;
    ``lines`` is the 1-based inclusive span in the emitted file.
    """

    file: str = ""
    """Which of ``Candidate.files`` the spans index into."""

    names: dict[str, str] = field(default_factory=dict)
    """Emitted name -> source name, for anything the Transform renamed."""

    procedures: frozenset[str] = frozenset()
    """Emitted names that stand for a source procedure.

    Skipped as reads -- a call is control flow, not dataflow -- with two
    refinements the Fortran result convention forces. A *store* to one is not
    skipped: assigning to a function's own name is how Fortran returns a
    result, and dropping it would lose the write the whole routine exists to
    make. And a *load* of the block's own subprogram name is not skipped
    either, because inside ``f`` the name ``f`` is the result variable --
    ``no_limiter = transfer(limiter_off, no_limiter)`` reads real data. A
    load at callee position stays skipped even then, so recursion does not
    count as a read.
    """

    reserved: frozenset[str] = frozenset()
    """Names the target file uses itself, which the Transform renamed around.

    A Fortran variable called ``np`` collides with the emitted module alias, so
    it is emitted as ``np_`` and has to be read back as ``np``. Which names
    those are is the backend's choice -- a backend importing different modules
    reserves different names -- so it says, rather than this verifier assuming.

    Python keywords need no declaration: ``lambda_`` reads back as ``lambda``
    because ``lambda`` is a keyword, which is a fact about the language rather
    than about any backend.
    """

    aliases: frozenset[str] = frozenset()
    """Module aliases whose attributes are a sibling translation's globals.

    ``_wv.omeps`` is a write of the sibling's ``omeps``, not machinery: the
    source spells the same thing as a bare use-imported name, and dropping it
    would blind the gate to every cross-module state access. An attribute
    that names a procedure stays skipped -- ``_wv.wv_sat_svp_water(t)`` is a
    call -- which is why ``procedures`` carries the siblings' names too.
    """

    scaffolding: frozenset[str] = frozenset()
    """Emitted names that are never data -- runtime shims, module aliases, the
    machinery a backend needs to spell a construct Python does not have.

    Skipped in both directions, unlike ``procedures``: a store to a runtime
    helper is bookkeeping, not a result.
    """

    @classmethod
    def from_notes(cls, notes: dict[str, Any]) -> Protocol | None:
        record = notes.get("rwset")
        if not isinstance(record, dict) or "blocks" not in record:
            return None
        return cls(
            blocks=list(record["blocks"]),
            file=str(record.get("file", "")),
            names={str(k): str(v) for k, v in (record.get("names") or {}).items()},
            procedures=frozenset(record.get("procedures") or ()),
            aliases=frozenset(record.get("aliases") or ()),
            reserved=frozenset(record.get("reserved") or ()),
            scaffolding=frozenset(record.get("scaffolding") or ()),
        )


class _Visitor(ast.NodeVisitor):
    """Read and write sets of emitted Python, in source-side vocabulary."""

    def __init__(self, protocol: Protocol, own: str = "") -> None:
        self.reads: set[str] = set()
        self.writes: set[str] = set()
        self.protocol = protocol
        self.own = own
        """The subprogram this block belongs to: a load of this name is its
        result variable, not a call."""

    # -- name mapping ---------------------------------------------------------

    def back(self, name: str, *, store: bool) -> str | None:
        """The source symbol an emitted name stands for, or ``None`` for none."""
        if DISCARD.fullmatch(name):
            return None
        if name.endswith("_") and (
            keyword.iskeyword(name[:-1]) or name[:-1] in self.protocol.reserved
        ):
            stripped = name[:-1]
            return self.protocol.names.get(stripped, stripped.lower())
        sentinel = PRESENT_SENTINEL.fullmatch(name)
        if sentinel and not store:
            return sentinel.group(1)
        if name in LITERALS or name in self.protocol.scaffolding:
            return None
        # A *store* to a procedure name is the Fortran result-variable
        # convention (`function f(...)` assigning to `f`), not a call; so is
        # a *load* of the block's own name.
        if name in self.protocol.procedures and not store and name != self.own:
            return None
        if HOISTED_LITERAL.fullmatch(name):
            return None
        return self.protocol.names.get(name, name.lower())

    def record(self, name: str, *, store: bool) -> None:
        mapped = self.back(name, store=store)
        if mapped is None:
            return
        (self.writes if store else self.reads).add(mapped)

    # -- nodes ----------------------------------------------------------------

    def visit_Name(self, node: ast.Name) -> None:
        self.record(node.id, store=isinstance(node.ctx, ast.Store))

    def visit_Call(self, node: ast.Call) -> None:
        """A procedure name at callee position is a call even when it is the
        block's own name -- recursion is not a read of the result variable."""
        callee = node.func
        if not (isinstance(callee, ast.Name) and callee.id in self.protocol.procedures):
            self.visit(callee)
        for argument in node.args:
            self.visit(argument)
        for keyword_argument in node.keywords:
            self.visit(keyword_argument.value)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        """``a.b`` is a read or write of ``a``; ``b`` is a component of it.

        The source side says the same -- a derived-type component name is an
        attribute, not a symbol -- so counting it here would fail every
        structure access.
        """
        root: ast.expr = node
        while isinstance(root, (ast.Attribute, ast.Subscript)):
            root = root.value
        if isinstance(root, ast.Name) and root.id in self.protocol.aliases:
            attribute: ast.expr = node
            while isinstance(attribute, (ast.Attribute, ast.Subscript)) and isinstance(
                attribute.value, (ast.Attribute, ast.Subscript)
            ):
                attribute = attribute.value  # the innermost alias.X attribute
            if isinstance(attribute, ast.Attribute):
                name = attribute.attr
                if name not in self.protocol.procedures:
                    mapped = self.protocol.names.get(name, name.lower())
                    (self.writes if isinstance(node.ctx, ast.Store) else self.reads).add(mapped)
        elif isinstance(root, ast.Name):
            self.record(root.id, store=isinstance(node.ctx, ast.Store))
        else:
            self.generic_visit(node)
        for inner in ast.walk(node):
            if isinstance(inner, ast.Subscript):
                self.visit(inner.slice)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        root: ast.expr = node.value
        while isinstance(root, ast.Subscript):
            root = root.value
        if isinstance(root, ast.Name):
            self.record(root.id, store=isinstance(node.ctx, ast.Store))
        elif isinstance(root, ast.Attribute):
            self.visit_Attribute(
                ast.copy_location(
                    ast.Attribute(value=root.value, attr=root.attr, ctx=node.ctx), root
                )
            )
        else:
            self.visit(node.value)  # subscript of an expression, e.g. a masked RHS
        self.visit(node.slice)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        pass  # a nested definition: its dataflow is at the call sites

    def visit_Return(self, node: ast.Return) -> None:
        pass  # signature plumbing, not dataflow

    def visit_Global(self, node: ast.Global) -> None:
        pass


def span_rwset(
    tree: ast.Module, lo: int, hi: int, protocol: Protocol, own: str = ""
) -> tuple[set[str], set[str]]:
    """Read and write sets of the statements between lines ``lo`` and ``hi``.

    Walks the whole file's tree rather than re-parsing the slice, because a
    block's emitted lines can sit inside scaffolding -- a loop header, a
    try/except -- that straddles the boundary. Re-parsing the slice on its own
    would either fail or silently lose the header.

    Which statements belong to a block is decided by line span and nothing
    else. The pipeline this came from also skipped any ``if`` whose test
    mentioned its goto-region label, matched as the substring ``_g`` in a dump
    of the test -- which also matches ``want_gam``, so every optional-output
    block for a variable named ``gam`` was dropped from verification without
    saying so. Names that are scaffolding are neutralised where names are
    resolved; statements are never skipped for containing one.
    """
    visitor = _Visitor(protocol, own)

    def within(node: ast.AST) -> bool:
        line = getattr(node, "lineno", None)
        return line is not None and lo <= line <= hi

    def walk_stmt(node: ast.stmt) -> None:
        if not hasattr(node, "lineno"):
            return
        bodies = [
            getattr(node, attr)
            for attr in ("body", "orelse", "finalbody")
            if getattr(node, attr, None)
        ]
        if isinstance(node, ast.Try):
            # Never taken whole: a Try is scaffolding and its handlers carry
            # control flow the source spells as a goto, not as dataflow.
            for body in bodies:
                for stmt in body:
                    walk_stmt(stmt)
            for handler in node.handlers:
                for stmt in handler.body:
                    walk_stmt(stmt)
            return
        contained = within(node) and all(
            within(stmt) for body in bodies for stmt in body if hasattr(stmt, "lineno")
        )
        if contained:
            if not isinstance(node, ast.Raise):  # a raise is scaffolding
                visitor.visit(node)
            return
        # Partially covered: take the header if its own line is in the span,
        # then descend. The alternative -- taking the whole statement -- would
        # attribute a neighbouring block's body to this one.
        if within(node):
            if isinstance(node, (ast.If, ast.While)):
                visitor.visit(node.test)
            elif isinstance(node, ast.For):
                visitor.visit(node.target)
                visitor.visit(node.iter)
        for body in bodies:
            for stmt in body:
                walk_stmt(stmt)

    for top in tree.body:
        if isinstance(top, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for stmt in top.body:
                walk_stmt(stmt)
        else:
            walk_stmt(top)
    return visitor.reads, visitor.writes


class ReadWriteSetVerifier(StaticVerifier):
    """Fail any block whose translation reads or writes a different set."""

    name = "static.rwset"
    provides = Confidence.SAMPLED

    def check(
        self,
        unit: Unit,
        candidate: Candidate,
        workspace: Path,
        executor: Executor,
        config: dict[str, Any],
    ) -> Verdict:
        protocol = Protocol.from_notes(candidate.notes)
        if protocol is None:
            # Fail closed. A Transform that records nothing is not a Transform
            # that passed; it is one this gate could not read.
            return self._verdict(
                candidate,
                Confidence.FAILED,
                {"blocks_checked": 0},
                f"transform {candidate.transform!r} recorded no rwset protocol in "
                "Candidate.notes, so its dataflow cannot be cross-checked",
            )

        source = self._emitted_source(candidate, protocol)
        if source is None:
            return self._verdict(
                candidate,
                Confidence.FAILED,
                {"blocks_checked": 0},
                f"candidate has no file {protocol.file!r} for its rwset spans to index",
            )
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            return self._verdict(
                candidate,
                Confidence.FAILED,
                {"blocks_checked": 0},
                f"emitted file does not parse: {exc}",
            )

        waivers = {str(k): str(v) for k, v in (config.get("waivers") or {}).items()}
        deferred = set(candidate.deferred)

        checked = 0
        waived: list[str] = []
        failures: list[dict[str, Any]] = []
        for block in protocol.blocks:
            key = f"{block['subprogram']}/{block['block']}"
            if key in deferred:
                continue  # no mechanical translation exists; nothing to compare
            if key in waivers:
                waived.append(f"{key}: {waivers[key]}")
                continue
            lo, hi = block["lines"]
            reads, writes = span_rwset(
                tree, int(lo), int(hi), protocol, own=str(block["subprogram"])
            )
            checked += 1
            want_reads, want_writes = set(block["reads"]), set(block["writes"])
            if reads != want_reads or writes != want_writes:
                failures.append(
                    {
                        "block": key,
                        "reads_source_only": sorted(want_reads - reads),
                        "reads_target_only": sorted(reads - want_reads),
                        "writes_source_only": sorted(want_writes - writes),
                        "writes_target_only": sorted(writes - want_writes),
                    }
                )

        metrics = {
            "blocks_checked": checked,
            "blocks_matched": checked - len(failures),
            "blocks_deferred": len(deferred),
            "blocks_waived": len(waived),
            "failures": failures,
        }
        if failures:
            listed = ", ".join(f["block"] for f in failures[:5])
            more = f" (+{len(failures) - 5} more)" if len(failures) > 5 else ""
            return self._verdict(
                candidate,
                Confidence.FAILED,
                metrics,
                f"{len(failures)}/{checked} blocks disagree: {listed}{more}",
            )
        # Every waiver is named in the detail. A gate that passed because
        # something was skipped has to say what it skipped.
        detail = f"{checked} blocks match"
        if waived:
            detail += "; waived " + "; ".join(waived)
        return self._verdict(candidate, Confidence.SAMPLED, metrics, detail)

    def _verdict(
        self, candidate: Candidate, confidence: Confidence, metrics: dict[str, Any], detail: str
    ) -> Verdict:
        return Verdict(
            unit=candidate.unit,
            candidate=candidate.digest(),
            verifier=self.name,
            confidence=confidence,
            metrics=metrics,
            detail=detail,
        )

    @staticmethod
    def _emitted_source(candidate: Candidate, protocol: Protocol) -> str | None:
        for path, content in candidate.files.items():
            if not protocol.file or str(path) == protocol.file:
                return content.decode()
        return None


def factory(**_config: Any) -> ReadWriteSetVerifier:
    return ReadWriteSetVerifier()
