"""Carrying a candidate's probes to the reference source."""

from __future__ import annotations

import pytest

from recast.verify.probe_inject import ProbeInjectionError, extract_probes, inject

CANDIDATE = """\
#include <cstdio>
#include "../../../gate_sdk/gate.h"
int main() {
  float* f = new float[4];
  #pragma omp target teams distribute parallel for
  for (int i = 0; i < 4; i++) f[i] = i;
  if (f[3] == 3) {
    printf("PASS\\n");
  }

  GATE_CHECKSUM_U32("f_checksum",
                    reinterpret_cast<const uint32_t*>(f), 4);
  GATE_STATS_F32("f_stats", f, 4);

  delete[] f;
  return 0;
}
"""

SERIAL = """\
#include <cstdio>
int main() {
  float* f = new float[4];
  for (int i = 0; i < 4; i++) f[i] = i;
  if (f[3] == 3) {
    printf("PASS\\n");
  }
  delete[] f;
  return 0;
}
"""


def test_extract_finds_include_and_one_block_with_anchors() -> None:
    include, blocks = extract_probes(CANDIDATE)
    assert include is not None and include.strip() == '#include "../../../gate_sdk/gate.h"'
    assert len(blocks) == 1
    assert blocks[0].lines[0].strip().startswith("GATE_CHECKSUM_U32")
    assert blocks[0].lines[-1].strip() == 'GATE_STATS_F32("f_stats", f, 4);'
    assert blocks[0].anchor_before == "}"
    assert blocks[0].anchor_after == "delete[] f;"


def test_inject_places_block_and_a_bare_include() -> None:
    out = inject(SERIAL, CANDIDATE)
    lines = out.splitlines()
    assert lines[1] == '#include "gate.h"'
    i = next(k for k, ln in enumerate(lines) if "GATE_STATS_F32" in ln)
    assert lines[i + 1].strip() == "delete[] f;"
    assert "reinterpret_cast<const uint32_t*>(f), 4);" in out
    assert out.count("GATE_") == 2


def test_anchor_tolerates_std_prefix_and_drifted_print_text() -> None:
    cand = CANDIDATE.replace('printf("PASS\\n");', 'std::printf("PASS: %d\\n", 1);')
    assert inject(SERIAL, cand).count("GATE_") == 2


def test_ambiguous_anchor_is_refused() -> None:
    tail = '  if (f[3] == 3) {\n    printf("PASS\\n");\n  }\n  delete[] f;\n'
    with pytest.raises(ProbeInjectionError):
        inject(SERIAL.replace(tail, tail + tail), CANDIDATE)


def test_no_probes_and_already_probed_are_refused() -> None:
    with pytest.raises(ProbeInjectionError):
        inject(SERIAL, SERIAL)
    with pytest.raises(ProbeInjectionError):
        inject(CANDIDATE, CANDIDATE)
