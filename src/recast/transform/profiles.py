"""What the compiler on the other side does.

Not a property of the source and not a property of the target: a description
of the Fortran compiler whose output the translation has to match. Two
compilers given the same source disagree on constructs whose answer is a
rounding away, and a translation matching one is failing against the other.

Target-independent on purpose. A Julia or C++ backend inherits the same
question -- did the reference build expand ``x**3`` or call ``pow`` -- and
should read the answer here rather than re-derive it.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["DEFAULT", "PROFILES", "Profile"]


@dataclass(frozen=True)
class Profile:
    """How one compiler lowers the constructs where lowering is observable."""

    name: str

    int_pow_expand: bool
    """``x**3`` expanded to repeated multiplication rather than a ``pow`` call.

    Square-and-multiply and ``pow`` do not round identically, so this changes
    the last bit of every integer power in the model.
    """

    cfold_mpfr: bool
    """Constant-argument intrinsics folded at compile time with MPFR.

    Correctly rounded, and therefore matching neither libgfortran nor glibc at
    run time -- ``gamma(1.8)`` differs from both. A translation that evaluates
    such a call at run time gets a different number than the reference binary
    has baked into it.
    """


PROFILES: dict[str, Profile] = {
    "gfortran": Profile("gfortran", int_pow_expand=True, cfold_mpfr=True),
    "ifx": Profile("ifx", int_pow_expand=False, cfold_mpfr=False),
    # Neither behaviour assumed. Preserves the source form, which is the only
    # choice that cannot be wrong in a way the operator did not ask for.
    "generic": Profile("generic", int_pow_expand=False, cfold_mpfr=False),
}

DEFAULT = "ifx"
"""What CESM's reference builds use. An operator comparing against a gfortran
build has to say so; there is no way to detect it from the source."""
