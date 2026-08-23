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

    intel_math: bool = False
    """Transcendentals taken from Intel's libimf rather than the system libm.

    ``ifx`` links libimf, and its ``exp``/``log``/``pow`` differ from glibc's
    by an ULP on some arguments. A translation held to an Intel-built
    reference has to call the same library, which the NumPy backend does
    through ``intel_math`` (a ctypes binding, loaded when first called). The
    output then *needs* a libimf to run: a library dependency by design, not
    a path one, and ``gfortran`` is the profile for a machine without one.
    """


PROFILES: dict[str, Profile] = {
    "gfortran": Profile("gfortran", int_pow_expand=True, cfold_mpfr=True),
    "ifx": Profile("ifx", int_pow_expand=False, cfold_mpfr=False, intel_math=True),
    # Neither behaviour assumed. Preserves the source form, which is the only
    # choice that cannot be wrong in a way the operator did not ask for.
    "generic": Profile("generic", int_pow_expand=False, cfold_mpfr=False),
}

DEFAULT = "ifx"
"""What CESM's reference builds use. An operator comparing against a gfortran
build has to say so; there is no way to detect it from the source."""
