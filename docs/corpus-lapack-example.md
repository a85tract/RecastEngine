# A worked example: a library over LAPACK

[`corpus-numfor-example.md`](corpus-numfor-example.md) follows a module
that passes. This page follows the one module in the corpus whose generated
Python does not import -- `linalg`, from
[fortran-utils](https://github.com/certik/fortran-utils) -- and stays with
it long enough to say why that is the correct outcome, what the same
mechanism did for the module beside it, and what it would take to change
the answer.

Every console block below is the shipped `translate` recipe, with no domain
extension installed and no stage configured.

## Getting the source in place

```bash
git submodule update --init --depth 1 corpus/fortran-utils
python tools/corpus.py stage fortran-utils
```

`stage` copies the ten files `cases.json` lists under `src/` -- `types`,
`constants`, `utils`, `sorting`, `splines`, `special`, `optimize`, `mesh`,
`ppm`, `linalg` -- flat into `output/fortran-utils/staged/`, with a
`recast.json` beside them pinning the run's output to `output/fortran-utils/`.

Not on that list, and not in the repository at all: `lapack`. `linalg.f90`
opens with

```fortran
module linalg
  use types, only: dp
  use lapack, only: dsyevd, dsygvd, ilaenv, zgetri, zgetrf, zheevd, &
       dgeev, zgeev, zhegvd, dgesv, zgesv, dgetrf, dgetri, dgelsy, zgelsy, &
       dgesvd, zgesvd, dgeqrf, dorgqr, dpotrf, dtrtrs
  use utils, only: stop_error, assert
  use constants, only: i_
```

and `lapack` is the library's own interface module for the compiled LAPACK
it links against. `cases.json` describes the case as "linear algebra over
LAPACK interfaces", which is the whole of the matter: `linalg` is 1,120
lines and fifty procedures of argument checking and workspace sizing around
calls into a library that has no Fortran source in the tree and no Python
translation anywhere.

## The run

```console
$ recast run translate output/fortran-utils/staged \
      --config output/fortran-utils/staged/recast.json --unit fortran:linalg
fortran:linalg
  [ok ] frontend   fortran
  [ok ] transform  translate.numpy             65 deferred block(s)
  [FAIL] verifier   static.rwset                failed: 13/229 blocks disagree: deig/B011, zeig/B012, deigvals/B010, zeigvals/B011, deigh_simple/B009 (+8 more)
  [ok ] store      fs-evidence                 1 verdict(s) recorded

1 unit(s), 1 verdict(s), FAILED
```

Three things happened, and it is worth taking them in the order the
pipeline did rather than the order the console prints them.

One practical step first, though. A failed unit's emitted Python is not in
the output tree: the `candidate/` directory under a unit's workspace is
staged by the differential verifier, and this run never reached it. The
corpus harness is what writes every unit's emitted modules flat into
`output/fortran-utils/translated/` -- the import probe below needs them in
one directory -- so run it once. (`run` also rewrites the case's row in
`corpus/baseline.json`; at an unmodified checkout it rewrites it
byte-identically.)

```console
$ python tools/corpus.py run fortran-utils
== fortran-utils
case           units bare   mech  parse  import   rwset  deferred  top refusal
fortran-utils     10    0   4/10  10/10    9/10    4/10        89  48x call to external subroutine 'X'
...
```

### What the translation did with the LAPACK calls

The 65 deferred blocks split two ways: 35 are `call to external
subroutine 'X'` and 30 are generic-dispatch refusals (every one of them
`assert_shape`, use-imported from `utils`). The first group is
every `call dgeev(...)`, `call dsygvd(...)`, `call dgesv(...)` in the file.
The rule has no source for `dgeev`, no interface record for it, and no
externals table naming it, so it declines:

```text
    # B010 <- L118-L119 AGENT_QUEUE: call to external subroutine 'dgeev'
    raise NotImplementedError("call to external subroutine 'dgeev'")  # B010
```

That is the right answer. `dgeev` is a compiled routine; a rule that emitted
`_lapack.dgeev(...)` here would be spelling a call it has no reason to
believe exists, and a rule that guessed at `numpy.linalg.eig` would be
choosing a different algorithm with different rounding and calling it a
translation.

Two references survive as calls, because they are *function* references and
the expression rules resolve a use-imported name through the module it came
from:

```python
    nb = _lapack.ilaenv(1, 'DGETRI', 'UN', n, (-1), (-1), (-1))
```

So the emitted header carries the import that spelling needs:

```python
from linalg_constants import *  # noqa: F401,F403
import constants_numpy as _constants
import lapack_numpy as _lapack
import types_numpy as _types
import utils_numpy as _utils
```

Three of those four modules are in the tree, translated beside `linalg`,
and the frontend found them by walking the USE statements
(`FortranFrontend._companions`). The fourth is not, and the file does not
import:

```console
$ cd output/fortran-utils/translated && python -c 'import linalg_numpy'
    import lapack_numpy as _lapack
ModuleNotFoundError: No module named 'lapack_numpy'
```

### Why the import is kept, and when it is not

The header rule is: a USE that resolves to no companion gets
`import <mod>_numpy as _<mod>` **only if something in the body binds to the
alias**. That rule arrived with the translator's #18, and it is what took
the corpus from 51 of 59 units importing to 58: a `use shr_kind_mod, only:
r8` or `use fftpack_kind` takes one constant and references the alias
nowhere, so its import named a module that was never going to exist and
failed the file for nothing.

`linalg` binds to `_lapack` twice, so the import stays, and the failure it
produces is the honest one -- at import time, naming the dependency, before
any number is computed. Dropping the import would move the same failure to
the first call of `ilaenv`, as a `NameError`, after whatever ran before it
had produced output. The rule is deliberately not that clever.

`splines`, in the same case, shows the other side of the same rule, and it
is easy to read it wrongly. Its source also says `use lapack, only: dgesv,
dgbsv`, and its emitted header has no `lapack` import at all:

```python
from splines_constants import *  # noqa: F401,F403
import types_numpy as _types
import utils_numpy as _utils
```

That is not because `splines` needs LAPACK any less. Its three LAPACK sites
are `call` statements, all three refused as external subroutines, so nothing
in the emitted body ever spells `_lapack.` -- and an import nothing binds to
is dropped. `splines` imports, its 51 blocks match the source's, and it
still cannot solve a spline system: the three blocks that would are the
three it deferred, standing in the output as raises. The import column
says "the file loads", and for a library over LAPACK that is all it says.

### The read/write check, and what it is actually reporting

The verdict lists 13 blocks of 229, and they are all one shape:

```json
{"block": "deig/B011", "reads_source_only": ["n"], "reads_target_only": [],
 "writes_source_only": [], "writes_target_only": []}
```

The Fortran block reads `n` and the Python one does not -- a limit of the
translation or of the analysis, and the gate does not decide which. Either
way the stage fails closed, the oracle build and the differential never
run, and one verdict is recorded instead of three. 216 of 229 blocks
matching buys nothing.

## What it would take

The path is not a translation rule. It is the externals mechanism the
engine already has: an audited shim module for `lapack` -- `dgeev` over
`scipy.linalg.lapack.dgeev`, `ilaenv` over a block-size table -- declared
in the case's externals config, so the call rule emits `_ext.dgeev(...)`
where it now refuses, and the header imports the shim where it now imports a
file that does not exist. That turns 35 refusals into calls into a library
that is, after all, the same LAPACK the Fortran linked against, and it lets
`linalg` reach the oracle gate, where the question of whether the wrapper
logic around those calls is right can finally be asked.

It is a case decision, not an engine one, and it is deliberately not made
here: the corpus measures libraries as they arrive, and a shim written for
one of them is the beginning of a domain package. When it is made, this
page is the before.
