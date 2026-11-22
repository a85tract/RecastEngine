import RecastProofs.Rules.StmtSoundness

/-!
# Kernel demonstration: `saxpy`

    subroutine saxpy(n, a, x, y)
      integer, intent(in) :: n
      real(8), intent(in) :: a, x(n)
      real(8), intent(inout) :: y(n)
      integer :: i
      do i = 1, n
        y(i) = a * x(i) + y(i)
      end do
    end subroutine

The baseline engine emits (`proofs/examples/FNP-kernel/emitted/kernel_mod_numpy.py`):

    for i in range(1, n + 1):
        y[i - 1] = ((a * x[i - 1]) + y[i - 1])

`saxpyP` below is that statement in the modelled Python syntax, with the
completion `else` the emitter adds when `i` is read after the loop (the
modelled emitter always emits it; without it the relation on `i` alone is not
preserved, see FNP-R0123's premise).  `saxpy_sound` is the composition
theorem instantiated: for every `n` and every related input state, the
Fortran and Python executions end in related states.
-/

namespace Recast.Rules

open Recast Rel Exact

def declInt : F.Decl := ⟨.int .i4, [], 0⟩
def declReal : F.Decl := ⟨.real .r8, [], 0⟩
def declArr : F.Decl := ⟨.real .r8, [.intLit .i4 1], 0⟩

/-- The declarations of `saxpy` (`n` is a symbolic size). -/
def Γk : F.Env := fun s =>
  if s = "n" then some declInt else if s = "i" then some declInt else if s = "a" then some declReal
  else if s = "x" then some declArr else if s = "y" then some declArr else none

/-- The rendered declared bounds (`Expressions.bound` on the text `1`). -/
def Λk : Tr.BoundEnv := fun s => if s = "x" then [.intLit 1] else if s = "y" then [.intLit 1] else []

def saxpyF : F.Stmt :=
  .doLoop "i" (.intLit .i4 1) (.var "n") none
    (.assign "y" (.cons (.scalar (.var "i")) .nil)
      (.bin .add (.bin .mul (.var "a") (.ref "x" (.cons (.scalar (.var "i")) .nil)))
        (.ref "y" (.cons (.scalar (.var "i")) .nil))))

def saxpyP : P.Stmt :=
  .forRange "i" (.intLit 1) (.bin .add (.name "n") (.intLit 1)) none
    (.assignIndex "y" (.cons (.scalar (.bin .sub (.name "i") (.intLit 1))) .nil)
      (.paren (.bin .add
        (.paren (.bin .mul (.name "a") (.index "x" (.cons (.scalar (.bin .sub (.name "i") (.intLit 1))) .nil))))
        (.index "y" (.cons (.scalar (.bin .sub (.name "i") (.intLit 1))) .nil)))))
    (some (completion "i" (.intLit 1) (.name "n") (.intLit 1)))

/-- The modelled emitter produces exactly the emitted statement. -/
theorem saxpy_trans : transStmt Γk Λk [] saxpyF = some saxpyP := by rfl

theorem Γk_bounds : BoundEnvOK Γk Λk := by
  intro a d hd
  simp only [Γk] at hd
  split_ifs at hd <;> cases hd <;> simp_all [Λk, declInt, declReal, declArr]

theorem Γk_static : StaticBounds Γk := by
  intro a d hd
  simp only [Γk] at hd
  split_ifs at hd <;> cases hd <;> simp [declInt, declReal, declArr]

theorem saxpy_wfs : WFS Γk saxpyF := by
  have hvi : WFE Γk (.var "i") := ⟨declInt, rfl, rfl⟩
  have hva : WFE Γk (.var "a") := ⟨declReal, rfl, rfl⟩
  have hvn : WFE Γk (.var "n") := ⟨declInt, rfl, rfl⟩
  have hrx : WFE Γk (.ref "x" (.cons (.scalar (.var "i")) .nil)) := ⟨hvi, trivial, declArr, .i4, rfl, rfl⟩
  have hry : WFE Γk (.ref "y" (.cons (.scalar (.var "i")) .nil)) := ⟨hvi, trivial, declArr, .i4, rfl, rfl⟩
  have hmul : WFE Γk (.bin .mul (.var "a") (.ref "x" (.cons (.scalar (.var "i")) .nil))) :=
    ⟨hva, hrx, trivial, trivial⟩
  have he : WFE Γk (.bin .add (.bin .mul (.var "a") (.ref "x" (.cons (.scalar (.var "i")) .nil)))
      (.ref "y" (.cons (.scalar (.var "i")) .nil))) := ⟨hmul, hry, trivial, trivial⟩
  refine ⟨⟨.i4, 0, rfl⟩, trivial, hvn, trivial, trivial, trivial, ⟨by decide, by decide⟩, ?_⟩
  exact ⟨he, hvi, trivial, declArr, .i4, rfl, rfl, .real .r8, rfl, Or.inl ⟨rfl, by decide⟩⟩

/-- The kernel is translated soundly: a forward simulation for every `n`. -/
theorem saxpy_sound (o : Oracles) : StmtSound o Γk saxpyF saxpyP :=
  transStmt_sound o Γk Λk [] Γk_bounds Γk_static saxpyF saxpy_wfs saxpyP saxpy_trans

end Recast.Rules
