import RecastProofs

-- The exact historical declarations; related-only checks do not prove this defect.
#check @Recast.Rules.FNP_R0102_obligation
#check @Recast.Rules.FNP_R0102_refuted_shrink
#check @Recast.Rules.FNP_R0102_exact_obligation
#check @Recast.Rules.FNP_R0102_sound_exact
#print axioms Recast.Rules.FNP_R0102_refuted_shrink
#print axioms Recast.Rules.FNP_R0102_sound_exact
