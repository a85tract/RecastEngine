import RecastProofs

-- The exact historical declarations; related-only checks do not prove this defect.
#check @Recast.Rules.FNP_R0104_obligation
#check @Recast.Rules.FNP_R0104_refuted_rank2
#check @Recast.Rules.FNP_R0104_rank1_obligation
#check @Recast.Rules.FNP_R0104_sound_rank1
#print axioms Recast.Rules.FNP_R0104_refuted_rank2
#print axioms Recast.Rules.FNP_R0104_sound_rank1
