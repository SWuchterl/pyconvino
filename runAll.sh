

declare -a setupsSingle=("ATLAS13Only" "ATLAS8Only" "CMSOnly")

declare -a pairs=("Combination_ATLAS13CMS13_corr" "Combination_ATLAS13CMS13_noCorr" "Combination_ATLAS813_corr" "Combination_ATLAS813_noCorr" "Combination_ATLAS8CMS13_corr" "Combination_ATLAS8CMS13_noCorr")

declare -a triples=("Combination_ATLAS813CMS13_corr_extrabJES" "Combination_ATLAS813CMS13_corrV2" "Combination_ATLAS813CMS13_noCorr")

declare -a triplesTest=("Combination_ATLAS813CMS13_corrDeltaPhi" "Combination_ATLAS813CMS13_corrExtreme" "Combination_ATLAS813CMS13_corrExtremeInv" "Combination_ATLAS813CMS13_corrMore" "Combination_ATLAS813CMS13_corrV2_noLineshape" "Combination_ATLAS813CMS13_corrV2_noRecoil" "Combination_ATLAS813CMS13_corrV2_statonly" "Combination_ATLAS813CMS13_corrV2_extra" "Combination_ATLAS813CMS13_corr_extrabJES_optA" "Combination_ATLAS813CMS13_corr_extrabJES_optB" "Combination_ATLAS813CMS13_corr_extrabJES_optC" "Combination_ATLAS813CMS13_corr_extrabJES_optR2")
# declare -a triplesTest=("Combination_ATLAS813CMS13_corrV2_extra")

run_combos() {
    for combsetup in "$@"; do
        echo "Running setup: ${combsetup}"
        outname=${combsetup/Combination_/}
        echo "Output name: ${outname}"
        convino ConvinoSetups/${combsetup}/rho_config.txt --prefix out/${outname} --export both --debug --verbose &> ${combsetup}.log
    done
}

# cross-check: CMS input with CMS_norm_tt0jet free (HEPData v2)
declare -a ttfreeSetups=("CMSOnly_ttfree" "Combination_ATLAS13CMS13_corr_ttfree" "Combination_ATLAS8CMS13_corr_ttfree" "Combination_ATLAS813CMS13_corr_extrabJES_ttfree")

# cross-check: post-fit nuisance values (from the free CMS input for CMS setups)
declare -a pullsSetups=("ATLAS13Only" "ATLAS8Only" "CMSOnly_ttfree" "Combination_ATLAS813_corr" "Combination_ATLAS13CMS13_corr_ttfree" "Combination_ATLAS8CMS13_corr_ttfree" "Combination_ATLAS813CMS13_corr_extrabJES_ttfree")

run_pulls() {
    for combsetup in "$@"; do
        outname=${combsetup/Combination_/}_pulls
        echo "Running setup with nuisance values: ${combsetup} -> ${outname}"
        convino ConvinoSetups/${combsetup}/rho_config.txt --use-nuisance-values --prefix out/${outname} --export both --debug --verbose &> ${combsetup}_pulls.log
    done
}

run_combos "${setupsSingle[@]}"
run_combos "${pairs[@]}"
run_combos "${triples[@]}"
run_combos "${triplesTest[@]}"
run_combos "${ttfreeSetups[@]}"
run_pulls "${pullsSetups[@]}"
