# S2 Parameter Source-of-Truth Report

## Active parameter path

The 531 foundation configuration is `ablation/configs/ic_foundation_531_v1.json:2-12`. IC-XNES maps the active model key through `ablation/ic_core/model_adapter.py:14-28` and uses full classes by default at `ablation/ic_core/model_adapter.py:31-45`; the runtime calls the shared adapter at `ablation/ic_core/runtime.py:104-110`. dPL uses the same full registry classes at `training/dpl/run_dpl_model.py:84-98`; the lite registry is selected only by explicit `--lite` at `training/dpl/run_dpl_model.py:613-629`. The parameter order is `list(parameter_specs)` and is recorded in `results/s2_parameter_bounds_from_code.csv`.

Active base counts are XAJ=15, GR4J=4, SIMHYD=10 and HBV=12. The ten active SIMHYD names are `simhyd_insc`, `simhyd_coeff`, `simhyd_sq`, `simhyd_smsc`, `simhyd_sub`, `simhyd_crak`, `simhyd_k`, `simhyd_etmul`, `simhyd_a` and `simhyd_theta`; this conflicts with a nine-parameter expectation. CN adds CTG and Kf; TGD adds alpha, tau and beta. Gamma UH shape/scale parameters are included in XAJ and SIMHYD host vectors, not added as a separate model call. HBV is standalone and has no MAXBAS.

## Code bounds and mappings

All bounds/defaults are extracted from `models/parameter_specs.py` and are project-specific calibration bounds. IC uses `normalized_to_physical` (`ablation/ic_core/parameter_adapter.py:56-80`); normalized values are clipped to [0,1], then mapped linearly except `tgd_tau`, which is log-interpolated. dPL applies sigmoid and output clipping at `training/dpl/run_dpl_model.py:166-168`; its head bias is initialized from parameter defaults at lines 148-164.

Runtime effective values are not identical to calibration bounds. In particular, hydrodl2 `uh_gamma` applies `relu(a)+0.1`, `relu(theta)+0.5`, samples at t=0.5, and normalizes each kernel (`/home/jingxin/code/dmg-research/.venv/lib/python3.10/site-packages/hydrodl2/core/calc/uh_routing.py:5-22`, version 1.3.4, SHA256 `a2305c37ca895efe3323e8fabe20421b84e511355d0aab67eb4d3b657b67c0a`). XAJ additionally jointly rescales KI/KG when their sum is at least one (`models/xaj.py:280-294`).

## Meaning and unit evidence

`results/s2_parameter_meaning_and_units.csv` separates CODE_AND_LITERATURE_AGREE from INFERRED_FROM_EQUATION and IMPLEMENTATION_SPECIFIC. The main classical meanings are supported by Zhao (1992) for XAJ, Perrin et al. (2003) for GR4J, Bergström (1992) for HBV, and Valéry et al. (2014) for CemaNeige. SIMHYD is explicitly called the differentiable SIMHYD implementation used in this study because the accessible authoritative source does not prove the exact current ten-parameter variant. TGD meanings are equation-derived and have no claimed classical provenance.

## Required manuscript edits

1. State that active XAJ has 15 parameters, including `xaj_a` and `xaj_theta`; do not use a 14-parameter description.
2. State all lower/upper bounds as project-specific code bounds, not literature ranges.
3. Explain linear mapping for all active parameters except log interpolation for `tgd_tau`, and sigmoid-to-bound dPL output.
4. Footnote that Gamma UH effective shape/scale are `relu(a)+0.1` and `relu(theta)+0.5`, with finite daily kernels.
5. Keep `parCWH` unit conflict visible: the spec says `1/day`, while the equation uses it as a multiplier of snowpack storage; recommended wording is dimensionless holding coefficient with an audit note.
6. Describe KI/KG as implemented free-water outflow coefficients and CI/CG as output-state memory/recession coefficients; do not interchange them.
7. Describe SIMHYD as the differentiable implementation used in this study; do not assert a unique canonical SIMHYD variant.
8. Describe TGD as project-specific and equation-defined, not as a literature model.
9. The requested S2 draft file was not found under `manuscript/`; no direct manuscript edits were made.
