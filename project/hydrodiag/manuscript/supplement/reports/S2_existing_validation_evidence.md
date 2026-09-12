# S2 Existing Validation Evidence

## Inventory

s2_existing_validation_inventory.csv records path, object, model/structure inference, full/lite/legacy status, inputs, dtype, tolerances, command, original result, evidence level, active-code status, reproducibility, and adaptation need.

The inventory contains 469 rows. Evidence levels are preserved as EXACT_ACTIVE_CODE, SAME_KERNEL_DIFFERENT_WRAPPER, LEGACY_BUT_RELEVANT, NOT_APPLICABLE, or BROKEN_OR_MISSING.

## Coverage

s2_existing_validation_coverage.csv is the model-by-structure coverage matrix for XAJ, GR4J, SIMHYD, and HBV under Base, TGD, CN, and the HBV reference role. It is intentionally separate from the new closure outputs.

## Directly reusable evidence

The active project tests cover full forward execution, backward smoke behavior, GR4J UH differentiability, TGD serial composition, CN fused composition, SIMHYD full-system water balance, and routing continuation. Existing raw-step and reference utilities are reused as same-kernel evidence.

## Evidence that is not promoted

The archived water-balance script estimates storage from final states and omits routing-tail and some daily diagnostic terms; it is legacy/relevant, not an exact whole-system proof. The old gradient audit uses a separate float32 representative finite-difference path for active compositions. Its three GR4J failures are retained and reclassified in the closure failure table.

## Reproduction

run_existing_model_validations.sh preserves the original pytest command in its log and runs without changing test logic. The new closure scripts are independent supplement diagnostics and are not presented as historical results.
