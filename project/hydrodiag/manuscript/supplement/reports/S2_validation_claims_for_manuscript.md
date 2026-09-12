# S2 Validation Claims For Manuscript

## Fully supported

The active foundation-531 registries select the full XAJ, GR4J, and SIMHYD classes for Base, CN, and TGD, and select the full HBV class for the snow-process reference.

At interior parameter points, the active full-model forward paths support backward propagation with finite gradients for the parameter rows marked PASS in the float64 differentiability table.

The validation keeps finite unit-hydrograph buffer water as model state and separates finite-window output from tail-aware system accounting.

## Supported with qualification

CN and TGD are validated as preprocessing modules coupled to the active host models through serial-wrapper and fused-wrapper equivalence checks.

The gradient evidence uses float64 central finite differences and directional derivatives at interior points; it does not establish smoothness at piecewise threshold or clipping boundaries.

The historical GR4J float32 low-signal finite-difference rows are treated as numerical-resolution limitations when the corresponding float64 and directional checks pass.

Mass-balance claims apply only to the layers and rows explicitly marked PASS in the closure tables; any unresolved diagnostic row must remain qualified.

## Prohibited

Do not describe successful model training as a strict gradient check.

Do not describe a finite-window unit-hydrograph tail as mass lost from the model.

Do not claim that the validation covers GD or PD as paper structures.

Do not claim exact canonical equivalence to a literature implementation from these wrapper and numerical checks alone.
