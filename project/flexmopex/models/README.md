# Flex-MOPEX Models

The model layer is split into three independent physical-model classes and two
parameter-network classes.

## Physical Models

| Class | Structure weights | Core step |
| --- | --- | --- |
| `StaticMopex` | No structure weights; all processes are active | `mopex_step_static` |
| `FixedWeightMopex` | Four fixed config buffers: `w_phen`, `w_int`, `w_snow`, `w_sub` | `mopex_step` |
| `LearnedWeightMopex` | Four learned Off/On logits converted to On probabilities | `mopex_step` |

## Parameter Networks

| Class | Outputs | Intended models |
| --- | --- | --- |
| `ParamRoutingNet` | `params`, `gamma_uh` | `StaticMopex`, `FixedWeightMopex` |
| `LearnedStructureNet` | `params`, `weights`, `gamma_uh` | `LearnedWeightMopex` |

The default config uses `LearnedWeightMopex` with `LearnedStructureNet`.
