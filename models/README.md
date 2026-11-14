# Models

The four models from `context.md` §5. Each has its own folder. The baseline
(Hahner et al. PV-RCNN †, α=0) is **not** here: it lives in `../baseline/` and is
reported as a fifth, separate system (`context.md` §10).

| Folder | Model | Status |
| --- | --- | --- |
| `model1/` | Reliability-gated multimodal transformer (lidar + left RGB) | Code, data pipeline and tests done (M0-M2). Not yet trained: the 200-frame trial (M3) is in progress. Design in `model1/docs/DESIGN.md`, runs in `model1/docs/RUN_LOG.md`. |
| `model2/` | Product-of-experts probabilistic fusion | Not started |
| `model3/` | Physics-informed degradation forecaster | Not started |
| `model4/` | Self-supervised test-time adaptation | Not started |

Rules that apply to every model here (from `context.md` §6 and §10):

- Train on STF clear weather only. Weather corruption is applied at evaluation time only.
- No corruption labels at inference.
- Evaluate under the same conditions as the baseline: same splits
  (3469/781/1847/946/786), same ground truth, same Car 3D AP definition (R40, IoU 0.5).
- No number from a model is reported as a comparison until the baseline gate
  (77.05 ± 2, clear/moderate) passes.
- If a change helps AP but not calibration, say so explicitly.
