# LSF submission templates

Skeletons for the campaign stages that run on a cluster rather than in the
package: library screening, generative design, and the docking funnel.

These are **templates, not runnable scripts**. Every `<placeholder>` is a
site-specific path, queue name, accounting code or cut-off that has to be
filled in. Resource figures are starting points, not measurements — benchmark
one task before submitting an array.

| file | stage |
|---|---|
| `screen_library.lsf` | score a sharded library with the packaged model |
| `generate_synthemol.lsf` | synthesis-aware generative search, one task per seed |
| `ligprep.lsf` | protonation and tautomer states at pH 7.4 |
| `glide_htvs.lsf` | fast docking pass over all prepared states |
| `glide_sp.lsf` | precise docking over the surviving fraction |

## Conventions used throughout

- **One array task per input chunk**, with an idempotent skip so a partial run
  resumes instead of restarting.
- **Write to `<name>.part`, then `mv`.** A killed task then leaves no file that
  looks complete to the skip check.
- **`rusage[mem=N]` is per core**, not per job, on LSF.
- **Gate each stage on the previous stage's completion marker**, not on the
  existence of its output file — output files appear before they are finished.

## Between HTVS and SP

Ligand preparation expands one compound into several protonation and tautomer
states, and the docking engine treats each as an independent ligand. Collapse
to the best-scoring state per parent compound *before* taking the top fraction,
or compounds that happen to produce more states are over-represented.
