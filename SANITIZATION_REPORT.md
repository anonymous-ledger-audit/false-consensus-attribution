# Release sanitization report

This anonymous release was prepared from a 16-script experiment archive. The
scientific source files were copied before editing; the original archive was
not modified.

## Scope

- Exactly 16 reported experiment and figure scripts are included.
- No exploratory or unreported model runner is included.
- No manuscript source, author block, local execution log, IDE file, dataset
  cache, or trained checkpoint is included.
- The compact artifact release contains 94 paper-relevant files under
  `artifacts/`; unrelated legacy figure suites and parsed pilot tables are not
  included.

## Documentation cleanup

Working-session wording was replaced with neutral technical documentation.
Local directory references in run instructions were replaced with
repository-relative instructions. One stale reference in a failure message was
generalized to the fitted-model stages.

## Verification

- Personal names, email addresses, account handles, institution names, and
  absolute user paths: not present in the released experiment sources.
- Conversational assistant markers: not present in the released experiment
  sources.
- Python syntax: all 16 scripts compile.
- Executable structure: identifiers, operators, numeric constants, function
  and class definitions, imports, and control flow match the raw scripts.
  Differences are restricted to comments, documentation strings, and one
  non-functional error-message string.

## Artifact cleanup

Every distributed numerical CSV, NumPy array, image, and PDF retains the frozen
run bytes. Two metadata-only JSON path fields were converted from an absolute
execution directory to repository-relative paths. Stage manifests, hard-pass
flags, audit row counts, plotted-values JSON files, and distributed paper figures
were independently verified after this conversion. Regenerable LaTeX helper files
are intentionally not distributed.

The read-only tabular and vision paper generators reproduce the distributed
plotted values exactly without model fitting or attribution calls.

Run `python tools/verify_release.py` and `python tools/verify_artifacts.py` to
repeat the source and artifact checks before publication.
