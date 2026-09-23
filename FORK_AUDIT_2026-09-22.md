# Fork audit and fixes — 2026-09-22

## Scope and provenance

Reviewed the 31-commit fork stack originally ending at
`7585e2c4d477c8cc20d1f73df5ddd59bc5bd1dc8`, against its upstream base
`50631b3d2c569ad8e5c112090cd28570b1268ee0`.
The pre-fix history is retained at `audit/pre-fixes-2026-09-22`.

The upstream master object inspected during this audit was
`ec5a12b85ae32fbccfa4276051382330a8e6458b`, 53 commits newer than the fork base.
This audit does not merge those upstream commits. Old Markdown benchmark claims
were not treated as evidence that the current rebased implementation is correct.

Review covered model loading/sharing, CPU scheduling, Metal kernel changes and
dispatch, QSA/pooled-cache construction and invalidation, expert streaming and
phase transitions, MTP/rejection/n-gram integration, prompt persistence, exposed
APIs, metrics, and associated build/test changes. Source review is not a proof
that all bugs or performance regressions have been eliminated.

## Findings fixed

| Finding | Consequence | Fix and owning commit (original hash) |
| --- | --- | --- |
| QSA block selection uses one spare row for all incomplete blocks | Holes or multiple sequences can overwrite live members and omit attention keys | Require a contiguous, single-sequence text prefix for block top-k; otherwise use per-cell selection. Check eligibility again on graph reuse. `c930d33a5` |
| Eagle3 did not implement prompt-cache boundary serialization | Restored draft KV could be treated as synchronized despite a missing deferred hidden row, including non-recurrent targets | Serialize and validate Eagle3 boundary state independently of recurrent rollback checkpoints. `1f8a22fc7` |
| Metal FA mask-block scan did not check the query edge | A partial query tile could read beyond an unpadded mask | Adopt the query-bound check from upstream `fb34fc262c1b43f1832c7472429fb2247d650493` (#29220); add non-padded 17/33/65-query cases. This is an inherited upstream bug, not solely a fork regression. `c0203cad0` |
| SSD prompt cache retained all checkpoint payloads in RAM | Inactive conversations retained potentially large recurrent snapshots; saving also copied those snapshots | Stream checkpoints from the active prompt, retain no checkpoint payloads in the disk index, and load only the selected conversation's sidecar. `1f8a22fc7` |
| Expert-cache resize allocated `slot_spec` after publishing changed tensors | Allocation failure could leave the current layer changed but absent from the rollback list | Allocate all slot-spec metadata before publication and move it into place. `9486f81d2` |
| Cache-size parsing accepted negative/overflowing sizes and retained stale units | Wrapped budgets or a previous slot count could override the requested GiB budget | Validate numeric bounds/signs, clear the alternate unit, and reject negative llama-bench budgets. `9486f81d2`, `ea7fc24b4` |
| Backend buffer ABI grew without incrementing its API version | Old dynamically loaded backends could pass the compatibility check with an incompatible buffer layout | Increment backend API version to 3; external backend plugins must be rebuilt. `a2c77d031` |
| Lookahead callback objects were allocated without an owner | Each model unload leaked the callback object and its scratch vectors | Give the layer unique ownership while graph callbacks retain a non-owning pointer. `73808e473` |

The persisted-state semantic version is increased from 3 to 4 in `1f8a22fc7`.
Previously computed cache entries are not reused under the corrected QSA behavior.
Their old files are not deleted by this audit.

## Verification

Built `llama-server`, `llama-bench`, `test-backend-ops`, and the targeted CPU tests.
The CPU suite covers:

- `test-speculative-state`
- `test-speculative-hybrid`
- `test-moe-wave`
- `test-server-state-key`
- `test-fork-attention`
- `test-moe-cache-resize`
- `test-moe-stream-priority`
- `test-memory-phase`
- `test-arg-parser`
- `test-server-prompt-cache`

All 10 passed. New assertions cover QSA fallback eligibility, lazy checkpoint
round-trips without resident disk-index checkpoint copies, cache-size bounds and
unit replacement, and slot-spec metadata after resizing. Existing allocation
failure tests exercise backend-buffer failures, not injected failures of every
individual C++ metadata allocation.

No GPU operations, live-server requests, server stop, or restart were performed
after the user's GPU restriction. The new Metal FA cases are compiled but not
GPU-validated in this pass. Eagle3's boundary format validation is covered by the
state tests, but a live Eagle3 model restore has not been exercised.

## Performance conclusions and remaining checks

The checkpoint change removes an identifiable source of unnecessary RAM use:
the sum of inactive cached conversations' checkpoint payloads, plus the temporary
copy previously made during a disk save. The actual saving depends on the saved
conversations; no fixed GiB saving or decode speedup was measured. Freed memory
does not by itself guarantee that the current automatic expert-budget policy
will allocate more expert slots.

No new Q3_K/IQ3_S/IQ3_XXS speedup is claimed by this audit. Current kernel dispatch
and quantized paths were reviewed, not benchmarked while the server owned the
GPU. The most relevant remaining verification is a fixed-context, fixed-width
ms/step run after the user releases the GPU, plus a real-model long-context QSA
rollback/restore comparison and the targeted Metal partial-mask cases. Recheck
memory usage with several cached conversations separately from kernel timings.

The user's deleted `AGENTS.md` and pre-existing untracked audit reports are not
part of these fixes. History changes are local; nothing is pushed.
