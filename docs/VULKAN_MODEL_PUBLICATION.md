# Vulkan multi-model publication proposal

Status: verified candidate; explicit publication approval pending.

Publish fourteen additive files (22,066,504 bytes) to the existing
`10dallasj/studiocast` Hugging Face model repository: eight FP32 ncnn graph/weight
artifacts and matching manifests/licenses for FastDVDnet sigma15, YuNet FP32,
and FLX v0.1.1 FP32. Keep their existing model IDs, original ONNX bytes,
defaults and every existing remote path unchanged. New files use immutable
content-named revision directories. Promote the matching local catalog only
after public retrieval verifies the published bytes.

The exact source/destination/size/SHA-256 inventory is
`/tmp/studiocast-vulkan-all-model-candidate-20261112/publication-proposal.json`.
Inventory SHA-256: `8411bb3f45ae31111f515724859128c13d7b59abf4b3058cef139f01fe8a0406`.

Candidate catalog SHA-256:
`46ac5d7c10544c7eda8124c79a1e97035466b335ba35a32d320b367fda637a67`.
The offline reproduction catalog uses local source paths for its test source.
The historical promotion derivative restores all ten original artifact objects,
including their existing public ONNX source candidates, and adds eight ncnn
objects. Its SHA-256 is
`62da60bdfb98034b758b3e76d9f65c6b518d89ea6da4dd2d247fa60ce82a67e2`
(`/tmp/studiocast-vulkan-all-model-candidate-20261112/curated-model-catalog-v1.promotion.json`).
It must not be activated: its eight new source paths were still unversioned,
while the publication inventory targets immutable revision directories.

The corrected, unapproved repository proposal is commit
`01eaaa4ea7546fed6911663d2a1754b0cbaa7177` on
`codex/vulkan-catalog-unapproved-proposal-20261113`. Its catalog SHA-256 is
`097cb1f2551263c52c52357839bd4350c090c93c5619193688432135a9c42d9e`.
All eight new sources now match the exact inventory destinations. The ten old
artifact objects and four unaffected pack objects are preserved; the six changed
metadata/license templates exactly match the verified inventory source bytes.
The fourteen-file publication inventory is unchanged. This proposal is not
merged or active.

The catalog retains seven packs and grows from ten to eighteen model artifacts
(188,659,386 bytes total). The new artifacts add 22,047,470 bytes. Original ONNX
artifacts are retained for existing backends. Matching license files retain
upstream terms and provenance.

Two independent offline conversions reproduced all eighteen source payloads.
Local-source and network-denied offline install transactions passed. Existing
unmarked destinations were rejected and preserved. Native GPU comparisons passed
unchanged maximum 2e-4 / mean 2e-5 gates; exact RGB FLX tail checks cover thirty
original-ONNX cases. See [completion evidence](VULKAN_EFFECTS_COMPLETION.md).
The available native matrix records passing component/composition cases and
explicit failed/unqualified combined deadline profiles on the P1000. Publication
approval does not waive those limits or authorize activation of an invalid runtime.

The isolated catalog proposal passes 21 model-transaction, 56 installer-backend,
13 packaging and 14 offscreen GUI scenarios. After publication approval, verify
every file by public retrieval at the new immutable Hugging Face commit, preserve
all existing remote paths, record the factual receipt, set the actual catalog
version/provenance/source pin, refresh exact hash fixtures, and independently
review before root integration. No installed pack mutation is included. Local
handoff: `/tmp/studiocast-vulkan-catalog-unapproved-proposal-handoff-20261113.md`.

Publication and catalog promotion require explicit approval under the local
AGENTS.md: “Do not push, publish, create remote pull requests, release artifacts,
or mutate external services unless the user explicitly authorizes it.”
This proposal includes no application release, Git push, or installed user-pack
changes. Prior MODNet publication approval does not authorize this inventory.
