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
The promotion derivative restores all ten original artifact objects, including
their existing public ONNX source candidates, and adds only the eight ncnn
objects. Its SHA-256 is
`62da60bdfb98034b758b3e76d9f65c6b518d89ea6da4dd2d247fa60ce82a67e2`
(`/tmp/studiocast-vulkan-all-model-candidate-20261112/curated-model-catalog-v1.promotion.json`).
The fourteen-file publication inventory is unchanged.

The catalog retains seven packs and grows from ten to eighteen model artifacts
(188,659,386 bytes total). The new artifacts add 22,047,470 bytes. Original ONNX
artifacts are retained for existing backends. Matching license files retain
upstream terms and provenance.

Two independent offline conversions reproduced all eighteen source payloads.
Local-source and network-denied offline install transactions passed. Existing
unmarked destinations were rejected and preserved. Native GPU comparisons passed
unchanged maximum 2e-4 / mean 2e-5 gates; exact RGB FLX tail checks cover thirty
original-ONNX cases. See [completion evidence](VULKAN_EFFECTS_COMPLETION.md).
The full combined production manager matrix remains in progress; publication
approval does not waive it or authorize activation of an invalid runtime.

Publication and catalog promotion require explicit approval under the local
AGENTS.md: “Do not push, publish, create remote pull requests, release artifacts,
or mutate external services unless the user explicitly authorizes it.”
This proposal includes no application release, Git push, or installed user-pack
changes. Prior MODNet publication approval does not authorize this inventory.
