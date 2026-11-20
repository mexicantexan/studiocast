# Vulkan multi-model publication record

Status: user-authorized publication verified; catalog activated locally.

The user approved the fourteen-file inventory and matching catalog activation.
One additive [Hugging Face commit 47134731](https://huggingface.co/10dallasj/studiocast/commit/4713473184bf7ece0be6c097ff75f6c6c528c8ec),
parent `f6be359c49ba07163287fe3732dc4a3024b91f60`, published exactly fourteen
files (22,066,504 bytes) to `10dallasj/studiocast`. Actual publication time was
`2026-11-14T01:41:23.325423+00:00` (November 13 locally). All 62 existing
remote paths, blob identities and byte sizes were preserved; the resulting tree
has 76 files.

| Pack | New files including metadata/license | Published bytes |
| --- | ---: | ---: |
| FastDVDnet sigma15 | 4 | 19,827,613 |
| YuNet FP32 | 4 | 225,670 |
| FLX v0.1.1 FP32 | 6 | 2,013,221 |

The exact approved source/destination/size/SHA-256 inventory remains
`/tmp/studiocast-vulkan-all-model-candidate-20261112/publication-proposal.json`,
SHA-256 `8411bb3f45ae31111f515724859128c13d7b59abf4b3058cef139f01fe8a0406`.
The publisher verified that inventory and every source byte before mutation,
used the actual remote parent as a concurrency guard, and refused overwrites.

All fourteen new files and all ten original catalog artifacts were retrieved
without authentication at the resulting immutable commit and independently
checked against their measured sizes and pinned SHA-256 values. Original
artifact retrieval used the real preserved source candidates, including
`matting/Good Quality/model.onnx` for MODNet. The before/after tree comparison
preserved all existing files, not only the catalog artifacts.

The factual receipt is
`/tmp/studiocast-vulkan-model-publication-20261113/publication-receipt.json`,
SHA-256 `89b171deb9a5218f0b3da2947f09ed972b4c14fd0bdec532150aa2b646fd3e95`.
It records the actual parent/commit, complete tree comparison, exact inventory
and all 24 unauthenticated immutable retrieval URLs and measured identities.
New public payload copies are retained beneath that receipt directory.

## Active catalog

Catalog version `2026-11-14`, SHA-256
`2e5a02a067765fbc54aa1d798d44b25f9d940629e83509b153274ef7d7cdca1b`,
pins its source base to
`https://huggingface.co/10dallasj/studiocast/resolve/4713473184bf7ece0be6c097ff75f6c6c528c8ec`.
Seven default packs now contain eighteen artifacts / 188,659,386 artifact bytes.
Eight new ncnn artifacts add 22,047,470 bytes; the remaining published bytes are
matching metadata and licenses. All ten original artifact objects, ONNX sources,
IDs/defaults and four unaffected pack objects are preserved. The three affected
packs retain their upstream provenance and record the actual ncnn publication.
Six packaged manifests/licenses are byte-identical to their published files.

The prepared proposal `01eaaa4` was integrated as `15a873b`, followed by factual
activation `9b6dec2` integrated as `ffc50d6`. Exact artifact-count, size, hash,
source-binding, GUI and package-verifier assertions were updated together.
Focused activation checks pass: 21 model-transaction tests, 56 installer-backend
tests, 13 packaging tests and 14 offscreen GUI scenarios. Integrated broad checks
and fresh critique are recorded in [the completion ledger](VULKAN_EFFECTS_COMPLETION.md).

## Historical candidates and limits

The offline reproduction catalog SHA-256
`46ac5d7c10544c7eda8124c79a1e97035466b335ba35a32d320b367fda637a67`
uses local paths and is never shipped. Historical derivative SHA-256
`62da60bdfb98034b758b3e76d9f65c6b518d89ea6da4dd2d247fa60ce82a67e2`
restored original ONNX sources but still had eight unversioned new paths; it
must not be activated. The corrected proposal SHA-256
`097cb1f2551263c52c52357839bd4350c090c93c5619193688432135a9c42d9e`
bound those eight sources to the exact inventory revision destinations before
approval. All historical candidate inputs remain unchanged.

Publication does not waive runtime, integrity, ownership, residency or deadline
gates. The documented P1000 combined-effect timeouts and other unavailable
hardware qualification remain explicit deployment-feedback limits. See
[VULKAN_EFFECTS_COMPLETION.md](VULKAN_EFFECTS_COMPLETION.md).

This action included model publication and local catalog activation only.
Installed user packs were untouched. Future installs/updates retain the normal
ownership-safe transaction rules; unmarked/custom packs are preserved. Git push,
application release and the signed distribution chain remain separate actions.
