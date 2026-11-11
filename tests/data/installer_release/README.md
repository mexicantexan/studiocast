# Installer release test fixtures

`fixture-ed25519-public.pem` is a **test-only, non-production** public key.
Its private key was used once to create these fixtures and is not committed.
It must never be trusted by a production installer or used to sign a release.

The tiny `.tar.gz` and `.AppImage` files are plain text on purpose: tests verify
the signed byte contract without executing or unpacking either artifact.
