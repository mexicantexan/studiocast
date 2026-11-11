# Production release trust roots

This directory owns the production Ed25519 public keys used to build installer
packages. Packaging stages explicitly selected keys under
`share/studiocast/installer/trust/keys/` and binds each file to a stable key ID
in consumer configuration. Key rotation should ship the new public key in an
older trusted installer before a release manifest starts using it.

The initial production trust root is
`studiocast-release-2026.pem`, bound to key ID
`studiocast-release-2026`. This directory contains public trust roots only.
Never copy the test key from `tests/data/installer_release/` here, and never
commit a private key, seed, signing token, or base64 private-key secret.

Pass each externally provisioned public key to the AppImage builder as
`--trusted-release-key <key-id>=<public-key.pem>`. Release-grade packaging
refuses to proceed without a validated Ed25519 public key and writes the key to
this installed directory. The release manifest and artifact signatures must use
the matching key ID.
