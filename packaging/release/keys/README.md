# Production release trust roots

Production installer packages must place the currently trusted Ed25519 public
key(s) in this directory and bind each file to a stable key ID in their consumer
configuration. Key rotation should ship the new public key in an older trusted
installer before a release manifest starts using it.

No production key has been supplied to this development repository yet. The
stable channel therefore fails closed until release engineering provisions that
public trust root. Never copy the test key from `tests/data/installer_release/`
here, and never commit a private key, seed, signing token, or base64 private-key
secret.
