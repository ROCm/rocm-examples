# PQC File API Notes

The KEM and signature executables expose file-oriented commands in addition to
their built-in self-tests. This interface is intended to demonstrate how a host
application can compose HIP cryptographic operations without linking the
example as a library.

## Common Behavior

- Exactly one API operation may be selected per invocation.
- `--batch N` must be a positive integer. Signature commands also enforce a
  parameter-set-specific upper bound so their 16-bit sampling nonces cannot
  wrap. The operation processes a batch and writes the first resulting record
  to each output file.
- Key, ciphertext, and signature inputs must have the exact size required by
  the parameter set. Message files may have any size.
- A successful command returns zero. Invalid arguments, file errors, HIP
  errors, or cryptographic verification failures return a nonzero value.
- Secret-key and shared-secret files contain sensitive material. The example
  does not provide secure storage or deletion.

## KEM Commands

### Key Generation

```text
--api-kem-keygen --batch N --pk-out FILE --sk-out FILE
```

Generates a batch of key pairs and writes one public key and one secret key.

### Encapsulation

```text
--api-kem-encaps --batch N --pk-in FILE --ct-out FILE --ss-out FILE
```

Reads one public key, duplicates it across the batch, and writes one KEM
ciphertext and the corresponding sender shared secret.

### Decapsulation

```text
--api-kem-decaps --batch N --sk-in FILE --ct-in FILE --ss-out FILE
```

Reads one secret key and one ciphertext, duplicates them across the batch, and
writes one receiver shared secret. A caller can compare this file with the
sender shared secret produced by encapsulation.

## Signature Commands

### Signing

```text
--api-sig-sign --batch N --msg-in FILE --pk-out FILE --sk-out FILE
               --sig-out FILE
```

Generates a key pair, signs the complete message file, and writes one public
key, one secret key, and one signature.

### Verification

```text
--api-sig-verify --batch N --msg-in FILE --pk-in FILE --sig-in FILE
```

Reads the message, public key, and signature, verifies them across the batch,
and returns nonzero if any verification fails.

## Integrating a Transfer Workflow

A host application can combine these commands with an authenticated symmetric
cipher such as AES-256-GCM:

1. The receiver generates a KEM key pair and transfers the public key.
2. The sender encapsulates a shared secret and transfers the KEM ciphertext.
3. Both sides derive a symmetric key from their matching shared secrets with an
   appropriate key-derivation function.
4. The sender encrypts the payload and signs a canonical manifest containing
   the ciphertext metadata.
5. The receiver decapsulates, verifies the manifest signature, and decrypts the
   payload only after authentication succeeds.

Key derivation, authenticated encryption, transport, identity binding, replay
protection, and secret storage remain responsibilities of the integrating
application and are outside this HIP example.
