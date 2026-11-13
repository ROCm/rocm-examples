# Applications: Post-Quantum Cryptography Example

## Description

This example demonstrates post-quantum key encapsulation and digital signature
workflows implemented with HIP for AMD GPUs. It includes batch implementations
of Kyber and Aigis-enc KEMs and ML-DSA and Aigis-sig signatures.

The executables support two use cases:

- a deterministic, small self-test suitable for CTest; and
- file-oriented commands that let another application perform key generation,
  encapsulation, decapsulation, signing, and verification.

The code is an educational HIP example. It has not undergone the review needed
for use as a production cryptographic library.

## Application Flow

The KEM example performs the following operations on the GPU:

1. Generate a public and secret key pair.
2. Encapsulate a shared secret with the public key.
3. Decapsulate the ciphertext with the secret key.
4. Compare the two shared secrets and reject a mismatch.

The signature example performs the following operations on the GPU:

1. Generate a public and secret key pair.
2. Sign a fixed test message.
3. Verify the signature with the public key.
4. Modify the signature and confirm that verification fails.

## Key APIs and Concepts

- `hipMalloc` and `hipFree` manage device buffers.
- `hipMemcpy` transfers keys, ciphertexts, messages, and signatures.
- HIP kernel launches execute batch sampling, NTT, polynomial arithmetic,
  packing, and verification stages.
- `hipGetLastError` and `hipDeviceSynchronize` report launch and execution
  failures.
- Structure-of-arrays layouts allow independent cryptographic records to be
  processed in parallel.

The implementation separates sampling, polynomial transforms, matrix-vector
operations, and packing into focused HIP kernels. NTT stages use shared memory,
and device buffers are reused across each batch. Signature key generation uses
separate seed expansion, matrix sampling, and secret sampling kernels.

## Supported Parameter Sets

The default build includes one representative parameter set from each family:

| Target | Algorithm |
| --- | --- |
| `applications_pqc_kyber768` | Kyber-768 |
| `applications_pqc_aigis_enc3` | Aigis-enc-3 |
| `applications_pqc_mldsa65` | ML-DSA-65 |
| `applications_pqc_aigis_sig2` | Aigis-sig2 |

Set `ROCM_EXAMPLES_PQC_BUILD_ALL_VARIANTS=ON` to also build Kyber-512,
Kyber-1024, Aigis-enc-1/2/4, ML-DSA-44/87, and Aigis-sig1/3.

## Building

From the repository root, build all default examples with CMake:

```bash
cmake -S . -B build
cmake --build build --target \
  applications_pqc_kyber768 \
  applications_pqc_aigis_enc3 \
  applications_pqc_mldsa65 \
  applications_pqc_aigis_sig2
```

To build every supported parameter set:

```bash
cmake -S . -B build -DROCM_EXAMPLES_PQC_BUILD_ALL_VARIANTS=ON
cmake --build build
```

The example can also be built independently:

```bash
cd Applications/pqc_trustflow_rocm
cmake -S . -B build
cmake --build build
```

GNU Make builds the same four default targets:

```bash
cd Applications/pqc_trustflow_rocm
make
```

Use `make all-variants` to build every parameter set.

## Testing

Run the PQC CTest self-tests from the repository root:

```bash
ctest --test-dir build --output-on-failure -R '^applications_pqc_'
```

For an individual executable built by the root CMake project:

```bash
./build/bin/Applications/applications_pqc_kyber768 --self-test
./build/bin/Applications/applications_pqc_mldsa65 --self-test
```

The Makefile provides the equivalent `make test` target.

## File-Oriented API

The commands below assume the example was built with Make from
`Applications/pqc_trustflow_rocm`, so the executables are in the current
directory.

The following example performs a KEM round trip:

```bash
./applications_pqc_kyber768 --api-kem-keygen --batch 8 \
  --pk-out kem_pk.bin --sk-out kem_sk.bin
./applications_pqc_kyber768 --api-kem-encaps --batch 8 \
  --pk-in kem_pk.bin --ct-out kem_ct.bin --ss-out sender_ss.bin
./applications_pqc_kyber768 --api-kem-decaps --batch 8 \
  --sk-in kem_sk.bin --ct-in kem_ct.bin --ss-out receiver_ss.bin
cmp sender_ss.bin receiver_ss.bin
```

The following example signs and verifies a file:

```bash
./applications_pqc_mldsa65 --api-sig-sign --batch 8 \
  --msg-in message.bin --pk-out sig_pk.bin --sk-out sig_sk.bin \
  --sig-out message.sig
./applications_pqc_mldsa65 --api-sig-verify --batch 8 \
  --msg-in message.bin --pk-in sig_pk.bin --sig-in message.sig
```

See [API notes](docs/api_notes.md) for command behavior and file contracts.
