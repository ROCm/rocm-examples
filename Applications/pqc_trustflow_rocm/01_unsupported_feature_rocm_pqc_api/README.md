# Unsupported Function Development: ROCm PQC API

This folder is prepared for the competition item:

```text
(1) Development of currently unsupported functions
```

## What This Adds

The contribution adds a ROCm/HIP post-quantum cryptography backend and an upper-layer file workflow:

- `kem_api/`: Kyber/Aigis-enc batch KEM backend with file-level keygen, encaps, and decaps API paths.
- `sig_api/`: ML-DSA/Aigis-sig batch signature backend with file-level sign and verify API paths.
- `trustflow_frontend/`: a multi-file secure packaging frontend that calls the ROCm KEM/SIG backends.
- `docs/`: quick-start and API notes for reproducing the workflow.

## Key API Examples

```bash
./kyber768_amd --api-kem-keygen --batch 128 --pk-out kem_pk.bin --sk-out receiver_sk.demo_secret
./kyber768_amd --api-kem-encaps --batch 128 --pk-in kem_pk.bin --ct-out kem_ct.bin --ss-out ss_sender.demo_secret
./kyber768_amd --api-kem-decaps --batch 128 --sk-in receiver_sk.demo_secret --ct-in kem_ct.bin --ss-out ss_receiver.demo_secret
```

```bash
./mldsa65_amd --api-sig-sign --batch 128 --msg-in manifest.payload.json --pk-out sig_pk.bin --sk-out sig_sk.demo_secret --sig-out manifest.sig
./mldsa65_amd --api-sig-verify --batch 128 --msg-in manifest.payload.json --pk-in sig_pk.bin --sig-in manifest.sig
```

## Build And Smoke Tests

```bash
cd kem_api
bash build_hip.sh kyber768
bash run_kem_smoke_amd.sh
```

```bash
cd sig_api
bash build_sig_amd.sh
bash run_sig_policy_smoke.sh 128
```

## Why It Fits The Scoring Item

This folder shows a previously unsupported ROCm application path: post-quantum KEM and signature workloads are not only ported to HIP, but also exposed as reusable file-level APIs and connected to a complete TrustFlow packaging workflow.
