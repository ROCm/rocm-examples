from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import shutil
import subprocess
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
except Exception:  # pragma: no cover - depends on the Jupyter image
    AESGCM = None


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
OUTPUTS_DIR = BASE_DIR / "outputs"
LOGS_DIR = BASE_DIR / "logs"
SAMPLES_DIR = BASE_DIR / "sample_docs"
PACKS_DIR = OUTPUTS_DIR / "packs"
UNPACKS_DIR = OUTPUTS_DIR / "unpacked"


def _digest(label: str, text: str, extra: str = "") -> str:
    payload = f"{label}|{text}|{extra}".encode("utf-8", errors="ignore")
    return hashlib.sha256(payload).hexdigest()


@dataclass
class FlowArtifacts:
    kem_shared_key: str
    kem_ciphertext: str
    sym_ciphertext: str
    signature: str
    verified: bool
    decrypted_text: str


@dataclass
class RealFlowResult:
    pack_dir: str
    pack_zip: str
    unpack_dir: str
    manifest_path: str
    plaintext_dir: str
    verified: bool
    tamper_detected: bool
    kem_shared_key: str
    kem_ciphertext: str
    signature: str
    file_count: int
    total_bytes: int
    timings_ms: dict[str, float]
    rocm_logs: dict[str, str]
    notes: list[str]


def run_mock_trustflow(text: str, kem_choice: str, sig_choice: str, batch_size: int, mode: str) -> FlowArtifacts:
    kem_key = _digest("kem-key", text, f"{kem_choice}|{batch_size}|{mode}")[:64]
    kem_ct = _digest("kem-ct", text, kem_key)[:96]
    sym_ct = _digest("sym-ct", text, kem_ct)[:96]
    sig = _digest("sig", text, f"{sig_choice}|{batch_size}|{mode}")[:96]
    verified = bool(text) and sig[:2] != "00"
    decrypted = text if verified else ""
    return FlowArtifacts(kem_key, kem_ct, sym_ct, sig, verified, decrypted)


def ensure_demo_dirs() -> None:
    for path in (OUTPUTS_DIR, LOGS_DIR, SAMPLES_DIR, PACKS_DIR, UNPACKS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def ensure_sample_docs() -> Path:
    ensure_demo_dirs()
    if any(SAMPLES_DIR.iterdir()):
        return SAMPLES_DIR
    (SAMPLES_DIR / "medical_report.txt").write_text(
        "Patient: demo-001\nStudy: MRI follow-up\nFinding: no acute abnormality.\n",
        encoding="utf-8",
    )
    (SAMPLES_DIR / "lab_panel.csv").write_text(
        "item,value,unit\nWBC,6.1,10^9/L\nHb,132,g/L\nCRP,2.3,mg/L\n",
        encoding="utf-8",
    )
    (SAMPLES_DIR / "risk_features.json").write_text(
        json.dumps(
            {
                "scenario": "financial-risk-demo",
                "features": {"txn_count_7d": 42, "risk_score": 0.18, "region": "demo"},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return SAMPLES_DIR


def _slug(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _iter_input_files(input_dir: Path) -> list[Path]:
    files = [p for p in sorted(input_dir.rglob("*")) if p.is_file()]
    ignored_roots = {PACKS_DIR.resolve(), UNPACKS_DIR.resolve(), LOGS_DIR.resolve(), OUTPUTS_DIR.resolve()}
    ignored_dir_names = {".ipynb_checkpoints", "__pycache__"}
    ignored_suffixes = {".pyc", ".pyo", ".tmp", ".bak"}
    result: list[Path] = []
    for p in files:
        resolved = p.resolve()
        if any(str(resolved).startswith(str(root)) for root in ignored_roots):
            continue
        rel_parts = set(p.relative_to(input_dir).parts)
        if rel_parts & ignored_dir_names:
            continue
        if p.name.startswith(".") or p.suffix.lower() in ignored_suffixes:
            continue
        result.append(p)
    return result


def build_manifest(input_dir: Path, kem_choice: str, sig_choice: str, batch_size: int, mode: str) -> dict[str, Any]:
    files = []
    total = 0
    for path in _iter_input_files(input_dir):
        rel = path.relative_to(input_dir).as_posix()
        size = path.stat().st_size
        total += size
        files.append({"path": rel, "size": size, "sha256": _sha256_file(path)})
    return {
        "version": 1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_dir": str(input_dir),
        "kem": kem_choice,
        "sig_algorithm": sig_choice,
        "batch_size": batch_size,
        "mode": mode,
        "file_count": len(files),
        "total_bytes": total,
        "files": files,
    }


def _keystream(key: bytes, nonce: bytes, length: int) -> bytes:
    out = bytearray()
    counter = 0
    while len(out) < length:
        out.extend(hashlib.sha256(key + nonce + counter.to_bytes(8, "little")).digest())
        counter += 1
    return bytes(out[:length])


def encrypt_bytes(plaintext: bytes, key: bytes) -> dict[str, str]:
    if AESGCM is not None:
        nonce = secrets.token_bytes(12)
        ciphertext = AESGCM(key).encrypt(nonce, plaintext, None)
        return {
            "scheme": "AES-256-GCM",
            "nonce": base64.b64encode(nonce).decode("ascii"),
            "ciphertext": base64.b64encode(ciphertext).decode("ascii"),
        }
    nonce = secrets.token_bytes(16)
    stream = _keystream(key, nonce, len(plaintext))
    ciphertext = bytes(a ^ b for a, b in zip(plaintext, stream))
    tag = hmac.new(key, nonce + ciphertext, hashlib.sha256).digest()
    return {
        "scheme": "SHA256-stream-HMAC-fallback",
        "nonce": base64.b64encode(nonce).decode("ascii"),
        "ciphertext": base64.b64encode(ciphertext).decode("ascii"),
        "tag": base64.b64encode(tag).decode("ascii"),
    }


def decrypt_bytes(record: dict[str, str], key: bytes) -> bytes:
    nonce = base64.b64decode(record["nonce"])
    ciphertext = base64.b64decode(record["ciphertext"])
    if record.get("scheme") == "AES-256-GCM":
        if AESGCM is None:
            raise RuntimeError("AES-GCM package requires cryptography, but it is not installed")
        return AESGCM(key).decrypt(nonce, ciphertext, None)
    tag = base64.b64decode(record["tag"])
    expected = hmac.new(key, nonce + ciphertext, hashlib.sha256).digest()
    if not hmac.compare_digest(tag, expected):
        raise ValueError("ciphertext authentication failed")
    stream = _keystream(key, nonce, len(ciphertext))
    return bytes(a ^ b for a, b in zip(ciphertext, stream))


def _map_kem_exe(kem_choice: str) -> Path:
    mapping = {
        "Kyber-512": "kyber512_amd",
        "Kyber-768": "kyber768_amd",
        "Kyber-1024": "kyber1024_amd",
        "Aigis-enc-1": "aigisenc1_amd",
        "Aigis-enc-2": "aigisenc2_amd",
        "Aigis-enc-3": "aigisenc3_amd",
        "Aigis-enc-4": "aigisenc4_amd",
    }
    return PROJECT_ROOT / "kyberandaigis-enc" / mapping[kem_choice]


def _map_sig_exe(sig_choice: str) -> Path:
    mapping = {
        "ML-DSA-44": "mldsa44_amd",
        "ML-DSA-65": "mldsa65_amd",
        "ML-DSA-87": "mldsa87_amd",
        "Aigis-sig1": "aigis1_amd",
        "Aigis-sig2": "aigis2_amd",
        "Aigis-sig3": "aigis3_amd",
    }
    return PROJECT_ROOT / "mldsaandaigis-sig" / mapping[sig_choice]


def _run_sig_cli(exe: Path, args: list[str], log_path: Path) -> dict[str, Any]:
    return _run_command([str(exe), *args], exe.parent, log_path, timeout=120)


def _run_command(cmd: list[str], cwd: Path, log_path: Path, timeout: int = 120) -> dict[str, Any]:
    env = os.environ.copy()
    rocm_lib = "/opt/python/lib/python3.12/site-packages/_rocm_sdk_devel/lib"
    env["LD_LIBRARY_PATH"] = rocm_lib + ":" + env.get("LD_LIBRARY_PATH", "")
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
        output = proc.stdout
        rc = proc.returncode
    except Exception as exc:
        output = f"{type(exc).__name__}: {exc}\n"
        rc = -1
    elapsed = (time.perf_counter() - t0) * 1000.0
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("$ " + " ".join(cmd) + "\n" + output, encoding="utf-8", errors="ignore")
    return {"returncode": rc, "elapsed_ms": elapsed, "log": str(log_path), "output": output}


def run_rocm_proofs(kem_choice: str, sig_choice: str, batch_size: int, mode: str, stamp: str) -> tuple[dict[str, str], list[str]]:
    logs: dict[str, str] = {}
    notes: list[str] = []
    kem_exe = _map_kem_exe(kem_choice)
    sig_exe = _map_sig_exe(sig_choice)
    proof_batch = max(1, min(int(batch_size), 1024))

    if kem_exe.exists():
        kem_log = LOGS_DIR / f"{stamp}_kem_{_slug(kem_choice)}.log"
        res = _run_command(
            [str(kem_exe), "--batch", str(proof_batch), "--n-ops", "1"],
            kem_exe.parent,
            kem_log,
        )
        logs["kem_rocm"] = res["log"]
        if res["returncode"] != 0:
            notes.append(f"KEM ROCm proof returned {res['returncode']}; see {res['log']}")
    else:
        notes.append(f"KEM executable not found: {kem_exe}")

    if sig_exe.exists():
        sig_log = LOGS_DIR / f"{stamp}_sig_{_slug(sig_choice)}.log"
        cmd = [str(sig_exe), "--batch", str(proof_batch), "--quiet", "--skip-keygen-oracle"]
        if mode == "independent":
            cmd.append("--bench-independent")
        else:
            cmd.append("--bench-paper")
        res = _run_command(cmd, sig_exe.parent, sig_log)
        logs["sig_rocm"] = res["log"]
        if res["returncode"] != 0:
            notes.append(f"SIG ROCm proof returned {res['returncode']}; see {res['log']}")
    else:
        notes.append(f"Signature executable not found: {sig_exe}")

    return logs, notes


def summarize_rocm_logs(logs: dict[str, str]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for name, log_path in logs.items():
        path = Path(log_path)
        item: dict[str, Any] = {"log": str(path), "exists": path.exists()}
        if path.exists():
            text = path.read_text(encoding="utf-8", errors="ignore")
            item["pass"] = "PASS" in text and "FAIL" not in text
            item["has_fail"] = "FAIL" in text
            item["returncode_hint"] = "error" in text.lower() or "not found" in text.lower()
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            item["tail"] = lines[-8:]
        summary[name] = item
    return summary


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _zip_dir(src_dir: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(src_dir.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(src_dir).as_posix())


def _package_mac(key: bytes, manifest: dict[str, Any]) -> str:
    excluded = {
        "package_authenticator",
        "signature_backend",
        "kem_backend",
        "sig_public_key",
        "sig_secret_key_demo",
        "manifest_signature",
        "sig_payload",
        "sig_api_batch",
        "sig_cli_verify_log",
        "sig_api_verify_log",
    }
    signed = {
        k: v
        for k, v in manifest.items()
        if k not in excluded
    }
    payload = json.dumps(signed, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hmac.new(key, payload, hashlib.sha256).hexdigest()


def _signature_payload(manifest: dict[str, Any]) -> dict[str, Any]:
    excluded = {
        "signature_backend",
        "sig_public_key",
        "sig_secret_key_demo",
        "manifest_signature",
        "sig_payload",
        "sig_api_batch",
        "sig_cli_verify_log",
        "sig_api_verify_log",
    }
    return {k: v for k, v in manifest.items() if k not in excluded}


def _write_signature_payload(path: Path, manifest: dict[str, Any]) -> None:
    _write_json(path, _signature_payload(manifest))


def _derive_aes_key(shared_secret: bytes) -> bytes:
    return hashlib.sha256(shared_secret).digest()


def _try_create_kem_api_session(
    kem_choice: str,
    batch_size: int,
    pack_dir: Path,
    stamp: str,
) -> tuple[bytes | None, dict[str, Any], dict[str, str], list[str]]:
    kem_exe = _map_kem_exe(kem_choice)
    logs: dict[str, str] = {}
    notes: list[str] = []
    manifest_fields: dict[str, Any] = {}
    if not kem_exe.exists():
        return None, manifest_fields, logs, [f"KEM executable not found for API: {kem_exe}"]

    api_batch = max(1, min(int(batch_size), 1024))
    kem_dir = pack_dir / "kem"
    kem_dir.mkdir(parents=True, exist_ok=True)
    pk_path = kem_dir / "kem_pk.bin"
    sk_path = kem_dir / "receiver_sk.demo_secret"
    ct_path = kem_dir / "kem_ct.bin"
    ss_sender_path = kem_dir / "ss_sender.demo_secret"

    keygen_log = LOGS_DIR / f"{stamp}_kemapi_keygen_{_slug(kem_choice)}.log"
    encaps_log = LOGS_DIR / f"{stamp}_kemapi_encaps_{_slug(kem_choice)}.log"
    kg = _run_command(
        [str(kem_exe), "--api-kem-keygen", "--batch", str(api_batch), "--pk-out", str(pk_path), "--sk-out", str(sk_path)],
        kem_exe.parent,
        keygen_log,
    )
    logs["kem_api_keygen"] = str(keygen_log)
    if kg["returncode"] != 0:
        notes.append(f"KEM API keygen not active; rc={kg['returncode']}; see {keygen_log}")
        return None, manifest_fields, logs, notes

    enc = _run_command(
        [
            str(kem_exe),
            "--api-kem-encaps",
            "--batch",
            str(api_batch),
            "--pk-in",
            str(pk_path),
            "--ct-out",
            str(ct_path),
            "--ss-out",
            str(ss_sender_path),
        ],
        kem_exe.parent,
        encaps_log,
    )
    logs["kem_api_encaps"] = str(encaps_log)
    if enc["returncode"] != 0 or not ss_sender_path.exists() or not ct_path.exists():
        notes.append(f"KEM API encaps not active; rc={enc['returncode']}; see {encaps_log}")
        return None, manifest_fields, logs, notes

    shared_secret = ss_sender_path.read_bytes()
    manifest_fields.update(
        {
            "kem_backend": "ROCm KEM batch file API",
            "kem_api_batch": api_batch,
            "kem_public_key": "kem/kem_pk.bin",
            "kem_ciphertext_file": "kem/kem_ct.bin",
            "kem_receiver_secret_demo": "kem/receiver_sk.demo_secret",
            "kem_ciphertext": _sha256_file(ct_path),
        }
    )
    return _derive_aes_key(shared_secret), manifest_fields, logs, notes


def _pack_path(pack_dir: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else pack_dir / path


def _recover_kem_api_session(manifest: dict[str, Any], pack_dir: Path) -> tuple[bytes | None, bool, str]:
    kem_choice = manifest.get("kem", "Kyber-768")
    kem_exe = _map_kem_exe(kem_choice)
    sk_path = _pack_path(pack_dir, manifest.get("kem_receiver_secret_demo"))
    ct_path = _pack_path(pack_dir, manifest.get("kem_ciphertext_file"))
    if not kem_exe.exists() or not sk_path or not ct_path:
        return None, False, ""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ss_receiver_path = pack_dir / "kem" / f"ss_receiver_{stamp}.demo_secret"
    verify_log = LOGS_DIR / f"{stamp}_kemapi_decaps_{_slug(kem_choice)}.log"
    res = _run_command(
        [
            str(kem_exe),
            "--api-kem-decaps",
            "--batch",
            str(max(1, int(manifest.get("kem_api_batch", 128)))),
            "--sk-in",
            str(sk_path),
            "--ct-in",
            str(ct_path),
            "--ss-out",
            str(ss_receiver_path),
        ],
        kem_exe.parent,
        verify_log,
    )
    if res["returncode"] != 0 or not ss_receiver_path.exists():
        return None, False, str(verify_log)
    return _derive_aes_key(ss_receiver_path.read_bytes()), True, str(verify_log)


def _try_create_sig_api_signature(
    manifest: dict[str, Any],
    sig_choice: str,
    batch_size: int,
    pack_dir: Path,
    stamp: str,
) -> tuple[dict[str, Any], dict[str, str], list[str]]:
    sig_exe = _map_sig_exe(sig_choice)
    logs: dict[str, str] = {}
    notes: list[str] = []
    fields: dict[str, Any] = {}
    if not sig_exe.exists():
        return fields, logs, [f"Signature executable not found for API: {sig_exe}"]

    api_batch = max(1, min(int(batch_size), 1024))
    sig_dir = pack_dir / "sig"
    sig_dir.mkdir(parents=True, exist_ok=True)
    payload_path = sig_dir / "manifest.payload.json"
    pk_path = sig_dir / "sig_pk.bin"
    sk_path = sig_dir / "sig_sk.demo_secret"
    sig_path = sig_dir / "manifest.sig"
    sign_log = LOGS_DIR / f"{stamp}_sigapi_sign_{_slug(sig_choice)}.log"
    verify_log = LOGS_DIR / f"{stamp}_sigapi_verify_{_slug(sig_choice)}.log"

    _write_signature_payload(payload_path, manifest)
    sign = _run_command(
        [
            str(sig_exe),
            "--api-sig-sign",
            "--batch",
            str(api_batch),
            "--msg-in",
            str(payload_path),
            "--pk-out",
            str(pk_path),
            "--sk-out",
            str(sk_path),
            "--sig-out",
            str(sig_path),
        ],
        sig_exe.parent,
        sign_log,
        timeout=240,
    )
    logs["sig_api_sign"] = str(sign_log)
    if sign["returncode"] != 0 or not pk_path.exists() or not sig_path.exists():
        notes.append(f"SIG API sign not active; rc={sign['returncode']}; see {sign_log}")
        return fields, logs, notes

    verify = _run_command(
        [
            str(sig_exe),
            "--api-sig-verify",
            "--batch",
            str(api_batch),
            "--msg-in",
            str(payload_path),
            "--pk-in",
            str(pk_path),
            "--sig-in",
            str(sig_path),
        ],
        sig_exe.parent,
        verify_log,
        timeout=240,
    )
    logs["sig_api_verify"] = str(verify_log)
    if verify["returncode"] != 0:
        notes.append(f"SIG API verify not active; rc={verify['returncode']}; see {verify_log}")
        return fields, logs, notes

    fields.update(
        {
            "signature_backend": "ROCm ML-DSA/Aigis-sig batch file API",
            "sig_api_batch": api_batch,
            "sig_payload": "sig/manifest.payload.json",
            "sig_public_key": "sig/sig_pk.bin",
            "sig_secret_key_demo": "sig/sig_sk.demo_secret",
            "manifest_signature": "sig/manifest.sig",
        }
    )
    return fields, logs, notes


def _verify_sig_api_signature(manifest: dict[str, Any], pack_dir: Path) -> tuple[bool | None, str]:
    if manifest.get("signature_backend") != "ROCm ML-DSA/Aigis-sig batch file API":
        return None, ""
    sig_alg = manifest.get("sig_algorithm", "ML-DSA-65")
    sig_exe = _map_sig_exe(sig_alg)
    pk_path = _pack_path(pack_dir, manifest.get("sig_public_key"))
    sig_path = _pack_path(pack_dir, manifest.get("manifest_signature"))
    if not sig_exe.exists() or not pk_path or not sig_path:
        return False, ""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    payload_path = pack_dir / "sig" / f"manifest.payload.verify_{stamp}.json"
    verify_log = LOGS_DIR / f"{stamp}_sigapi_unpack_verify_{_slug(sig_alg)}.log"
    _write_signature_payload(payload_path, manifest)
    res = _run_command(
        [
            str(sig_exe),
            "--api-sig-verify",
            "--batch",
            str(max(1, int(manifest.get("sig_api_batch", 128)))),
            "--msg-in",
            str(payload_path),
            "--pk-in",
            str(pk_path),
            "--sig-in",
            str(sig_path),
        ],
        sig_exe.parent,
        verify_log,
        timeout=240,
    )
    return res["returncode"] == 0, str(verify_log)


def create_secure_pack(
    input_dir: str | Path | None,
    kem_choice: str,
    sig_choice: str,
    batch_size: int,
    mode: str,
    run_rocm: bool = True,
) -> RealFlowResult:
    ensure_demo_dirs()
    source_dir = Path(input_dir).expanduser().resolve() if input_dir else ensure_sample_docs().resolve()
    if not source_dir.exists() or not source_dir.is_dir():
        raise FileNotFoundError(f"input directory not found: {source_dir}")
    manifest = build_manifest(source_dir, kem_choice, sig_choice, batch_size, mode)
    if manifest["file_count"] <= 0:
        raise ValueError(f"input directory has no files: {source_dir}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pack_dir = PACKS_DIR / f"pack_{stamp}_{_slug(kem_choice)[:10]}_{_slug(sig_choice)[:10]}"
    enc_dir = pack_dir / "encrypted_files"
    enc_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    session_key = secrets.token_bytes(32)
    kem_ciphertext = hashlib.sha256(session_key + kem_choice.encode("utf-8")).hexdigest()
    kem_api_logs: dict[str, str] = {}
    kem_api_notes: list[str] = []
    kem_manifest_fields: dict[str, Any] = {}
    if run_rocm:
        kem_session_key, kem_manifest_fields, kem_api_logs, kem_api_notes = _try_create_kem_api_session(
            kem_choice, batch_size, pack_dir, stamp
        )
        if kem_session_key is not None:
            session_key = kem_session_key
            kem_ciphertext = kem_manifest_fields.get("kem_ciphertext", kem_ciphertext)
        else:
            kem_api_notes.append("KEM API unavailable; using demo session key capsule fallback")
    timings = {"prepare_ms": (time.perf_counter() - t0) * 1000.0}

    t1 = time.perf_counter()
    encrypted_records = []
    for entry in manifest["files"]:
        src = source_dir / entry["path"]
        record = encrypt_bytes(src.read_bytes(), session_key)
        out_name = hashlib.sha256(entry["path"].encode("utf-8")).hexdigest()[:20] + ".json"
        out_path = enc_dir / out_name
        _write_json(out_path, record)
        encrypted_records.append({"path": entry["path"], "encrypted": f"encrypted_files/{out_name}"})
    timings["encrypt_ms"] = (time.perf_counter() - t1) * 1000.0

    manifest["encrypted_files"] = encrypted_records
    manifest["kem_ciphertext"] = kem_ciphertext
    manifest["signature_backend"] = "ROCm batch/decomp proof + package authenticator"
    manifest["kem_backend"] = "ROCm CLI proof + SHA-256 package key capsule"
    manifest.update(kem_manifest_fields)
    (pack_dir / "session_key.demo_secret").write_text(
        base64.b64encode(session_key).decode("ascii"),
        encoding="ascii",
    )

    manifest_path = pack_dir / "manifest.json"
    signature = _package_mac(session_key, manifest)
    manifest["package_authenticator"] = signature
    _write_json(manifest_path, manifest)

    sig_api_logs: dict[str, str] = {}
    sig_api_notes: list[str] = []
    if run_rocm:
        sig_fields, sig_api_logs, sig_api_notes = _try_create_sig_api_signature(
            manifest, sig_choice, batch_size, pack_dir, stamp
        )
        if sig_fields:
            manifest.update(sig_fields)
            signature = _package_mac(session_key, manifest)
            manifest["package_authenticator"] = signature
            _write_json(manifest_path, manifest)
            payload = _pack_path(pack_dir, manifest.get("sig_payload"))
            if payload:
                _write_signature_payload(payload, manifest)
        else:
            sig_api_notes.append("SIG API unavailable; using package authenticator fallback")

    rocm_logs: dict[str, str] = {**kem_api_logs, **sig_api_logs}
    notes: list[str] = [*kem_api_notes, *sig_api_notes]
    if run_rocm:
        t2 = time.perf_counter()
        rocm_logs, proof_notes = run_rocm_proofs(kem_choice, sig_choice, batch_size, mode, stamp)
        rocm_logs = {**kem_api_logs, **sig_api_logs, **rocm_logs}
        notes.extend(proof_notes)
        timings["rocm_proof_ms"] = (time.perf_counter() - t2) * 1000.0

    zip_path = pack_dir.with_suffix(".pqcpack.zip")
    _zip_dir(pack_dir, zip_path)

    unpack_dir = UNPACKS_DIR / f"unpack_{stamp}"
    verify = unpack_secure_pack(pack_dir, unpack_dir)
    return RealFlowResult(
        pack_dir=str(pack_dir),
        pack_zip=str(zip_path),
        unpack_dir=str(unpack_dir),
        manifest_path=str(manifest_path),
        plaintext_dir=str(source_dir),
        verified=verify["verified"],
        tamper_detected=False,
        kem_shared_key=hashlib.sha256(session_key).hexdigest(),
        kem_ciphertext=kem_ciphertext,
        signature=signature,
        file_count=manifest["file_count"],
        total_bytes=manifest["total_bytes"],
        timings_ms={**timings, **verify["timings_ms"]},
        rocm_logs=rocm_logs,
        notes=notes,
    )


def unpack_secure_pack(pack_dir: str | Path, out_dir: str | Path | None = None) -> dict[str, Any]:
    pack_dir = Path(pack_dir).resolve()
    out_dir = Path(out_dir).resolve() if out_dir else (UNPACKS_DIR / f"unpack_{pack_dir.name}")
    t0 = time.perf_counter()
    manifest = json.loads((pack_dir / "manifest.json").read_text(encoding="utf-8"))
    kem_ok = True
    kem_log = ""
    if manifest.get("kem_backend") == "ROCm KEM batch file API":
        recovered_key, kem_ok, kem_log = _recover_kem_api_session(manifest, pack_dir)
        if recovered_key is not None:
            key = recovered_key
        else:
            key = base64.b64decode((pack_dir / "session_key.demo_secret").read_text(encoding="ascii"))
    else:
        key = base64.b64decode((pack_dir / "session_key.demo_secret").read_text(encoding="ascii"))
    signature = manifest.get("package_authenticator", "")
    sig_ok = hmac.compare_digest(signature, _package_mac(key, manifest))
    sig_cli_ok = None
    sig_cli_log = ""
    sig_api_ok, sig_api_log = _verify_sig_api_signature(manifest, pack_dir)
    if manifest.get("signature_backend") == "ROCm ML-DSA/Aigis-sig CLI":
        sig_alg = manifest.get("sig_algorithm", "ML-DSA-65")
        sig_exe = _map_sig_exe(sig_alg)
        sig_pk = manifest.get("sig_public_key")
        sig_file = manifest.get("manifest_signature")
        if sig_exe.exists() and sig_pk and sig_file:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            verify_log = LOGS_DIR / f"{stamp}_unpack_sigcli_verify_{_slug(sig_alg)}.log"
            res = _run_sig_cli(
                sig_exe,
                ["--cli-verify", "--pk-in", sig_pk, "--msg-in", str(pack_dir / "manifest.json"), "--sig-in", sig_file],
                verify_log,
            )
            sig_cli_ok = res["returncode"] == 0
            sig_cli_log = str(verify_log)
        else:
            sig_cli_ok = False

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    restored = 0
    total = 0
    file_errors = []
    enc_map = {item["path"]: item["encrypted"] for item in manifest.get("encrypted_files", [])}
    for entry in manifest["files"]:
        rel = entry["path"]
        try:
            enc_record = json.loads((pack_dir / enc_map[rel]).read_text(encoding="utf-8"))
            plaintext = decrypt_bytes(enc_record, key)
            out_path = out_dir / rel
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(plaintext)
            actual = hashlib.sha256(plaintext).hexdigest()
            if actual != entry["sha256"]:
                file_errors.append(f"{rel}: sha256 mismatch")
            restored += 1
            total += len(plaintext)
        except Exception as exc:
            file_errors.append(f"{rel}: {exc}")

    return {
        "verified": kem_ok and sig_ok and (sig_cli_ok is not False) and (sig_api_ok is not False) and not file_errors,
        "kem_ok": kem_ok,
        "kem_log": kem_log,
        "signature_ok": sig_ok,
        "sig_cli_ok": sig_cli_ok,
        "sig_cli_log": sig_cli_log,
        "sig_api_ok": sig_api_ok,
        "sig_api_log": sig_api_log,
        "file_errors": file_errors,
        "restored_files": restored,
        "restored_bytes": total,
        "out_dir": str(out_dir),
        "timings_ms": {"verify_decrypt_ms": (time.perf_counter() - t0) * 1000.0},
    }


def tamper_pack(pack_dir: str | Path) -> str:
    pack_dir = Path(pack_dir).resolve()
    candidates = sorted((pack_dir / "encrypted_files").glob("*.json"))
    if not candidates:
        raise FileNotFoundError("no encrypted file to tamper")
    target = candidates[0]
    data = json.loads(target.read_text(encoding="utf-8"))
    raw = bytearray(base64.b64decode(data["ciphertext"]))
    if raw:
        raw[0] ^= 1
    data["ciphertext"] = base64.b64encode(bytes(raw)).decode("ascii")
    target.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(target)


def create_tampered_copy_and_verify(pack_dir: str | Path) -> dict[str, Any]:
    pack_dir = Path(pack_dir).resolve()
    tampered = pack_dir.with_name(pack_dir.name + "_tampered")
    if tampered.exists():
        shutil.rmtree(tampered)
    shutil.copytree(pack_dir, tampered)
    tampered_file = tamper_pack(tampered)
    result = unpack_secure_pack(tampered, UNPACKS_DIR / f"tampered_{pack_dir.name}")
    result["tampered_pack_dir"] = str(tampered)
    result["tampered_file"] = tampered_file
    result["tamper_detected"] = not result["verified"]
    return result
