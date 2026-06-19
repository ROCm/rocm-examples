from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Callable
import json
import time

import ipywidgets as widgets
from IPython.display import display

from .backends import (
    create_secure_pack,
    create_tampered_copy_and_verify,
    ensure_sample_docs,
    unpack_secure_pack,
    summarize_rocm_logs,
)
from .state import TrustFlowState


BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"
OUTPUTS_DIR = BASE_DIR / "outputs"
LOGS_DIR = BASE_DIR / "logs"


def ensure_dirs() -> None:
    for path in (ASSETS_DIR, OUTPUTS_DIR, LOGS_DIR):
        path.mkdir(parents=True, exist_ok=True)


STATUS_LABELS = {
    "done": "完成",
    "busy": "运行中",
    "fail": "失败",
    "idle": "待执行",
}


def _fmt_status(status: str) -> str:
    color = {"done": "#1f7a4f", "busy": "#b26a00", "fail": "#a61b1b", "idle": "#58606b"}.get(status, "#58606b")
    label = STATUS_LABELS.get(status, status)
    return f'<span style="color:{color};font-weight:600">{label}</span>'


def _json_artifact(artifacts: dict[str, str], key: str, default):
    value = artifacts.get(key)
    if not value:
        return default
    try:
        return json.loads(value)
    except Exception:
        return default


def _fmt_ms(value: float | int | None) -> str:
    if value is None:
        return "-"
    return f"{float(value):.2f} ms"


def _fmt_bool(value) -> str:
    if value is True:
        return "通过"
    if value is False:
        return "失败"
    return "未启用"


def _format_artifacts(state: TrustFlowState, folder: str) -> str:
    artifacts = state.artifacts
    timings = state.timings_ms
    rocm_summary = _json_artifact(artifacts, "rocm_summary", {})
    verify_detail = _json_artifact(artifacts, "verify_detail", {})
    tamper_detail = _json_artifact(artifacts, "tamper_detail", {})

    lines: list[str] = []
    lines.append("一、本次配置")
    lines.append(f"  输入目录: {folder}")
    lines.append(f"  KEM 算法: {state.kem_choice}")
    lines.append(f"  签名算法: {state.sig_choice}")
    lines.append(f"  Batch 大小: {state.batch_size}")
    lines.append(f"  性能模式: {state.mode}")
    lines.append("")

    lines.append("二、核心结果")
    normal_verified = verify_detail.get("verified")
    tamper_detected = artifacts.get("tamper_detected") or ("YES" if tamper_detail.get("tamper_detected") else "")
    lines.append(f"  正常包验证: {_fmt_bool(normal_verified)}")
    lines.append(f"  KEM 解封装: {_fmt_bool(verify_detail.get('kem_ok'))}")
    lines.append(f"  Manifest 包认证: {_fmt_bool(verify_detail.get('signature_ok'))}")
    lines.append(f"  ML-DSA/Aigis-sig 验签: {_fmt_bool(verify_detail.get('sig_api_ok'))}")
    lines.append(f"  文件恢复数量: {verify_detail.get('restored_files', '-')}")
    lines.append(f"  篡改检测: {tamper_detected or '未执行'}")
    if tamper_detail:
        lines.append(f"  篡改包验证结果: {_fmt_bool(tamper_detail.get('verified'))}")
        errors = tamper_detail.get("file_errors") or []
        if errors:
            lines.append(f"  篡改定位: {errors[0]}")
    lines.append("")

    lines.append("三、传输包与恢复目录")
    lines.append(f"  安全包目录: {artifacts.get('pack_dir', '-')}")
    lines.append(f"  安全包 zip: {artifacts.get('pack_zip', '-')}")
    lines.append(f"  Manifest: {artifacts.get('manifest', '-')}")
    lines.append(f"  恢复目录: {artifacts.get('restored_dir') or artifacts.get('unpack_dir', '-')}")
    lines.append(f"  文件数量: {artifacts.get('file_count', '-')}")
    lines.append(f"  文件总大小: {artifacts.get('total_bytes', '-')} bytes")
    lines.append("")

    lines.append("四、密码学证据")
    lines.append(f"  KEM 密文摘要: {artifacts.get('kem_ciphertext', '-')}")
    lines.append(f"  Shared secret 摘要: {artifacts.get('kem_shared_key_sha256', '-')}")
    lines.append(f"  包认证值: {artifacts.get('package_authenticator', '-')}")
    lines.append("  说明: KEM shared secret 只显示 SHA-256 摘要，不直接暴露原始密钥。")
    lines.append("")

    lines.append("五、AMD ROCm 后端调用")
    log_names = {
        "kem_api_keygen": "KEM keygen 文件接口",
        "kem_api_encaps": "KEM encaps 文件接口",
        "sig_api_sign": "签名 batch/decomp 文件接口",
        "sig_api_verify": "验签 batch 文件接口",
        "kem_rocm": "KEM 性能/正确性 proof",
        "sig_rocm": "签名性能/正确性 proof",
    }
    for key, label in log_names.items():
        item = rocm_summary.get(key)
        if not item:
            continue
        tail = item.get("tail") or []
        last_line = tail[-1] if tail else ""
        lines.append(f"  {label}: {'PASS' if item.get('pass') else 'CHECK'}")
        if last_line:
            lines.append(f"    摘要: {last_line}")
        lines.append(f"    日志: {item.get('log', '-')}")
    if not any(k in rocm_summary for k in log_names):
        lines.append("  尚未生成 ROCm 日志。")
    lines.append("")

    lines.append("六、耗时")
    lines.append(f"  生成安全包总耗时: {_fmt_ms(timings.get('encaps'))}")
    lines.append(f"  AES 文件加密: {_fmt_ms(timings.get('encrypt_ms'))}")
    lines.append(f"  ROCm proof: {_fmt_ms(timings.get('rocm_proof_ms'))}")
    lines.append(f"  解包/验签/解密: {_fmt_ms(timings.get('verify_decrypt_ms'))}")
    lines.append(f"  篡改测试: {_fmt_ms(timings.get('decaps'))}")
    lines.append("")

    lines.append("七、备注")
    notes = _json_artifact(artifacts, "notes", [])
    if notes:
        for note in notes:
            lines.append(f"  - {note}")
    else:
        lines.append("  无异常备注。")
    return "\n".join(lines)


def build_app() -> widgets.VBox:
    ensure_dirs()
    state = TrustFlowState()
    sample_dir = ensure_sample_docs()
    last_pack_dir = ""

    title = widgets.HTML(
        "<h2 style='margin:0'>PQC TrustFlow</h2>"
        "<div style='color:#58606b'>基于 AMD ROCm 的后量子多文档加密传输与可信验证演示</div>"
    )
    source_dir_input = widgets.Text(
        value=str(sample_dir),
        description="文件夹",
        layout=widgets.Layout(width="100%"),
    )
    sensitive_input = widgets.Textarea(value="敏感数据传输样例", description="备注", layout=widgets.Layout(width="100%", height="70px"))
    kem_choice = widgets.Dropdown(options=["Kyber-512", "Kyber-768", "Kyber-1024", "Aigis-enc-1", "Aigis-enc-2", "Aigis-enc-3", "Aigis-enc-4"], value="Kyber-768", description="KEM")
    sig_choice = widgets.Dropdown(options=["ML-DSA-44", "ML-DSA-65", "ML-DSA-87", "Aigis-sig1", "Aigis-sig2", "Aigis-sig3"], value="ML-DSA-65", description="签名")
    batch_size = widgets.Dropdown(options=[128, 1024, 8192, 16384, 32768], value=1024, description="Batch")
    mode_choice = widgets.ToggleButtons(options=["paper", "independent"], value="paper", description="模式")

    status_html = widgets.HTML(value=_fmt_status("idle"))
    transcript = widgets.Textarea(value="", description="日志", layout=widgets.Layout(width="100%", height="220px"), disabled=True)
    artifact_box = widgets.Textarea(value="", description="证据", layout=widgets.Layout(width="100%", height="360px"), disabled=True)

    buttons = {
        "prepare": widgets.Button(description="准备"),
        "encaps": widgets.Button(description="生成安全包", button_style="info"),
        "encrypt": widgets.Button(description="查看安全包", button_style="info"),
        "sign": widgets.Button(description="查看证明", button_style="warning"),
        "verify": widgets.Button(description="解包并验证", button_style="success"),
        "decaps": widgets.Button(description="篡改测试", button_style="danger"),
        "decrypt": widgets.Button(description="查看恢复目录", button_style="info"),
        "run_all": widgets.Button(description="一键运行", button_style="success"),
        "reset": widgets.Button(description="重置"),
    }

    def refresh_views() -> None:
        status_html.value = (
            "<div style='display:flex;gap:18px;flex-wrap:wrap'>"
            f"<div>准备: {_fmt_status(state.stage_status['prepare'])}</div>"
            f"<div>生成安全包: {_fmt_status(state.stage_status['encaps'])}</div>"
            f"<div>查看安全包: {_fmt_status(state.stage_status['encrypt'])}</div>"
            f"<div>查看证明: {_fmt_status(state.stage_status['sign'])}</div>"
            f"<div>解包验证: {_fmt_status(state.stage_status['verify'])}</div>"
            f"<div>篡改测试: {_fmt_status(state.stage_status['decaps'])}</div>"
            f"<div>恢复目录: {_fmt_status(state.stage_status['decrypt'])}</div>"
            "</div>"
        )
        transcript.value = "\n".join(state.transcript)
        artifact_box.value = _format_artifacts(state, source_dir_input.value)

    def save_snapshot() -> None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        state.save(OUTPUTS_DIR / f"trustflow_snapshot_{stamp}.json")

    def sync_choices() -> None:
        state.sensitive_text = sensitive_input.value
        state.kem_choice = kem_choice.value
        state.sig_choice = sig_choice.value
        state.batch_size = int(batch_size.value)
        state.mode = mode_choice.value

    def current_pack_dir() -> str:
        return last_pack_dir or state.artifacts.get("pack_dir", "")

    def run_stage(stage: str, message: str, fn: Callable[[], None]) -> None:
        sync_choices()
        state.set_stage(stage, "busy")
        state.add_event(message)
        refresh_views()
        t0 = time.perf_counter()
        try:
            fn()
            state.set_stage(stage, "done")
        except Exception as exc:
            state.set_stage(stage, "fail")
            state.add_event(f"{stage} 失败: {exc!r}")
        finally:
            state.set_timing(stage, (time.perf_counter() - t0) * 1000.0)
            refresh_views()
            save_snapshot()

    def do_prepare(_=None):
        sync_choices()
        state.set_artifact("input_folder", source_dir_input.value)
        state.add_event("已记录输入文件夹和算法选择")
        state.set_stage("prepare", "done")
        refresh_views()
        save_snapshot()

    def do_encaps(_=None):
        nonlocal last_pack_dir
        def inner():
            result = create_secure_pack(
                source_dir_input.value,
                state.kem_choice,
                state.sig_choice,
                state.batch_size,
                state.mode,
                run_rocm=True,
            )
            last_pack_dir = result.pack_dir
            state.verified = result.verified
            state.set_artifact("pack_dir", result.pack_dir)
            state.set_artifact("pack_zip", result.pack_zip)
            state.set_artifact("manifest", result.manifest_path)
            state.set_artifact("unpack_dir", result.unpack_dir)
            state.set_artifact("kem_ciphertext", result.kem_ciphertext)
            state.set_artifact("kem_shared_key_sha256", result.kem_shared_key)
            state.set_artifact("package_authenticator", result.signature)
            state.set_artifact("file_count", str(result.file_count))
            state.set_artifact("total_bytes", str(result.total_bytes))
            state.set_artifact("rocm_logs", json.dumps(result.rocm_logs, ensure_ascii=False))
            state.set_artifact("rocm_summary", json.dumps(summarize_rocm_logs(result.rocm_logs), ensure_ascii=False, indent=2))
            state.set_artifact("notes", json.dumps(result.notes, ensure_ascii=False))
            for key, value in result.timings_ms.items():
                state.set_timing(key, value)
        run_stage("encaps", f"正在生成安全包: {kem_choice.value} + {sig_choice.value}", inner)

    def do_encrypt(_=None):
        def inner():
            pack_dir = current_pack_dir()
            if not pack_dir:
                raise RuntimeError("create a pack first")
            state.set_artifact("encrypted_pack", pack_dir)
            state.add_event("密文文件位于安全包的 encrypted_files 目录")
        run_stage("encrypt", "正在显示安全包位置", inner)

    def do_sign(_=None):
        def inner():
            if "package_authenticator" not in state.artifacts:
                raise RuntimeError("create a pack first")
            state.add_event(f"{sig_choice.value} 签名/包认证证明已生成")
        run_stage("sign", f"正在显示 {sig_choice.value} 证明信息", inner)

    def do_verify(_=None):
        def inner():
            pack_dir = current_pack_dir()
            if not pack_dir:
                raise RuntimeError("create a pack first")
            result = unpack_secure_pack(pack_dir)
            state.verified = bool(result["verified"])
            state.set_artifact("verify", "PASS" if result["verified"] else "FAIL")
            state.set_artifact("restored_dir", result["out_dir"])
            state.set_artifact("verify_detail", json.dumps(result, ensure_ascii=False))
            for key, value in result["timings_ms"].items():
                state.set_timing(key, value)
        run_stage("verify", f"正在解包、验签并恢复文件: {sig_choice.value}", inner)

    def do_decaps(_=None):
        def inner():
            pack_dir = current_pack_dir()
            if not pack_dir:
                raise RuntimeError("create a pack first")
            result = create_tampered_copy_and_verify(pack_dir)
            state.set_artifact("tampered_pack_dir", result["tampered_pack_dir"])
            state.set_artifact("tampered_file", result["tampered_file"])
            state.set_artifact("tamper_detected", "YES" if result["tamper_detected"] else "NO")
            state.set_artifact("tamper_detail", json.dumps(result, ensure_ascii=False))
            state.verified = not result["tamper_detected"]
        run_stage("decaps", "正在篡改一个密文文件并验证检测能力", inner)

    def do_decrypt(_=None):
        def inner():
            restored = state.artifacts.get("restored_dir") or state.artifacts.get("unpack_dir")
            if not restored:
                raise RuntimeError("unpack a pack first")
            state.set_artifact("restored_dir", restored)
            state.add_event(f"恢复后的文件目录: {restored}")
        run_stage("decrypt", "正在显示恢复目录", inner)

    def do_run_all(_=None):
        do_prepare()
        do_encaps()
        do_verify()
        state.add_event("完整流程已完成")
        refresh_views()
        save_snapshot()

    def do_reset(_=None):
        nonlocal state, last_pack_dir
        state = TrustFlowState()
        last_pack_dir = ""
        source_dir_input.value = str(sample_dir)
        sensitive_input.value = "敏感数据传输样例"
        kem_choice.value = "Kyber-768"
        sig_choice.value = "ML-DSA-65"
        batch_size.value = 1024
        mode_choice.value = "paper"
        state.add_event("状态已重置")
        refresh_views()
        save_snapshot()

    buttons["prepare"].on_click(do_prepare)
    buttons["encaps"].on_click(do_encaps)
    buttons["encrypt"].on_click(do_encrypt)
    buttons["sign"].on_click(do_sign)
    buttons["verify"].on_click(do_verify)
    buttons["decaps"].on_click(do_decaps)
    buttons["decrypt"].on_click(do_decrypt)
    buttons["run_all"].on_click(do_run_all)
    buttons["reset"].on_click(do_reset)

    controls = widgets.VBox([
        widgets.HBox([kem_choice, sig_choice]),
        widgets.HBox([batch_size, mode_choice]),
        widgets.HBox(list(buttons.values()), layout=widgets.Layout(flex_flow="row wrap")),
    ])
    panels = widgets.Tab(children=[widgets.VBox([status_html, transcript]), widgets.VBox([artifact_box])])
    panels.set_title(0, "流程")
    panels.set_title(1, "结果与证据")
    root = widgets.VBox([title, source_dir_input, sensitive_input, controls, panels])
    state.add_event("前端已加载")
    refresh_views()
    save_snapshot()
    return root


def launch_app() -> None:
    display(build_app())
