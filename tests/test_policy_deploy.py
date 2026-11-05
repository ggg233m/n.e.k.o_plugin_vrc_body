# -*- coding: utf-8 -*-
"""policy_deploy 的 preflight 负向测试（2026-11-05）。

核心安全属性：**坏的部署配置必须以非零码被拒绝**，不能「形状对就放行」。
这里专门构造各类「形状对但语义错」的 manifest，逐个确认 preflight 会挡住。

不加载模型、不需要 GPU、不碰真实数据集（用临时目录 + 假文件）。
"""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "research" / "tools" / "policy_deploy.py"
PY = str(ROOT / ".venv_xpu" / "Scripts" / "python.exe")


def run_preflight(man_path):
    cmd = [PY, str(TOOL), "preflight", "--manifest", str(man_path)]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def base_manifest(tmp: Path, **over):
    """构造一个最小合法 manifest，再按需覆盖字段。"""
    head = tmp / "head.pt"
    adapter = tmp / "adapter_model.safetensors"
    schema_p = tmp / "input_schema.json"
    model_dir = tmp / "model"
    model_dir.mkdir(exist_ok=True)
    import hashlib
    head.write_bytes(b"fake-head")
    adapter.write_bytes(b"fake-adapter")
    schema_p.write_text("{}", encoding="utf-8")
    sys.path.insert(0, str(ROOT / "research" / "tools"))
    import policy_deploy as D
    schema = dict(D.REQUIRED_SCHEMA)
    schema.update(dict(drop_wp=True, drop_topo=False, omega_unit="rad_s",
                       prompt_format="card", quad_feats=False))
    schema.update(over.pop("schema", {}))
    man = {
        "kind": "vrc_policy_deployment",
        "version": 1,
        "model_dir": str(model_dir),
        "run_dir": str(tmp),
        "files": {
            "head": {"path": str(head), "sha256": D.sha256(head)},
            "adapter": {"path": str(adapter), "sha256": D.sha256(adapter)},
            "input_schema": {"path": str(schema_p), "sha256": D.sha256(schema_p)},
        },
        "schema": schema,
        "training": {"dataset": None},
        "calibration": {"v_max_mps": 1.2, "dv_max_mps2": 0.5,
                        "w_max_rad_s": 1.57, "dw_max_rad_s2": 3.14},
        "runtime_env": {"VRC_POLICY_STALE": "0.8"},
    }
    man.update(over)
    p = tmp / "man.json"
    p.write_text(json.dumps(man), encoding="utf-8")
    return p, man


class PreflightRejectsBadConfigTests(unittest.TestCase):
    """每一条都对应一个「看起来能跑、实际错口径」的真实风险。"""

    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.tmp = Path(self._td.name)

    def tearDown(self):
        self._td.cleanup()

    def test_act_dim_2_rejected(self):
        """act_dim=2（没有 omega 通道）必须被拒 —— 否则 omega 永远不执行。"""
        p, _ = base_manifest(self.tmp, schema={"act_dim": 2});
        rc, out = run_preflight(p)
        self.assertNotEqual(rc, 0, "act_dim=2 竟然通过了 preflight")
        self.assertIn("act_dim", out)

    def test_wrong_history_frame_rejected(self):
        """history_frame_s=0.4 必须被拒（当前口径 0.2）。"""
        p, _ = base_manifest(self.tmp, schema={"history_frame_s": 0.4});
        rc, out = run_preflight(p)
        self.assertNotEqual(rc, 0)
        self.assertIn("history_frame_s", out)

    def test_wrong_omega_unit_rejected(self):
        """omega_unit=deg_s 必须被拒（实现是 rad/s）。"""
        p, _ = base_manifest(self.tmp, schema={"omega_unit": "deg_s"});
        rc, out = run_preflight(p)
        self.assertNotEqual(rc, 0)
        self.assertIn("omega_unit", out)

    def test_missing_step_s_rejected(self):
        """step_s 不符（例如 0.05）必须被拒。"""
        p, _ = base_manifest(self.tmp, schema={"step_s": 0.05});
        rc, out = run_preflight(p)
        self.assertNotEqual(rc, 0)

    def test_hash_mismatch_rejected(self):
        """head SHA256 对不上必须被拒（防止 checkpoint 被换掉）。"""
        p, man = base_manifest(self.tmp)
        man["files"]["head"]["sha256"] = "0" * 64
        p.write_text(json.dumps(man), encoding="utf-8")
        rc, out = run_preflight(p)
        self.assertNotEqual(rc, 0, "hash 不匹配竟然通过")
        self.assertIn("SHA256", out)

    def test_missing_file_rejected(self):
        """文件不存在必须被拒。"""
        p, man = base_manifest(self.tmp)
        man["files"]["adapter"]["path"] = str(self.tmp / "nope.safetensors")
        p.write_text(json.dumps(man), encoding="utf-8")
        rc, out = run_preflight(p)
        self.assertNotEqual(rc, 0)

    def test_bad_w_max_rejected(self):
        """w_max<=0 必须被拒（否则 omega_to_turn 恒 0，转向彻底失效）。"""
        p, _ = base_manifest(self.tmp,
                             calibration={"v_max_mps": 1.2, "dv_max_mps2": 0.5,
                                           "w_max_rad_s": 0.0, "dw_max_rad_s2": 3.14})
        rc, out = run_preflight(p)
        self.assertNotEqual(rc, 0, "w_max=0 竟然通过")

    def test_missing_wp_stop_semantics_rejected(self):
        """缺 wp_stop_semantics 必须被拒（2026-11-05 审计要求）。

        动机：state!=localized 时 service 兜底 wp=(0,0)，act() 判成到点 →
        **整块归零**，模型完全不被执行，且与「已到终点」不可区分。
        这必须是一个被显式确认的部署语义，而不是踩到才知道。
        """
        p, man = base_manifest(self.tmp)
        man["calibration"].pop("wp_stop_semantics", None)
        p.write_text(json.dumps(man), encoding="utf-8")
        rc, out = run_preflight(p)
        self.assertNotEqual(rc, 0, "缺 wp_stop_semantics 竟然通过")
        self.assertIn("wp_stop_semantics", out)

    def test_missing_stale_rejected(self):
        """缺 VRC_POLICY_STALE 必须被拒（缺省会退化成不可控的丢块窗口）。"""
        p, _ = base_manifest(self.tmp, runtime_env={});
        rc, out = run_preflight(p)
        self.assertNotEqual(rc, 0)

    def test_missing_drop_wp_rejected(self):
        """schema 缺 drop_wp 字段必须被拒（缺失会静默退化成 False，换口径）。"""
        p, man = base_manifest(self.tmp)
        man["schema"].pop("drop_wp", None)
        p.write_text(json.dumps(man), encoding="utf-8")
        rc, out = run_preflight(p)
        self.assertNotEqual(rc, 0)


class PreflightAcceptsGoodConfigTests(unittest.TestCase):
    def test_real_b3_manifest_passes(self):
        """真实 B3 manifest 必须通过（若这条挂了说明部署配置本身坏了）。"""
        man = ROOT / "deploy" / "b3_peak_norm_3ep.deploy.json"
        if not man.exists():
            self.skipTest("deploy manifest 不存在")
        rc, out = run_preflight(man)
        self.assertEqual(rc, 0, "真实 manifest 未通过 preflight:\n%s" % out[-2000:])


if __name__ == "__main__":
    unittest.main()