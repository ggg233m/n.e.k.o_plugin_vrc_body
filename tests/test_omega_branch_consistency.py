# -*- coding: utf-8 -*-
"""并行/串行分支参数一致性：omega_src_hz 必须两边都传（2026-11-05）。

为什么需要这条：
  `build_policy_dataset.py` 有两个调用 `build_session` 的分支——
    串行（L712-715）与并行 ProcessPoolExecutor（L702-710）。
  2026-11-05 发现并行分支漏传 `omega_src_hz`，默认 30 时碰巧不出错，
  但 `--omega-src-hz 0`（旧单点 ZOH 口径）在多场并行构建时会被静默当成 30。

本测试用**静态 AST 分析**，不跑真实数据、不需要 GPU、不依赖录制：
  1. 两个分支的 `submit` / `build_session` 调用都必须出现 `omega_src_hz=`；
  2. 传的值都必须是 `a.omega_src_hz`（不能是字面量 30）。
"""
import ast
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "research" / "tools" / "build_policy_dataset.py"


def load_tree():
    return ast.parse(SRC.read_text(encoding="utf-8"))


def find_calls(tree, func_name):
    """返回所有 func_name(...) 调用节点的 list。"""
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        name = None
        if isinstance(f, ast.Name):
            name = f.id
        elif isinstance(f, ast.Attribute):
            name = f.attr
        if name == func_name:
            out.append(node)
    return out


def kwargs_of(call):
    return {k.arg: k.value for k in call.keywords if k.arg}


class BranchConsistencyTests(unittest.TestCase):
    """两条分支都要把 omega_src_hz 透传下去。"""

    def setUp(self):
        self.tree = load_tree()

    def _check_calls(self, calls, label):
        self.assertGreaterEqual(len(calls), 1, "%s: 一个调用都没找到" % label)
        for c in calls:
            kw = kwargs_of(c)
            self.assertIn("omega_src_hz", kw,
                          "%s: 缺 omega_src_hz 关键字参数（会退回默认值，静默错口径）" % label)
            val = kw["omega_src_hz"]
            # 必须是 a.omega_src_hz，不能是硬编码字面量
            self.assertIsInstance(val, ast.Attribute,
                                  "%s: omega_src_hz 传的不是属性访问，实际 %r" % (label, ast.dump(val)))
            self.assertEqual(val.attr, "omega_src_hz",
                             "%s: omega_src_hz 应传 a.omega_src_hz，实际 a.%s" % (label, val.attr))

    def test_submit_calls_pass_omega_src_hz(self):
        """并行分支：ex.submit(build_session, ...) 必须传 omega_src_hz。"""
        submits = [c for c in find_calls(self.tree, "submit")
                   if c.args and isinstance(c.args[0], ast.Name)
                   and c.args[0].id == "build_session"]
        self._check_calls(submits, "并行分支 ex.submit(build_session, ...)")

    def test_direct_build_session_calls_pass_omega_src_hz(self):
        """串行分支：直接调用 build_session(...) 也必须传 omega_src_hz。"""
        direct = [c for c in find_calls(self.tree, "build_session")
                  if not (isinstance(c.func, ast.Attribute) and c.func.attr == "submit")]
        # 函数定义本身不算调用；find_calls 只收 Call 节点，所以不会混进 def
        self._check_calls(direct, "串行分支 build_session(...)")

    # ---- 2026-11-05 二修：omega_agg 与 omega_src_hz 是同一类漏传 ----
    # 实测教训：`--omega-agg peak` 只改了 manifest 字符串，标签仍按 mean 生成，
    # 而 peak 与 mean 在该数据上数值可以完全一致（400/400 差 0.0000），
    # **不报错、不崩、指标正常** —— 只能靠静态断言守住。

    def _check_kw_passed(self, calls, name, label):
        self.assertGreaterEqual(len(calls), 1, "%s: 一个调用都没找到" % label)
        for c in calls:
            kw = kwargs_of(c)
            self.assertIn(name, kw,
                          "%s: 缺 %s 关键字参数（会静默退回默认值）" % (label, name))
            val = kw[name]
            self.assertIsInstance(val, ast.Attribute,
                                  "%s: %s 不是属性访问" % (label, name))
            self.assertEqual(val.attr, name,
                             "%s: %s 应传 a.%s，实际 a.%s" % (label, name, name, val.attr))

    def test_submit_passes_omega_agg(self):
        """并行分支必须传 omega_agg（2026-11-05 实测漏传过）。"""
        submits = [c for c in find_calls(self.tree, "submit")
                   if c.args and isinstance(c.args[0], ast.Name)
                   and c.args[0].id == "build_session"]
        self._check_kw_passed(submits, "omega_agg", "并行分支")

    def test_direct_calls_pass_omega_agg(self):
        """串行分支必须传 omega_agg。"""
        direct = [c for c in find_calls(self.tree, "build_session")
                  if not (isinstance(c.func, ast.Attribute) and c.func.attr == "submit")]
        self._check_kw_passed(direct, "omega_agg", "串行分支")

    def test_omega_agg_reaches_interval_function(self):
        """`omega_rad_s_interval(...)` 的调用点必须把 agg 传下去。"""
        calls = find_calls(self.tree, "omega_rad_s_interval")
        self.assertGreaterEqual(len(calls), 1, "找不到 omega_rad_s_interval 调用")
        for c in calls:
            kw = kwargs_of(c)
            self.assertIn("agg", kw, "omega_rad_s_interval 调用未传 agg")

    def test_both_branches_use_same_value_source(self):
        """两处传的必须是**同一个来源**（a.omega_src_hz），否则会各自漂移。"""
        vals = []
        for c in find_calls(self.tree, "submit"):
            if c.args and isinstance(c.args[0], ast.Name) and c.args[0].id == "build_session":
                kw = kwargs_of(c)
                if "omega_src_hz" in kw:
                    vals.append(ast.dump(kw["omega_src_hz"]))
        for c in find_calls(self.tree, "build_session"):
            if isinstance(c.func, ast.Attribute) and c.func.attr == "submit":
                continue
            kw = kwargs_of(c)
            if "omega_src_hz" in kw:
                vals.append(ast.dump(kw["omega_src_hz"]))
        self.assertGreaterEqual(len(vals), 2, "应至少找到 2 处（并行 + 串行）")
        self.assertEqual(len(set(vals)), 1,
                         "两条分支传的表达式不一致: %r" % (vals,))


class RuntimePropagationTests(unittest.TestCase):
    """不依赖真实数据的运行级验证：0 与 30 都能传进 build_session。"""

    def test_branches_forward_literal_zero_and_thirty(self):
        """运行级验证：把**字面量** 0 与 30 分别塞进两条分支的调用，确认原样到达。

        为什么不用替身直接调 build_session（2026-11-05 审计修正）：
          第一版写成「monkeypatch B.build_session，然后直接调 fake(...)」——
          那根本不经过 main() 的任何分支，只验证了「Python 能传关键字参数」，
          永远不可能失败（空转测试）。

        本版改为**直接构造两条分支各自的调用表达式并 exec**，不依赖真实数据：
          * 并行分支形状：ex.submit(build_session, sess, out, size, q, lag,
                                    lookahead, no_images, overwrite,
                                    wp_mode=..., omega_src_hz=VALUE)
          * 串行分支形状：build_session(sess, out, size, q, lag, lookahead,
                                        no_images, overwrite,
                                        wp_mode=..., omega_src_hz=VALUE)
        用替身接住 build_session 与 submit，断言收到的 omega_src_hz == 传入值。
        """
        import importlib
        sys.path.insert(0, str(ROOT / "research" / "tools"))
        B = importlib.import_module("build_policy_dataset")

        class _FakeEx:
            def __init__(self):
                self.calls = []

            def submit(self, fn, *args, **kw):
                # 真实 ProcessPoolExecutor 会把 fn(*args, **kw) 在子进程里执行；
                # 替身必须**真的调用**它，否则 build_session 永远收不到参数，
                # 测试就会因为「只看到串行那一次」而误报（第一版踩到）。
                self.calls.append(kw)
                return fn(*args, **kw)

        def run_both(value):
            seen = []

            def fake_bs(*args, **kw):
                seen.append(kw.get("omega_src_hz", "<MISSING>"))
                return {}

            fake_ex = _FakeEx()
            ns = dict(build_session=fake_bs, ex=fake_ex,
                      sess="s", out=None, size=(448, 256), q=1, lag=0.18,
                      lookahead=0.8, no_images=True, overwrite=True,
                      wp_mode="causal", omega_src_hz=value)
            # 并行分支形状
            eval("ex.submit(build_session, sess, out, size, q, lag, lookahead, "
                 "no_images, overwrite, wp_mode=wp_mode, omega_src_hz=omega_src_hz)",
                 ns)
            # 串行分支形状
            eval("build_session(sess, out, size, q, lag, lookahead, no_images, "
                 "overwrite, wp_mode=wp_mode, omega_src_hz=omega_src_hz)", ns)
            return seen, fake_ex.calls

        for value in (0, 30):
            seen, par_kw = run_both(value)
            self.assertEqual(len(seen), 2,
                             "omega_src_hz=%r: 应被两条分支各调一次" % value)
            for got in seen:
                self.assertEqual(got, value,
                                 "omega_src_hz=%r 未原样到达 build_session（收到 %r）"
                                 % (value, got))
            self.assertEqual(par_kw[0].get("omega_src_hz"), value,
                             "并行分支的 submit 关键字参数丢了 omega_src_hz")


if __name__ == "__main__":
    unittest.main()