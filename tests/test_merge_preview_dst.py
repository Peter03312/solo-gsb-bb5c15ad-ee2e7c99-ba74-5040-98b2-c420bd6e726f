"""外部刷卡批次合并预览：时刻差口径回归测试（服务时区采用夏令时）。

修复的缺陷：候选配对用 ``datetime.timestamp()`` 求差——它把 naive 记录
时刻按**服务本地时区**解释，而返回的 ``abs_diff_seconds`` /
``delta_seconds`` 用记录时刻直接相减。服务部署在采用夏令时的时区
（如 America/New_York）时两套口径相互矛盾：

- 春季拨快日 2026-03-08：01:59:30 与 03:00:00 的记录时刻相差
  60 分 30 秒（不存在的 02 点被跳过），时间戳只差 30 秒，
  两条记录在 120 秒容差内被**误配**为重复，返回的绝对差却是 3630；
- 秋季拨回日 2026-11-01：01:59:30 与 02:00:00 的记录时刻相差
  30 秒（02 点第二次出现），时间戳相差 3630 秒，本该配对却被**漏配**。

修复后候选筛选、全局一对一匹配费用、返回的绝对及有符号秒差统一为
与输入文本一致的**记录时刻差**（naive 时刻直接相减），结果与服务
时区 / 夏令时无关；容差按闭区间判定，先最大化配对数、再最小化总秒差、
行号破同分的既有规则不变。

纯核心层用例只依赖标准库；端到端用例依赖 FastAPI / httpx（同其他测试）。

运行：``python3 -m pytest tests/ -q``；
无 pytest 时也可直接 ``python3 tests/test_merge_preview_dst.py``
执行核心层用例（TZ 用例需 Unix ``time.tzset``）。
"""
from __future__ import annotations

import contextlib
import os
import sys
import time

# 支持直接 ``python3 tests/test_merge_preview_dst.py`` 运行（无 pytest 时）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import pytest
except ImportError:  # 允许无 pytest 的环境直接执行本文件（核心层用例）
    class _PytestShim:
        def fixture(self, *args, **kwargs):
            def wrap(fn):
                return fn
            return wrap(args[0]) if args and callable(args[0]) else wrap

    pytest = _PytestShim()

from app.core import (
    MERGE_TOLERANCE_MAX_SECONDS,
    external_merge_preview,
    parse_records_text,
)

STATION = "科技园"

SPRING_FORWARD = ("2026-03-08 01:59:30", "2026-03-08 03:00:00")
FALL_BACK = ("2026-11-01 01:59:30", "2026-11-01 02:00:00")


@contextlib.contextmanager
def timezone(name: str):
    """临时切换进程本地时区（Unix tzset），退出后恢复原环境。"""
    old_tz = os.environ.get("TZ")
    old_tzset = hasattr(time, "tzset")
    try:
        os.environ["TZ"] = name
        time.tzset()
        yield
    finally:
        if old_tz is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = old_tz
        if old_tzset:
            time.tzset()


def _merge(base_text: str, import_text: str, tolerance=120):
    base, errors = parse_records_text(base_text)
    assert errors == []
    imports, errors = parse_records_text(import_text)
    assert errors == []
    result = external_merge_preview(base, imports, tolerance)
    assert result.ok, result.errors
    return result


def _line(card: str, when: str, action: str = "进", station: str = STATION) -> str:
    return f"{card}, {when}, {station}, {action}\n"


@pytest.fixture()
def temp_db(tmp_path, monkeypatch):
    """每个用例一个独立数据库文件（db.connect 默认读取模块级 DB_PATH）。"""
    from app import db
    path = str(tmp_path / "bus-test.db")
    monkeypatch.setattr(db, "DB_PATH", path)
    db.init_db()
    return path


# ---------------------------------------------------------------------------
# 夏令时切换日：与输入文本一致的记录时刻差
# ---------------------------------------------------------------------------

def test_spring_forward_wall_gap_not_paired_under_dst():
    """纽约春季拨快：01:59:30 与 03:00:00 文本相差 3630 秒，不得误配。"""
    with timezone("America/New_York"):
        result = _merge(_line("A", SPRING_FORWARD[0]),
                        _line("A", SPRING_FORWARD[1]), 120)
    assert result.pair_count == 0
    assert result.total_abs_diff_seconds == 0
    assert len(result.base_only) == 1
    assert len(result.import_only) == 1
    # 未配对时合并文本保留两侧，按记录时刻排序（早者在前）
    assert result.merged_text == (
        f"A, {SPRING_FORWARD[0]}, {STATION}, 进\n"
        f"A, {SPRING_FORWARD[1]}, {STATION}, 进")


def test_fall_back_wall_gap_paired_under_dst():
    """纽约秋季拨回：01:59:30 与 02:00:00 文本相差 30 秒，必须配对。"""
    with timezone("America/New_York"):
        result = _merge(_line("A", FALL_BACK[0]),
                        _line("A", FALL_BACK[1]), 120)
    assert result.pair_count == 1
    pair = result.pairs[0]
    assert pair["base_time"] == FALL_BACK[0]
    assert pair["import_time"] == FALL_BACK[1]
    # 绝对秒差与有符号秒差同为文本时刻差 30 秒（旧实现会漏配）
    assert pair["abs_diff_seconds"] == 30
    assert pair["delta_seconds"] == 30
    assert result.total_abs_diff_seconds == 30
    assert len(result.base_only) == 0
    assert len(result.import_only) == 0
    # 配对取基准行：合并文本只剩基准时刻，不含导入侧的 02:00:00
    assert result.merged_text == f"A, {FALL_BACK[0]}, {STATION}, 进"


def test_dst_result_independent_of_service_timezone():
    """同一请求在 UTC / 纽约 / 上海 / 圣保罗 / 奥克兰结果完全一致。"""
    cases = [
        (_line("A", SPRING_FORWARD[0]), _line("A", SPRING_FORWARD[1]), 120),
        (_line("A", FALL_BACK[0]), _line("A", FALL_BACK[1]), 120),
        # 跨拨回窗口的多记录分组（稠密图，走匈牙利分支）
        ("A, 2026-11-01 01:59:30, 科技园, 进\n"
         "A, 2026-11-01 02:00:30, 科技园, 进\n",
         "A, 2026-11-01 02:00:00, 科技园, 进\n"
         "A, 2026-11-01 02:00:40, 科技园, 进\n", 120),
        # 普通日期
        (_line("A", "2026-09-18 08:00:00"),
         _line("A", "2026-09-18 08:00:03"), 30),
    ]
    snapshots: dict[str, list] = {}
    for tz in ("UTC", "America/New_York", "Asia/Shanghai",
               "America/Sao_Paulo", "Pacific/Auckland"):
        with timezone(tz):
            rows = []
            for bt, it, tol in cases:
                r = _merge(bt, it, tol)
                rows.append([
                    (p["base_time"], p["import_time"],
                     p["abs_diff_seconds"], p["delta_seconds"])
                    for p in r.pairs] + [
                    r.pair_count, r.total_abs_diff_seconds, r.merged_text])
            snapshots[tz] = rows
    reference = snapshots["UTC"]
    for tz, rows in snapshots.items():
        assert rows == reference, f"{tz} 与 UTC 的合并结果不一致"


# ---------------------------------------------------------------------------
# 容差闭区间边界与普通日期
# ---------------------------------------------------------------------------

def test_tolerance_closed_interval():
    """时间差恰等于容差仍算候选（闭区间）；多 1 秒即不配对。"""
    bt, it = (_line("A", "2026-03-08 03:30:00"),
              _line("A", "2026-03-08 03:30:30"))
    assert _merge(bt, it, 30).pair_count == 1
    assert _merge(bt, it, 29).pair_count == 0
    # 容差 0：必须文本时刻完全相同
    assert _merge(bt, it, 0).pair_count == 0
    assert _merge(bt, bt, 0).pair_count == 1


def test_signed_diff_negative_when_import_earlier():
    """有符号秒差 = 导入 − 基准；导入更早时为负，绝对差取正。"""
    result = _merge(_line("A", "2026-11-01 02:00:00"),
                    _line("A", "2026-11-01 01:59:30"), 120)
    pair = result.pairs[0]
    assert pair["delta_seconds"] == -30
    assert pair["abs_diff_seconds"] == 30
    assert result.total_abs_diff_seconds == 30


def test_card_station_action_must_match():
    """卡号 / 站点 / 方向不同，即使时刻相同也不配对（既有规则不变）。"""
    result = _merge(
        "A, 2026-11-01 01:59:30, 科技园, 进\n",
        "A, 2026-11-01 01:59:30, 中心公园, 进\n"
        "B, 2026-11-01 01:59:30, 科技园, 进\n"
        "A, 2026-11-01 01:59:30, 科技园, 出\n", 120)
    assert result.pair_count == 0
    assert len(result.import_only) == 3


# ---------------------------------------------------------------------------
# 全局一对一匹配：先最大化配对数，再最小化总秒差
# ---------------------------------------------------------------------------

def test_max_cardinality_then_min_total_diff_across_fallback():
    """跨秋季拨回窗口：贪心会少配 / 错配，全局匹配仍取最多配对与最小总差。

    基准 01:59:30 / 02:00:30，导入 02:00:00 / 02:00:40（容差 120）：
    两条边均为候选，最优为 (01:59:30↔02:00:00=30) +
    (02:00:30↔02:00:40=10)，总差 40；错误交叉配对总差为 100。
    """
    result = _merge(
        "A, 2026-11-01 01:59:30, 科技园, 进\n"
        "A, 2026-11-01 02:00:30, 科技园, 进\n",
        "A, 2026-11-01 02:00:00, 科技园, 进\n"
        "A, 2026-11-01 02:00:40, 科技园, 进\n", 120)
    assert result.pair_count == 2
    diffs = sorted(p["abs_diff_seconds"] for p in result.pairs)
    assert diffs == [10, 30]
    assert result.total_abs_diff_seconds == 40


def test_spring_forward_pair_explored_but_superseded():
    """候选边费用使用同一套秒差：跨拨缺小时的误候选不再进入匹配。

    03:00:00 在文本口径下距 01:59:30 为 3630 秒（超容差），但距
    03:00:10 仅 10 秒；旧实现时间戳口径下 03:00:00 同时是前两者的
    候选，可能抢占本应配对 03:00:10 的基准。
    """
    result = _merge(
        "A, 2026-03-08 01:59:30, 科技园, 进\n"
        "A, 2026-03-08 03:00:10, 科技园, 进\n",
        "A, 2026-03-08 03:00:00, 科技园, 进\n", 120)
    assert result.pair_count == 1
    pair = result.pairs[0]
    assert pair["base_time"] == "2026-03-08 03:00:10"
    assert pair["import_time"] == "2026-03-08 03:00:00"
    assert pair["abs_diff_seconds"] == 10


def test_tie_break_by_line_numbers():
    """总秒差相同的配对方案，按基准行号和、再按导入行号和破同分。"""
    result = _merge(
        "A, 2026-09-18 08:00:00, 科技园, 进\n"
        "A, 2026-09-18 08:01:00, 科技园, 进\n",
        "A, 2026-09-18 08:00:05, 科技园, 进\n"
        "A, 2026-09-18 08:01:05, 科技园, 进\n", 120)
    assert result.pair_count == 2
    chosen = {(p["base"]["line"], p["import"]["line"]) for p in result.pairs}
    assert chosen == {(1, 1), (2, 2)}  # 行号和更小的同序方案


def test_invalid_tolerance_rejected():
    """容差超范围 / 非整数仍返回 CONFIG_BAD（既有行为不变）。"""
    base, _ = parse_records_text(_line("A", "2026-09-18 08:00:00"))
    imports, _ = parse_records_text(_line("A", "2026-09-18 08:00:00"))
    for bad in (-1, MERGE_TOLERANCE_MAX_SECONDS + 1, 1.5, "30", None):
        result = external_merge_preview(base, imports, bad)
        assert result.ok is False
        assert result.errors[0]["code"] == "CONFIG_BAD"


# ---------------------------------------------------------------------------
# 端到端：HTTP API 字段与只读行为保持兼容
# ---------------------------------------------------------------------------

def test_http_endpoint_dst_semantics_and_fields(temp_db):
    """HTTP 层在夏令时时区下仍按文本时刻差判定，响应字段保持不变。"""
    from fastapi.testclient import TestClient
    from app import db, main
    from app.schemas import TextIn

    client = TestClient(main.app)
    # 基准：保存 2026-11-01 01:59:30；待导入：02:00:00（文本差 30 秒）
    main.put_records(TextIn(text=_line("A", FALL_BACK[0])))
    with timezone("America/New_York"):
        resp = client.post(
            "/api/external-merge-preview",
            json={"text": _line("A", FALL_BACK[1]),
                  "tolerance_seconds": 120})
    assert resp.status_code == 200
    data = resp.json()
    # 既有响应字段全部保留
    assert set(data) == {
        "ok", "tolerance_seconds", "base_count", "import_count",
        "pair_count", "base_only_count", "import_only_count",
        "merged_count", "total_abs_diff_seconds", "pairs",
        "base_only", "import_only", "merged_text",
    }
    assert data["pair_count"] == 1
    assert data["total_abs_diff_seconds"] == 30
    pair = data["pairs"][0]
    assert pair["abs_diff_seconds"] == 30
    assert pair["delta_seconds"] == 30
    assert pair["base"]["source"] == "base"
    assert pair["import"]["source"] == "import"
    assert data["merged_count"] == data["base_count"] == 1

    # 只读：待导入记录未入库，已保存记录仍只有基准一条
    saved = client.get("/api/records").json()["items"]
    assert len(saved) == 1
    assert saved[0]["time"] == FALL_BACK[0]


# ---------------------------------------------------------------------------
# 无 pytest 时的直接运行入口（核心层用例）
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    core_tests = [v for k, v in sorted(globals().items())
                  if k.startswith("test_") and k != "test_http_endpoint_dst_semantics_and_fields"]
    for fn in core_tests:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(core_tests)} core tests passed")
