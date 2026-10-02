"""Render evaluation outcomes into a Markdown report."""
from __future__ import annotations

from .metrics import TaskOutcome, summarize


def _table(headers: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


def render(outcomes: list[TaskOutcome], title: str = "智能体评测",
           compare_with: list[TaskOutcome] | None = None) -> str:
    by_method: dict[str, list[TaskOutcome]] = {}
    for o in outcomes:
        by_method.setdefault(o.method, []).append(o)

    lines = [f"# {title}", ""]

    # headline per-method table
    rows = []
    for method, outs in by_method.items():
        s = summarize(outs)
        rows.append([
            method, s["n_tasks"], s["task_success_rate"], s["mean_best_return"],
            s["mean_lin_vel_error"], s["mean_fall_rate"], s["mean_trials"],
            s["mean_tool_calls"], s["tool_success_rate"],
            s["mean_wall_seconds"], int(s["mean_tokens"]),
        ])
    lines += ["## 各方法总体指标", "",
              _table(["方法", "任务数", "成功率", "平均回报",
                      "平均速度误差", "平均摔倒率", "平均试验数",
                      "平均工具调用", "工具成功率", "平均耗时(s)",
                      "平均token"], rows), ""]

    # per-task detail
    lines += ["## 逐任务结果", ""]
    rows = []
    for o in outcomes:
        rows.append([o.method, o.task, "达标" if o.success else "未达标",
                     round(o.best_return, 1), round(o.best_lin_vel_error, 3),
                     round(o.best_fall_rate, 2), o.n_trials, o.tool_calls,
                     o.tool_errors,
                     "是" if o.fault_injected else "-",
                     "是" if o.recovered_from_fault else ("否" if o.fault_injected else "-"),
                     o.notes])
    lines += [_table(["方法", "任务", "结果", "回报", "速度误差",
                      "摔倒率", "试验数", "工具调用", "工具错误", "故障注入",
                      "已恢复", "备注"], rows), ""]

    # fault recovery
    faulted = [o for o in outcomes if o.fault_injected]
    if faulted:
        rec = sum(o.recovered_from_fault for o in faulted)
        lines += ["## 鲁棒性", "",
                  f"共 {len(faulted)} 个任务注入工具故障，成功恢复 {rec} 个。", ""]

    # comparison (used for the optimize -> re-evaluate round)
    if compare_with is not None:
        lines += ["## 与上一轮评测对比", ""]
        prev = {o.method: summarize([o]) for o in compare_with}
        cur = {o.method: summarize([o]) for o in outcomes}
        rows = []
        for method in sorted(set(prev) | set(cur)):
            p = prev.get(method, {})
            c = cur.get(method, {})
            rows.append([
                method,
                f"{p.get('task_success_rate', '-')} -> {c.get('task_success_rate', '-')}",
                f"{p.get('mean_best_return', '-')} -> {c.get('mean_best_return', '-')}",
                f"{p.get('mean_trials', '-')} -> {c.get('mean_trials', '-')}",
            ])
        lines += [_table(["方法", "成功率", "平均回报", "平均试验数"], rows), ""]

    return "\n".join(lines)
