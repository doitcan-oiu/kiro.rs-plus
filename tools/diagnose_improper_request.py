#!/usr/bin/env python3
"""
离线诊断 `Improperly formed request`（上游 400）常见成因。

使用方法：
  python3 tools/diagnose_improper_request.py logs/docker.log

脚本会从日志中提取 `request_body=...{json}`，对请求做一组启发式校验并输出汇总与样本。
目标是快速定位“项目侧可修复的请求构造问题”，而不是复现上游的完整校验逻辑。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple


ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


@dataclass(frozen=True)
class RequestSummary:
    line_no: int
    conversation_id: Optional[str]
    content_len: int
    tools_n: int
    tool_results_n: int
    history_n: int
    json_len: int


def strip_ansi(s: str) -> str:
    return ANSI_RE.sub("", s)


def iter_request_bodies(log_text: str) -> Iterable[Tuple[int, Dict[str, Any]]]:
    for line_no, line in enumerate(log_text.splitlines(), 1):
        if "request_body" not in line:
            continue
        clean = strip_ansi(line)
        idx = clean.find("request_body")
        brace = clean.find("{", idx)
        if brace == -1:
            continue
        body_str = clean[brace:].strip()
        try:
            body = json.loads(body_str)
        except Exception:
            # 日志可能被截断/破坏；跳过即可
            continue
        if isinstance(body, dict):
            yield line_no, body


def _get(d: Dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = d
    for part in path.split("."):
        if not isinstance(cur, dict):
            return default
        cur = cur.get(part)
    return cur if cur is not None else default


def summarize(body: Dict[str, Any], line_no: int) -> RequestSummary:
    conversation_id = _get(body, "conversationState.conversationId")
    content = _get(body, "conversationState.currentMessage.userInputMessage.content", "")
    tools = _get(body, "conversationState.currentMessage.userInputMessage.userInputMessageContext.tools", [])
    tool_results = _get(
        body,
        "conversationState.currentMessage.userInputMessage.userInputMessageContext.toolResults",
        [],
    )
    history = _get(body, "conversationState.history", [])

    json_len = len(json.dumps(body, ensure_ascii=False, separators=(",", ":")))

    return RequestSummary(
        line_no=line_no,
        conversation_id=conversation_id if isinstance(conversation_id, str) else None,
        content_len=len(content) if isinstance(content, str) else -1,
        tools_n=len(tools) if isinstance(tools, list) else -1,
        tool_results_n=len(tool_results) if isinstance(tool_results, list) else -1,
        history_n=len(history) if isinstance(history, list) else -1,
        json_len=json_len,
    )


def find_issues(
    body: Dict[str, Any],
    *,
    max_history_messages: int,
    large_payload_bytes: int,
    huge_payload_bytes: int,
) -> List[str]:
    issues: List[str] = []

    cs = body.get("conversationState") or {}
    cm = _get(body, "conversationState.currentMessage.userInputMessage", {})
    ctx = cm.get("userInputMessageContext") or {}

    content = cm.get("content")
    images = cm.get("images") or []
    tools = ctx.get("tools") or []
    tool_results = ctx.get("toolResults") or []
    history = cs.get("history") or []

    if isinstance(content, str) and content.strip() == "":
        if images:
            issues.append("E_CONTENT_EMPTY_WITH_IMAGES")
        elif tool_results:
            issues.append("E_CONTENT_EMPTY_WITH_TOOL_RESULTS")
        else:
            issues.append("E_CONTENT_EMPTY")

    # Tool 规范检查：description/schema
    empty_desc: List[str] = []
    missing_schema: List[str] = []
    missing_type: List[str] = []
    for t in tools if isinstance(tools, list) else []:
        if not isinstance(t, dict):
            issues.append("E_TOOL_SHAPE_INVALID")
            continue
        spec = t.get("toolSpecification")
        if not isinstance(spec, dict):
            issues.append("E_TOOL_SPEC_MISSING")
            continue

        name = spec.get("name")
        name_s = name if isinstance(name, str) else "<noname>"

        desc = spec.get("description")
        if isinstance(desc, str) and desc.strip() == "":
            empty_desc.append(name_s)

        inp = spec.get("inputSchema")
        js = inp.get("json") if isinstance(inp, dict) else None
        if isinstance(js, dict):
            if "$schema" not in js:
                missing_schema.append(name_s)
            if "type" not in js:
                missing_type.append(name_s)
        else:
            issues.append("E_TOOL_INPUT_SCHEMA_NOT_OBJECT")

    if empty_desc:
        issues.append("E_TOOL_DESCRIPTION_EMPTY")
    if missing_schema:
        issues.append("W_TOOL_SCHEMA_MISSING_$SCHEMA")
    if missing_type:
        issues.append("W_TOOL_SCHEMA_MISSING_TYPE")

    # Tool result 是否能在 history 的 tool_use 里找到（启发式）
    tool_use_ids: set[str] = set()
    history_tool_result_ids: set[str] = set()
    tool_def_names_ci: set[str] = set()

    for t in tools if isinstance(tools, list) else []:
        spec = t.get("toolSpecification") if isinstance(t, dict) else None
        if isinstance(spec, dict) and isinstance(spec.get("name"), str):
            tool_def_names_ci.add(spec["name"].lower())

    for h in history if isinstance(history, list) else []:
        if not isinstance(h, dict):
            continue
        am = h.get("assistantResponseMessage")
        if not isinstance(am, dict):
            # 也可能是 user 消息（含 tool_result）
            um = h.get("userInputMessage")
            if not isinstance(um, dict):
                continue
            uctx = um.get("userInputMessageContext")
            if not isinstance(uctx, dict):
                continue
            trs = uctx.get("toolResults")
            if not isinstance(trs, list):
                continue
            for tr in trs:
                if not isinstance(tr, dict):
                    continue
                tid = tr.get("toolUseId")
                if isinstance(tid, str):
                    history_tool_result_ids.add(tid)
            continue

        tus = am.get("toolUses")
        if isinstance(tus, list):
            for tu in tus:
                if not isinstance(tu, dict):
                    continue
                tid = tu.get("toolUseId")
                if isinstance(tid, str):
                    tool_use_ids.add(tid)
                # 历史 tool_use 的 name 必须在 tools 中有定义（上游常见约束）
                nm = tu.get("name")
                if isinstance(nm, str) and tool_def_names_ci and nm.lower() not in tool_def_names_ci:
                    issues.append("E_HISTORY_TOOL_USE_NAME_NOT_IN_TOOLS")

        # 同一条历史消息可能同时包含 userInputMessage（少见，但兼容）
        um = h.get("userInputMessage")
        if isinstance(um, dict):
            uctx = um.get("userInputMessageContext")
            if isinstance(uctx, dict):
                trs = uctx.get("toolResults")
                if isinstance(trs, list):
                    for tr in trs:
                        if not isinstance(tr, dict):
                            continue
                        tid = tr.get("toolUseId")
                        if isinstance(tid, str):
                            history_tool_result_ids.add(tid)

    # history 内部的 tool_result 必须能在 history 的 tool_use 中找到（否则极易触发 400）
    if history_tool_result_ids and tool_use_ids:
        if any(tid not in tool_use_ids for tid in history_tool_result_ids):
            issues.append("E_HISTORY_TOOL_RESULT_ORPHAN")
    elif history_tool_result_ids and not tool_use_ids:
        issues.append("E_HISTORY_TOOL_RESULT_ORPHAN")

    # currentMessage 的 tool_result 必须能在 history 的 tool_use 中找到
    orphan_results = 0
    current_tool_result_ids: set[str] = set()
    if tool_use_ids and isinstance(tool_results, list):
        for tr in tool_results:
            if not isinstance(tr, dict):
                continue
            tid = tr.get("toolUseId")
            if isinstance(tid, str):
                current_tool_result_ids.add(tid)
                if tid not in tool_use_ids:
                    orphan_results += 1
    if orphan_results:
        issues.append("W_TOOL_RESULT_ORPHAN")

    # history 的 tool_use 必须在 history/currentMessage 的 tool_result 中出现（否则极易触发 400）
    all_tool_result_ids = history_tool_result_ids | current_tool_result_ids
    if tool_use_ids and all_tool_result_ids:
        if any(tid not in all_tool_result_ids for tid in tool_use_ids):
            issues.append("E_HISTORY_TOOL_USE_ORPHAN")
    elif tool_use_ids and not all_tool_result_ids:
        issues.append("E_HISTORY_TOOL_USE_ORPHAN")

    # history 过长（强启发式；日志里经常与 400 同现）
    if isinstance(history, list) and len(history) > max_history_messages:
        issues.append("W_HISTORY_TOO_LONG")

    # payload 大小（强启发式；上游可能有不透明的硬限制）
    json_len = len(json.dumps(body, ensure_ascii=False, separators=(",", ":")))
    if json_len > huge_payload_bytes:
        issues.append("W_PAYLOAD_HUGE")
    elif json_len > large_payload_bytes:
        issues.append("W_PAYLOAD_LARGE")

    return issues


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", nargs="?", default="logs/docker.log", help="docker.log 路径")
    parser.add_argument("--max-samples", type=int, default=5, help="每类问题输出样本数量")
    parser.add_argument("--dump-dir", default=None, help="可选：把 request_body JSON 按行号落盘")
    parser.add_argument("--max-history", type=int, default=100, help="history 过长阈值（启发式）")
    parser.add_argument("--large-bytes", type=int, default=400_000, help="payload 大阈值（启发式）")
    parser.add_argument("--huge-bytes", type=int, default=800_000, help="payload 巨大阈值（启发式）")
    args = parser.parse_args(argv)

    log_path = args.log
    try:
        log_text = open(log_path, "r", encoding="utf-8", errors="replace").read()
    except FileNotFoundError:
        print(f"ERROR: log file not found: {log_path}", file=sys.stderr)
        return 2

    dump_dir = args.dump_dir
    if dump_dir:
        os.makedirs(dump_dir, exist_ok=True)

    issue_counter: Counter[str] = Counter()
    issues_to_samples: Dict[str, List[RequestSummary]] = defaultdict(list)
    total = 0

    for line_no, body in iter_request_bodies(log_text):
        total += 1
        summary = summarize(body, line_no)
        issues = find_issues(
            body,
            max_history_messages=args.max_history,
            large_payload_bytes=args.large_bytes,
            huge_payload_bytes=args.huge_bytes,
        )

        # 允许 dump 以便做最小化重放/差分调试
        if dump_dir:
            out_path = os.path.join(dump_dir, f"req_line_{line_no}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(body, f, ensure_ascii=False, indent=2)

        if not issues:
            issues = ["(NO_HEURISTIC_MATCH)"]

        for issue in set(issues):
            issue_counter[issue] += 1
            if len(issues_to_samples[issue]) < args.max_samples:
                issues_to_samples[issue].append(summary)

    print(f"Parsed request_body entries: {total}")
    print("")

    if not issue_counter:
        print("No request_body entries found.")
        return 0

    print("Issue counts:")
    for issue, cnt in issue_counter.most_common():
        print(f"  {cnt:4d}  {issue}")
    print("")

    print("Samples:")
    for issue, cnt in issue_counter.most_common():
        samples = issues_to_samples.get(issue) or []
        if not samples:
            continue
        print(f"- {issue} (showing {len(samples)}/{cnt})")
        for s in samples:
            print(
                "  line={line} conversationId={cid} content_len={cl} tools={tn} toolResults={trn} history={hn} json_len={jl}".format(
                    line=s.line_no,
                    cid=s.conversation_id or "None",
                    cl=s.content_len,
                    tn=s.tools_n,
                    trn=s.tool_results_n,
                    hn=s.history_n,
                    jl=s.json_len,
                )
            )
        print("")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
