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
    """从日志中提取 request_body JSON。

    支持两种日志格式：
    1. sensitive-logs 模式：request_body={"conversationState":...}（可能被截断）
    2. 普通模式：kiro_request_body_bytes=135777（无 JSON 内容）

    对于截断的 JSON，尝试用 raw_decode 解析到第一个完整 JSON 对象。
    同时排除 response_body 的误匹配。
    """
    decoder = json.JSONDecoder()
    # 两种来源：
    # 1. handler DEBUG: "Kiro request body: {json}"（发送前，完整内容）
    # 2. provider ERROR: "request_body={json}"（400 后，可能被截断）
    kiro_body_marker = "Kiro request body: "
    body_re = re.compile(r"(?<![a-z_])request_body=")

    for line_no, line in enumerate(log_text.splitlines(), 1):
        clean = strip_ansi(line)

        # 来源 1: handler 的 DEBUG 日志（优先，内容更完整）
        kiro_idx = clean.find(kiro_body_marker)
        if kiro_idx != -1:
            brace = clean.find("{", kiro_idx + len(kiro_body_marker))
            if brace == -1:
                continue
        elif "request_body=" in clean:
            # 来源 2: provider 的 ERROR 日志
            match = body_re.search(clean)
            if not match:
                continue
            brace = clean.find("{", match.end())
            if brace == -1:
                continue
        else:
            continue

        # 使用 raw_decode 解析第一个完整 JSON 对象（忽略行尾其他 tracing 字段）
        try:
            body, _ = decoder.raw_decode(clean, brace)
        except json.JSONDecodeError:
            # JSON 被截断（sensitive-logs 的 truncate_middle），尝试提取可用信息
            body_str = clean[brace:]
            yield line_no, _partial_parse_request_body(body_str, line_no)
            continue

        if isinstance(body, dict):
            yield line_no, body


def _partial_parse_request_body(truncated_json: str, line_no: int) -> Dict[str, Any]:
    """从截断的 JSON 中尽量提取结构信息。

    即使 JSON 不完整，也能通过正则提取 conversationId、工具数量等关键字段，
    用于启发式诊断。
    """
    info: Dict[str, Any] = {"_partial": True, "_raw_len": len(truncated_json)}

    # 提取 conversationId
    m = re.search(r'"conversationId"\s*:\s*"([^"]+)"', truncated_json)
    if m:
        info["_conversationId"] = m.group(1)

    # 统计 toolUseId 出现次数（近似 tool_use 数量）
    info["_toolUseId_count"] = len(re.findall(r'"toolUseId"', truncated_json))

    # 统计 toolSpecification 出现次数（近似 tool 定义数量）
    info["_toolSpec_count"] = len(re.findall(r'"toolSpecification"', truncated_json))

    # 统计 assistantResponseMessage 出现次数（近似 history 轮数）
    info["_assistant_msg_count"] = len(re.findall(r'"assistantResponseMessage"', truncated_json))

    # 统计 userInputMessage 出现次数
    info["_user_msg_count"] = len(re.findall(r'"userInputMessage"', truncated_json))

    return info


def _get(d: Dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = d
    for part in path.split("."):
        if not isinstance(cur, dict):
            return default
        cur = cur.get(part)
    return cur if cur is not None else default


def summarize(body: Dict[str, Any], line_no: int) -> RequestSummary:
    # 处理 partial 解析的情况
    if body.get("_partial"):
        return RequestSummary(
            line_no=line_no,
            conversation_id=body.get("_conversationId"),
            content_len=-1,
            tools_n=body.get("_toolSpec_count", -1),
            tool_results_n=body.get("_toolUseId_count", -1),
            history_n=body.get("_assistant_msg_count", -1),
            json_len=body.get("_raw_len", 0),
        )

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
    # partial 解析的请求只能做有限诊断
    if body.get("_partial"):
        issues: List[str] = ["W_TRUNCATED_LOG"]
        raw_len = body.get("_raw_len", 0)
        if raw_len > large_payload_bytes:
            issues.append("W_PAYLOAD_LARGE")
        return issues

    issues = []

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
    parser = argparse.ArgumentParser(
        description="离线诊断 Improperly formed request（上游 400）常见成因"
    )
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

    # 先扫描所有 400 Improperly formed request 的 ERROR 行，提取上下文
    print("=" * 60)
    print("Phase 1: 扫描 400 Improperly formed request 错误")
    print("=" * 60)
    error_lines = _scan_400_errors(log_text)
    if error_lines:
        for el in error_lines:
            print(f"\n  [line {el['line_no']}] bytes={el.get('body_bytes', '?')} "
                  f"url={el.get('url', '?')}")
            if "_req_body_line" in el:
                print(f"    ↳ 关联请求体: line {el['_req_body_line']}"
                      f"{' (truncated)' if el.get('_req_body_partial') else ''}")
            if "summary" in el:
                s = el["summary"]
                print(f"    ↳ conversationId={s.conversation_id or 'None'} "
                      f"content_len={s.content_len} tools={s.tools_n} "
                      f"toolResults={s.tool_results_n} history={s.history_n} "
                      f"json_len={s.json_len}")
            if "issues" in el and el["issues"]:
                print(f"    ↳ issues: {', '.join(el['issues'])}")
            elif "_req_body" in el and el["_req_body"].get("_partial"):
                body = el["_req_body"]
                print(f"    ↳ partial: toolSpecs={body.get('_toolSpec_count', '?')} "
                      f"toolUseIds={body.get('_toolUseId_count', '?')} "
                      f"assistantMsgs={body.get('_assistant_msg_count', '?')} "
                      f"userMsgs={body.get('_user_msg_count', '?')} "
                      f"raw_len={body.get('_raw_len', '?')}")
    else:
        print("  未发现 400 Improperly formed request 错误")
    print()

    # 再扫描 request_body 条目
    print("=" * 60)
    print("Phase 2: 解析 request_body 条目")
    print("=" * 60)

    issue_counter: Counter[str] = Counter()
    issues_to_samples: Dict[str, List[RequestSummary]] = defaultdict(list)
    total = 0
    partial_count = 0

    for line_no, body in iter_request_bodies(log_text):
        total += 1
        if body.get("_partial"):
            partial_count += 1
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

    print(f"Parsed request_body entries: {total} (complete: {total - partial_count}, truncated: {partial_count})")
    print("")

    if not issue_counter:
        print("No request_body entries found.")
        if not error_lines:
            print("\nHint: 如果使用非 sensitive-logs 模式，日志中不包含 request_body 内容。")
            print("      请使用 --features sensitive-logs 重新编译，或检查 kiro_request_body_bytes 字段。")
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


def _scan_400_errors(log_text: str) -> List[Dict[str, Any]]:
    """扫描日志中的 400 Improperly formed request 错误行，关联最近的请求体。

    对每个 400 错误，向上查找最近的 'Kiro request body:' DEBUG 行，
    解析其中的请求体并做启发式诊断。
    """
    lines = log_text.splitlines()
    results = []
    body_bytes_re = re.compile(r"(?:kiro_)?request_body_bytes=(\d+)")
    url_re = re.compile(r"request_url=(\S+)")
    decoder = json.JSONDecoder()

    for line_no_0, line in enumerate(lines):
        if "Improperly formed request" not in line:
            continue
        clean = strip_ansi(line)
        entry: Dict[str, Any] = {"line_no": line_no_0 + 1}

        m = body_bytes_re.search(clean)
        if m:
            entry["body_bytes"] = int(m.group(1))

        m = url_re.search(clean)
        if m:
            entry["url"] = m.group(1)

        # 向上查找最近的 "Kiro request body:" 行（最多回溯 20 行）
        req_body = None
        for back in range(1, min(21, line_no_0 + 1)):
            prev_line = strip_ansi(lines[line_no_0 - back])
            marker = "Kiro request body: "
            idx = prev_line.find(marker)
            if idx == -1:
                continue
            brace = prev_line.find("{", idx + len(marker))
            if brace == -1:
                break
            try:
                req_body, _ = decoder.raw_decode(prev_line, brace)
            except json.JSONDecodeError:
                # 截断的 JSON，做 partial 解析
                entry["_req_body_partial"] = True
                req_body = _partial_parse_request_body(prev_line[brace:], line_no_0 - back + 1)
            entry["_req_body_line"] = line_no_0 - back + 1
            break

        if req_body and isinstance(req_body, dict):
            entry["_req_body"] = req_body
            if not req_body.get("_partial"):
                # 完整 JSON，做深度诊断
                issues = find_issues(
                    req_body,
                    max_history_messages=100,
                    large_payload_bytes=80_000,
                    huge_payload_bytes=200_000,
                )
                entry["issues"] = issues
                summary = summarize(req_body, entry.get("_req_body_line", 0))
                entry["summary"] = summary

        results.append(entry)

    return results


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
