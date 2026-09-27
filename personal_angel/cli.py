"""Command line: `python -m personal_angel analyze <media> --profile pc_cpu --ask "..."`"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from .runner import run_investigation

def _print_event(event: dict[str, Any]) -> None:
    kind = event.get("type")
    if kind == "run":
        print(f"▶ run {event['run_id']} (profile {event['profile']}, llm {event['llm']})")
    elif kind == "stage":
        print(f"[{event['stage']}] {event.get('detail', '')}")
    elif kind == "hypothesis":
        h = event["hypothesis"]
        print(f"★ hypothesis: {h['statement']} (p={h['probability']:.2f})")
    elif kind == "thought":
        print(f"\n💭 Step {event['step']} — {event['thought']}\n   → {event['action']}({json.dumps(event['action_input'])[:160]})")
    elif kind == "observation":
        print(f"   ← {event['observation'][:700]}")
    elif kind == "belief":
        print(f"   belief {event['before']:.2f} → {event['after']:.2f} ({event['source']} support {event['support']:+.2f})")
    elif kind == "question":
        q = event["question"]
        print(f"   ❓ asked ({q['language']}): {q['text']}  →  {q.get('answer')}")
    elif kind == "decision":
        d = event["decision"]
        print(f"   ⚡ {d['action']}: {'EXECUTED (simulated)' if d['executed'] else d['result']}")
    elif kind == "warning":
        print(f"   ⚠ {event['detail']}")
    elif kind == "final":
        r = event["report"]
        v = r.get("verdict", {})
        print(f"\n==== VERDICT: {v.get('level', '?').upper()} — {v.get('headline', '')} ====")
        print(r["final_answer"])
        print(f"belief={r['belief']}, executed={r['executed_actions']}")
        t = r["telemetry"]
        print(f"telemetry: {t['wall_time_s']}s, model calls {t['model_calls']}, tokens {t['input_tokens']}/{t['output_tokens']}, "
              f"frames {t['frames_processed']}/{t['frames_total']} (skipped {t['frames_skipped']}), cloud-equivalent ${t['cost']['cloud_equivalent_usd']}")

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="personal-angel")
    sub = parser.add_subparsers(dest="command", required=True)
    a = sub.add_parser("analyze", help="Investigate a video/audio/image file")
    a.add_argument("media")
    a.add_argument("--ask", default="Did anything unusual or dangerous happen? Decide what to do.")
    a.add_argument("--profile", default="fixture")
    a.add_argument("--context", "--scenario", dest="context", default=None, help="optional operator note (the scene itself is inferred)")
    a.add_argument("--answer", default=None, help="scripted answer of the person for ask_user (demo/CLI)")
    a.add_argument("--run-name", default=None)
    a.add_argument("--json", action="store_true", help="print the final report JSON")
    s = sub.add_parser("serve", help="Start the web app")
    s.add_argument("--profile", default="fixture")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8600)
    d = sub.add_parser("desktop", help="Run as a desktop application (native window, local model server auto-started)")
    d.add_argument("--profile", default="pc_cpu")
    d.add_argument("--port", type=int, default=8600)
    d.add_argument("--no-window", action="store_true", help="backend only (open the URL yourself)")
    args = parser.parse_args(argv)
    if args.command == "serve":
        from .server.app import serve

        serve(args.profile, args.host, args.port)
        return 0
    if args.command == "desktop":
        from .desktop import launch

        return launch(args.profile, args.port, not args.no_window)
    provider = None
    if args.answer is not None:
        provider = lambda q: args.answer
    else:
        def provider(q):
            try:
                return input(f"\n❓ ({q.language}) {q.text}\n   your answer [{q.timeout_s:.0f}s, blank = no response]: ").strip() or None
            except EOFError:
                return None
    report = run_investigation(args.media, args.ask, args.profile, args.context, args.run_name, provider, _print_event)
    if args.json:
        print(json.dumps(report, indent=2, default=str))
    return 0

if __name__ == "__main__":
    sys.exit(main())
