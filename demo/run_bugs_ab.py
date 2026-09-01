"""A/B en bugworld: el mismo agente mantenedor con y sin subcortex sobre los mismos bugs."""
from __future__ import annotations

import argparse
import asyncio
import shutil
from pathlib import Path

from bugworld.mutate import load_bugs
from bugworld.tools import registry
from bugworld.world import BugWorld
from demo.ab import load_baseline, print_table, run_episodes, save, summarize
from demo.bug_agent import build_app

RUN_ROOT = Path(".bugworld/run")


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bugs", default=str(Path("bugworld/bugs.json")))
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--consolidate-every", type=int, default=10)
    ap.add_argument("--only", choices=["baseline", "subcortex"], default=None)
    ap.add_argument("--out", default="results-bugs.json")
    ap.add_argument("--baseline-from", default=None)
    args = ap.parse_args()
    bugs = load_bugs(Path(args.bugs))[:args.n]
    results, summaries = {}, {}
    if args.baseline_from:
        results["baseline"] = load_baseline(args.baseline_from, len(bugs))
        summaries["baseline"] = summarize(results["baseline"])
    for name, flag in (("baseline", False), ("subcortex", True)):
        if (args.only and args.only != name) or name in results:
            continue
        registry.clear()
        root = RUN_ROOT / name
        shutil.rmtree(root, ignore_errors=True)
        app, sc = build_app(flag)
        rows = await run_episodes(
            name, bugs, app=app, sc=sc, registry=registry, app_name="bugworld",
            make_world=lambda bug, i, root=root: BugWorld(bug, root),
            features=lambda w: w.features(), prompt=lambda w: w.intro(), label=lambda bug: bug.kind,
            extra=lambda w: {"edits": w.edits, "rewrites": w.rewrites, "file": w.bug.file,
                             "description": w.bug.description},
            with_subcortex=flag, consolidate_every=args.consolidate_every if flag else 0)
        results[name] = rows
        summaries[name] = summarize(rows)
        save(args.out, results, summaries)
    print_table(summaries)
    print(f"\nGuardado en {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
