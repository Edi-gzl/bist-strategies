"""Pipeline progress tracking for F_intraday (orchestrator + child scripts)."""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from shared import configs as CFG

STATUS_JSON = CFG.F_INTRADAY_SUMMARY / 'pipeline_status.json'
STATUS_MD = CFG.F_INTRADAY_SUMMARY / 'pipeline_status.md'


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec='seconds')


def _elapsed_str(seconds: float) -> str:
    s = int(max(0, seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f'{h}h {m:02d}m {sec:02d}s'
    if m:
        return f'{m}m {sec:02d}s'
    return f'{sec}s'


def write_status(payload: dict[str, Any]) -> None:
    """Atomic-ish status write for manual audit / tail -f."""
    CFG.F_INTRADAY_SUMMARY.mkdir(parents=True, exist_ok=True)
    payload['updated_at'] = _now_iso()
    STATUS_JSON.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    lines = [
        '# F_intraday pipeline status',
        '',
        f'**Updated:** {payload["updated_at"]}',
        '',
        f'**Pipeline:** {payload.get("pipeline", "F_intraday")}',
        f'**Phase:** {payload.get("phase", "—")}',
        f'**Step:** {payload.get("step_index", "?")} / {payload.get("step_total", "?")} — `{payload.get("current_step", "—")}`',
        '',
        f'**Detail:** {payload.get("detail", "—")}',
        f'**Elapsed (pipeline):** {payload.get("elapsed_pipeline", "—")}',
        f'**Elapsed (step):** {payload.get("elapsed_step", "—")}',
        '',
        '## Completed',
        '',
    ]
    for s in payload.get('steps_completed', []):
        lines.append(f'- {s}')
    if payload.get('next_step'):
        lines += ['', '## Next', '', f'- {payload["next_step"]}']
    STATUS_MD.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def touch_child_step(script: str, detail: str = '', phase: str = 'running') -> None:
    """Child scripts call this to merge into status file."""
    base: dict[str, Any] = {
        'pipeline': 'F_intraday',
        'phase': phase,
        'current_step': script,
        'detail': detail,
    }
    if STATUS_JSON.exists():
        try:
            base = {**json.loads(STATUS_JSON.read_text(encoding='utf-8')), **base}
        except json.JSONDecodeError:
            pass
    step_start = os.environ.get('ORCH_STEP_START')
    if step_start:
        try:
            base['elapsed_step'] = _elapsed_str(time.time() - float(step_start))
        except ValueError:
            pass
    orch_start = os.environ.get('ORCH_START')
    if orch_start:
        try:
            base['elapsed_pipeline'] = _elapsed_str(time.time() - float(orch_start))
        except ValueError:
            pass
    if os.environ.get('ORCH_STEP_INDEX'):
        base['step_index'] = os.environ.get('ORCH_STEP_INDEX')
        base['step_total'] = os.environ.get('ORCH_STEP_TOTAL')
    write_status(base)


class PipelineTracker:
    """Orchestrator step runner with live status file."""

    def __init__(self, steps: list[tuple[str, str]], *, label: str = 'F_intraday'):
        self.steps = steps  # (script_name, description)
        self.label = label
        self.t0 = time.time()
        self.completed: list[str] = []
        self._step_t0 = self.t0

    def banner(self, msg: str) -> None:
        elapsed = _elapsed_str(time.time() - self.t0)
        print(f'\n{"=" * 78}')
        print(f'[{elapsed}] {msg}')
        print('=' * 78, flush=True)

    def start_step(self, index: int, script: str, description: str, extra: str = '') -> None:
        self._step_t0 = time.time()
        detail = description + (f' | {extra}' if extra else '')
        n = len(self.steps)
        self.banner(f'STEP {index}/{n}: {script} — {description}')
        print(f'  Status file: {STATUS_JSON}', flush=True)
        next_s = self.steps[index][0] if index < n else None
        write_status({
            'pipeline': self.label,
            'phase': 'running',
            'step_index': index,
            'step_total': n,
            'current_step': script,
            'detail': detail,
            'elapsed_pipeline': _elapsed_str(time.time() - self.t0),
            'elapsed_step': '0s',
            'steps_completed': self.completed.copy(),
            'next_step': next_s,
        })

    def end_step(self, script: str, ok: bool = True) -> None:
        dt = _elapsed_str(time.time() - self._step_t0)
        mark = 'OK' if ok else 'FAIL'
        self.completed.append(f'{script} ({dt}) [{mark}]')
        print(f'  >> Finished {script} in {dt} [{mark}]', flush=True)

    def finish(self, message: str = 'Pipeline complete') -> None:
        write_status({
            'pipeline': self.label,
            'phase': 'done',
            'step_index': len(self.steps),
            'step_total': len(self.steps),
            'current_step': '(none)',
            'detail': message,
            'elapsed_pipeline': _elapsed_str(time.time() - self.t0),
            'elapsed_step': '—',
            'steps_completed': self.completed,
            'next_step': None,
        })
        self.banner(message)
