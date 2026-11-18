"""Durable, append-only fit claims for the frozen Target-Centric study.

A claim is charged before any optimizer/PCA decomposition begins. Failed and
unfinished claims remain charged forever. Loading a journal does not authorize
re-running an interrupted optimization. This module never truncates or repairs
journals and never silently retries claims.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path

LIMITS = {'neural': 45, 'linear': 9, 'pca': 3, 'classifier': 54}
KINDS = frozenset(('neural', 'linear', 'pca'))
STATUSES = frozenset(('complete', 'failed'))


class BudgetError(RuntimeError):
    """A claim or terminal transition is disallowed."""


class LedgerCorruptionError(BudgetError):
    """The persisted journal cannot be safely replayed."""


class LedgerBusyError(BudgetError):
    """Another process owns the journal lock; no retry is performed."""


class BudgetLedger:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_path = self.path.with_name(self.path.name + '.lock')
        # Validate any pre-existing journal without resetting its state.
        with self._exclusive():
            self._read()

    @contextmanager
    def _exclusive(self):
        try:
            descriptor = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as error:
            raise LedgerBusyError(f'Ledger lock exists; inspect without automatic recovery: {self.lock_path}') from error
        try:
            os.write(descriptor, str(os.getpid()).encode('ascii'))
            os.fsync(descriptor)
            yield
        finally:
            os.close(descriptor)
            self.lock_path.unlink()

    @staticmethod
    def _counts(jobs):
        counts = {kind: sum(job['kind'] == kind for job in jobs.values()) for kind in KINDS}
        counts['classifier'] = counts['neural'] + counts['linear']
        counts['total'] = counts['classifier'] + counts['pca']
        return counts

    def _read(self):
        if not self.path.exists():
            return [], {}
        raw = self.path.read_bytes()
        if raw and not raw.endswith(b'\n'):
            raise LedgerCorruptionError('Journal has a partial last line; preserve it and investigate, never truncate')
        events, jobs = [], {}
        for number, line in enumerate(raw.splitlines(), start=1):
            try:
                record = json.loads(line.decode('utf-8'))
                if record.get('schema') != 1 or record.get('sequence') != number:
                    raise ValueError('Invalid schema or event sequence')
                key = record['job_key']
                if not isinstance(key, str) or not key.strip():
                    raise ValueError('Invalid job key')
                if record['event'] == 'claim':
                    kind = record['kind']
                    if kind not in KINDS or key in jobs:
                        raise ValueError('Invalid kind or duplicate job claim')
                    jobs[key] = {'kind': kind, 'status': 'pending', 'claim_sequence': number}
                    if any(self._counts(jobs)[kind] > limit for kind, limit in LIMITS.items()):
                        raise ValueError('Journal already exceeds frozen budget')
                elif record['event'] == 'finish':
                    if key not in jobs or jobs[key]['status'] != 'pending':
                        raise ValueError('Finish without a pending claim or duplicate finish')
                    if record['status'] not in STATUSES or not isinstance(record['details'], dict):
                        raise ValueError('Invalid terminal status/details')
                    jobs[key]['status'] = record['status']
                    jobs[key]['details'] = record['details']
                    jobs[key]['finish_sequence'] = number
                else:
                    raise ValueError('Unknown event')
            except (KeyError, ValueError, TypeError, AttributeError, UnicodeDecodeError) as error:
                raise LedgerCorruptionError(f'Invalid journal line {number}: {error}') from error
            events.append(record)
        return events, jobs

    def _append(self, record):
        # Serialize before opening, so invalid details cannot partly append.
        data = (json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(',', ':')) + '\n').encode('utf-8')
        with self.path.open('ab') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())

    @staticmethod
    def _base(event, sequence, key):
        return {'schema': 1, 'sequence': sequence, 'event': event,
                'job_key': key, 'recorded_at_utc': datetime.now(timezone.utc).isoformat()}

    def claim(self, job_key, kind):
        if not isinstance(job_key, str) or not job_key.strip():
            raise ValueError('job_key must be a nonempty canonical string')
        if kind not in KINDS:
            raise ValueError(f'kind must be one of {sorted(KINDS)}')
        with self._exclusive():
            events, jobs = self._read()
            if job_key in jobs:
                raise BudgetError(f'Job already claimed ({jobs[job_key]["status"]}); retry forbidden: {job_key}')
            counts = self._counts(jobs)
            if counts[kind] >= LIMITS[kind]:
                raise BudgetError(f'Frozen {kind} fit budget exhausted: {LIMITS[kind]}')
            if kind != 'pca' and counts['classifier'] >= LIMITS['classifier']:
                raise BudgetError('Frozen classifier fit budget exhausted: 54')
            record = self._base('claim', len(events) + 1, job_key)
            record['kind'] = kind
            self._append(record)
            return record

    def finish(self, job_key, status='complete', details=None):
        if status not in STATUSES:
            raise ValueError('finish status must be complete or failed')
        if details is None:
            details = {}
        if not isinstance(details, dict):
            raise ValueError('finish details must be a JSON object')
        with self._exclusive():
            events, jobs = self._read()
            if job_key not in jobs or jobs[job_key]['status'] != 'pending':
                raise BudgetError('Finish requires an existing pending claim; terminal events are immutable')
            record = self._base('finish', len(events) + 1, job_key)
            record.update(status=status, details=details)
            self._append(record)
            return record

    def summary(self):
        with self._exclusive():
            _, jobs = self._read()
            counts = self._counts(jobs)
            counts.update({status: sum(job['status'] == status for job in jobs.values())
                           for status in ('complete', 'failed', 'pending')})
            counts['limits'] = dict(LIMITS)
            counts['jobs'] = jobs
            return counts
