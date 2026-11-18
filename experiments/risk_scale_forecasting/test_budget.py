import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from budget import BudgetLedger, BudgetError, LedgerBusyError, LedgerCorruptionError


class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / 'claims.jsonl'
        self.ledger = BudgetLedger(self.path)

    def tearDown(self):
        self.temp.cleanup()

    def test_empty_then_durable_claim_and_finish(self):
        self.assertEqual(self.ledger.summary()['total'], 0)
        with patch('budget.os.fsync', wraps=__import__('os').fsync) as sync:
            self.ledger.claim('2022/F0', 'har')
            self.assertGreaterEqual(sync.call_count, 2)  # lock and journal
        self.assertTrue(self.path.read_bytes().endswith(b'\n'))
        self.ledger.finish('2022/F0', details={'iterations': 4})
        state = BudgetLedger(self.path).summary()
        self.assertEqual((state['har'], state['supervised'], state['complete']), (1, 1, 1))
        self.assertEqual(state['jobs']['2022/F0']['details'], {'iterations': 4})

    def test_failed_and_pending_claims_remain_charged_after_restart(self):
        self.ledger.claim('failed-fit', 'neural')
        self.ledger.finish('failed-fit', 'failed', {'error': 'synthetic failure'})
        self.ledger.claim('interrupted-fit', 'neural')
        state = BudgetLedger(self.path).summary()
        self.assertEqual((state['neural'], state['failed'], state['pending']), (2, 1, 1))
        for key in ['failed-fit', 'interrupted-fit']:
            with self.assertRaises(BudgetError):BudgetLedger(self.path).claim(key, 'neural')

    def test_duplicate_key_across_kinds_rejected(self):
        self.ledger.claim('same', 'neural')
        before = self.path.read_bytes()
        with self.assertRaises(BudgetError):self.ledger.claim('same', 'har')
        self.assertEqual(before, self.path.read_bytes())

    def test_all_frozen_budgets_and_restart(self):
        for kind, maximum in [('neural', 27), ('har', 6), ('pca', 0)]:
            for i in range(maximum):self.ledger.claim(f'{kind}/{i}', kind)
            with self.assertRaises(BudgetError):BudgetLedger(self.path).claim(f'{kind}/overflow', kind)
        state = BudgetLedger(self.path).summary()
        self.assertEqual((state['supervised'], state['pca'], state['total']), (33, 0, 33))
        self.assertEqual(state['pending'], 33)

    def test_terminal_transitions_cannot_be_rewritten(self):
        with self.assertRaises(BudgetError):self.ledger.finish('absent')
        self.ledger.claim('a', 'har');self.ledger.finish('a', 'failed')
        with self.assertRaises(BudgetError):self.ledger.finish('a', 'complete')
        with self.assertRaises(ValueError):self.ledger.finish('a', 'retry')

    def test_invalid_kind_key_and_details_do_not_append(self):
        for key, kind in [('', 'neural'), ('a', 'unknown')]:
            with self.assertRaises(ValueError):self.ledger.claim(key, kind)
        self.ledger.claim('a', 'har');before = self.path.read_bytes()
        with self.assertRaises(ValueError):self.ledger.finish('a', details={'bad': float('nan')})
        self.assertEqual(before, self.path.read_bytes())
        self.assertEqual(self.ledger.summary()['pending'], 1)

    def test_partial_or_corrupt_journal_refuses_recovery(self):
        self.ledger.claim('a', 'har')
        original = self.path.read_bytes()
        self.path.write_bytes(original + b'{"event":')
        with self.assertRaises(LedgerCorruptionError):BudgetLedger(self.path)
        self.assertEqual(self.path.read_bytes(), original + b'{"event":')
        self.path.write_bytes(original + b'garbage\n')
        with self.assertRaises(LedgerCorruptionError):BudgetLedger(self.path)

    def test_external_duplicate_event_is_corrupt(self):
        self.ledger.claim('a', 'har')
        row=json.loads(self.path.read_text());row['sequence']=2
        with self.path.open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
        with self.assertRaises(LedgerCorruptionError):BudgetLedger(self.path)

    def test_lock_is_fail_closed_without_auto_retry(self):
        self.ledger.lock_path.write_text('another-process')
        with self.assertRaises(LedgerBusyError):self.ledger.claim('a', 'neural')
        self.assertEqual(self.ledger.lock_path.read_text(), 'another-process')
        self.ledger.lock_path.unlink()
        self.assertEqual(self.ledger.summary()['total'], 0)


if __name__ == '__main__':unittest.main()
