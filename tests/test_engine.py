import unittest

from cpm.engine import PDMScheduler


class TestPDMScheduler(unittest.TestCase):
    def test_simple_fs_chain(self):
        scheduler = PDMScheduler(normalize_to_zero=False)
        scheduler.add_activity("A", "A", 3, "")
        scheduler.add_activity("B", "B", 2, "A:FS:0")
        scheduler.add_activity("C", "C", 1, "B:FS:0")

        ok, _ = scheduler.calculate()
        self.assertTrue(ok)
        self.assertEqual(scheduler.project_duration, 6)

        self.assertEqual(scheduler.activities["A"].es, 0)
        self.assertEqual(scheduler.activities["A"].ef, 3)
        self.assertEqual(scheduler.activities["B"].es, 3)
        self.assertEqual(scheduler.activities["B"].ef, 5)
        self.assertEqual(scheduler.activities["C"].es, 5)
        self.assertEqual(scheduler.activities["C"].ef, 6)

        self.assertEqual(scheduler.critical_path, ["A", "B", "C"])

    def test_negative_lag_allows_negative_es(self):
        scheduler = PDMScheduler(normalize_to_zero=False)
        scheduler.add_activity("A", "A", 5, "")
        scheduler.add_activity("B", "B", 3, "A:SS:-2")

        ok, _ = scheduler.calculate()
        self.assertTrue(ok)

        self.assertEqual(scheduler.activities["A"].es, 0)
        self.assertEqual(scheduler.activities["B"].es, -2)
        self.assertEqual(scheduler.activities["B"].ef, 1)

    def test_multiple_critical_paths(self):
        scheduler = PDMScheduler()
        scheduler.add_activity("A", "A", 2, "")
        scheduler.add_activity("B", "B", 2, "")
        scheduler.add_activity("C", "C", 2, "A:FS:0")
        scheduler.add_activity("D", "D", 2, "B:FS:0")
        scheduler.add_activity("E", "E", 2, "C:FS:0;D:FS:0")

        ok, _ = scheduler.calculate()
        self.assertTrue(ok)

        paths = {tuple(p) for p in scheduler.critical_paths}
        self.assertEqual(len(paths), 2)
        self.assertIn(("A", "C", "E"), paths)
        self.assertIn(("B", "D", "E"), paths)


if __name__ == "__main__":
    unittest.main()
