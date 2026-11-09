import random
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from adaptive_region_ordering_search import AdaptiveRegionOrderingSearch

class AdaptiveRegionOrderingSearchTests(unittest.TestCase):
    def test_finds_every_present_value(self):
        rng = random.Random(20261109)
        for n in range(1, 150):
            values = list(range(n)); rng.shuffle(values)
            search = AdaptiveRegionOrderingSearch(values)
            for target in values:
                index, comparisons = search.search(target)
                self.assertIsNotNone(index)
                self.assertEqual(values[index], target)
                self.assertGreaterEqual(comparisons, 1)
                self.assertLessEqual(comparisons, n)

    def test_missing_values_return_none(self):
        values = [19, -4, 8, 100, 7, 0]
        search = AdaptiveRegionOrderingSearch(values)
        for target in (-999, 1, 2, 200):
            index, comparisons = search.search(target)
            self.assertIsNone(index)
            self.assertEqual(comparisons, len(values))

    def test_duplicates_return_a_valid_matching_position(self):
        values = [5, 1, 5, 9, 1, 5, 7]
        search = AdaptiveRegionOrderingSearch(values)
        for target in (1, 5, 7, 9):
            index, _ = search.search(target)
            self.assertIsNotNone(index)
            self.assertEqual(values[index], target)

    def test_probability_and_order_invariants(self):
        rng = random.Random(28)
        search = AdaptiveRegionOrderingSearch(list(range(200)))
        for _ in range(2_000):
            search.learn(rng.randrange(4))
            self.assertAlmostEqual(sum(search.pred), 1.0, places=10)
            self.assertTrue(all(p >= 0 for p in search.pred))
            self.assertEqual(set(search.order), {0, 1, 2, 3})

    def test_empty_and_single_element_arrays(self):
        empty = AdaptiveRegionOrderingSearch([])
        self.assertEqual(empty.search(1), (None, 0))
        one = AdaptiveRegionOrderingSearch([42])
        self.assertEqual(one.search(42), (0, 1))
        self.assertEqual(one.search(-1), (None, 1))

if __name__ == "__main__":
    unittest.main(verbosity=2)
