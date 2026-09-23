import random
import unittest

from scripts.tpe_series2 import reparar


class ReparaTests(unittest.TestCase):
    FAMS = [("pq", ["p0q0", "p1q1"]),
            ("a", ["A", "A_L1"]), ("b", ["B"]), ("c", ["C"])]

    def test_repara_a_k_exacto(self):
        rng = random.Random(0)
        for _ in range(200):
            base = {f: rng.choice([None, *vs]) for f, vs in self.FAMS}
            S = reparar(base, self.FAMS, rng)
            pq = [v for v in S if v.startswith("p")]
            ex = [v for v in S if not v.startswith("p")]
            self.assertEqual(len(pq), 1)
            self.assertEqual(len(ex), 2)

    def test_determinista_misma_semilla(self):
        base = {"pq": "p0q0", "a": "A", "b": "B", "c": "C"}
        r1 = reparar(dict(base), self.FAMS, random.Random(7))
        r2 = reparar(dict(base), self.FAMS, random.Random(7))
        self.assertEqual(r1, r2)


if __name__ == "__main__":
    unittest.main()
