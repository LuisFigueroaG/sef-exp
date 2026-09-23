import random
import unittest

from sef_exp import tpe_comun


class ReparaTests(unittest.TestCase):
    FAMS = [("pq", ["p0q0", "p1q1"]),
            ("a", ["A", "A_L1"]), ("b", ["B"]), ("c", ["C"])]

    def test_repara_a_k_exacto(self):
        from scripts.tpe_series2 import reparar
        rng = random.Random(0)
        for _ in range(200):
            base = {f: rng.choice([None, *vs]) for f, vs in self.FAMS}
            S = reparar(base, self.FAMS, rng)
            pq = [v for v in S if v.startswith("p")]
            ex = [v for v in S if not v.startswith("p")]
            self.assertEqual(len(pq), 1)
            self.assertEqual(len(ex), 2)

    def test_determinista_misma_semilla(self):
        from scripts.tpe_series2 import reparar
        base = {"pq": "p0q0", "a": "A", "b": "B", "c": "C"}
        r1 = reparar(dict(base), self.FAMS, random.Random(7))
        r2 = reparar(dict(base), self.FAMS, random.Random(7))
        self.assertEqual(r1, r2)

    def test_repara_generico_a_k(self):
        fams = [("f1", ["a1", "a2"]), ("f2", ["b1"]),
                ("f3", ["c1"]), ("f4", ["d1"])]
        rng = random.Random(3)
        for _ in range(200):
            base = {f: rng.choice([None, *vs]) for f, vs in fams}
            S = tpe_comun.reparar(base, fams, rng, 2)
            self.assertEqual(len(S), 2)
            self.assertEqual(len(set(S)), 2)


if __name__ == "__main__":
    unittest.main()
