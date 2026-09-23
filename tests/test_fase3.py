import csv
import io
import random
import unittest
from contextlib import redirect_stdout

from sef_exp.fase2 import catalogo, vecindarios
from sef_exp.fase3 import (
    TRIAL_COLUMNS,
    VARIABLES,
    VARIANTES,
    crossover,
    leer_oraculo,
    mutate,
    run_variant,
)


class OperatorsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.universe = catalogo()
        cls.oracle = leer_oraculo(cls.universe)
        cls.near = vecindarios(cls.universe)
        cls.iv = {v: 0.2 for v in VARIABLES}
        cls.optimum = max(float(r["J"]) for r in cls.oracle.values() if r["J"])

    def run_short(self, variant, seed, limit):
        f = io.StringIO()
        writer = csv.DictWriter(f, fieldnames=TRIAL_COLUMNS, lineterminator="\n")
        writer.writeheader()
        with redirect_stdout(io.StringIO()):
            summary = run_variant(variant, seed, self.universe, self.oracle,
                                  self.near, self.iv, writer, f, self.optimum,
                                  trial_limit=limit)
        return list(csv.DictReader(io.StringIO(f.getvalue()))), summary

    def test_operators_keep_family_and_cardinality_rules(self):
        rng = random.Random(9)
        valid = set(self.universe)
        for _ in range(500):
            a, b = rng.sample(self.universe, 2)
            crossed, _ = crossover(a, b, rng)
            self.assertIn(crossed, valid)
            self.assertIn(mutate(a, rng), valid)
            self.assertIn(mutate(a, rng, doubles=False), valid)

    def test_run_is_reproducible_and_evaluates_unique_sets(self):
        first, summary1 = self.run_short(VARIANTES[2], 0, 100)
        second, summary2 = self.run_short(VARIANTES[2], 0, 100)
        self.assertEqual([r["S"] for r in first], [r["S"] for r in second])
        self.assertEqual(len(first), 100)
        self.assertEqual(len({r["S"] for r in first}), 100)
        self.assertEqual(summary1[0]["mejor_J"], summary2[0]["mejor_J"])
        self.assertGreaterEqual(int(summary1[0]["propuestas"]), 100)

    def test_surrogate_is_prequential_and_repair_ablation_counts_rejections(self):
        trials, summary = self.run_short(VARIANTES[2], 1, 500)
        self.assertEqual(len(trials), 500)
        self.assertGreater(int(summary[2]["ajustes_sustituto"]), 0)
        self.assertGreater(int(summary[2]["predicciones_validas"]), 0)
        self.assertTrue(any(r["pred_J"] for r in trials))
        _, no_repair = self.run_short(VARIANTES[-1], 1, 250)
        self.assertGreater(int(no_repair[1]["estructurales_rechazados"]), 0)


if __name__ == "__main__":
    unittest.main()
