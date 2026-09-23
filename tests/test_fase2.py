import unittest

from sef_exp.evaluador import evaluar, hacer_folds, precomputar, precomputar_woe
from sef_exp.fase2 import FAMILIAS, cargar_train, catalogo, vecindarios


class CatalogoTests(unittest.TestCase):
    def test_space_has_expected_size_and_family_rule(self):
        universe = catalogo()
        self.assertEqual(len(universe), 1792)
        self.assertEqual(len(set(universe)), 1792)
        lookup = {v: family for family, choices in FAMILIAS for v in choices}
        for S in universe:
            self.assertEqual(len(S), 6)
            self.assertEqual(len({lookup[v] for v in S}), 6)

    def test_neighbors_are_symmetric_and_exact_distance(self):
        universe = catalogo()
        near = vecindarios(universe)
        for S in universe[:12]:
            for distance in (1, 2):
                for other in near[S][distance - 1]:
                    self.assertEqual(len(set(S) - set(other)), distance)
                    self.assertIn(S, near[other][distance - 1])


class EvaluadorCacheTests(unittest.TestCase):
    def test_cached_woe_matches_direct_transform(self):
        x, y = cargar_train()
        folds = hacer_folds(y)
        sets = (
            ("AM03", "CVPRPAYMNT11", "CVSC150", "GO243", "IN20", "PR09"),
            ("AM03", "AT93", "BC111", "CVPRPAYMNT11", "CVSC100", "GO06"),
        )
        vars_ = sorted({v for S in sets for v in S})
        pre = precomputar(x, y, folds, vars_)
        cache = precomputar_woe(x, folds, pre, vars_)
        for S in sets:
            direct = evaluar(S, x, y, folds, pre)
            cached = evaluar(S, x, y, folds, pre, cache)
            self.assertEqual(direct["estado"], cached["estado"])
            self.assertEqual(direct["causa"], cached["causa"])
            self.assertEqual(direct["aucs"], cached["aucs"])
            self.assertEqual(direct["J"], cached["J"])


if __name__ == "__main__":
    unittest.main()
