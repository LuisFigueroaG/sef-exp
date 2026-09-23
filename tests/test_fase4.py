import unittest

from sef_exp import evaluador as evaluador_mod
from sef_exp import fase3b
from sef_exp.fase4 import CFG, FAMILIAS_TAIWAN, cargar_taiwan


class Fase4Tests(unittest.TestCase):
    def test_espacio_y_umbral(self):
        self.assertEqual(len(CFG.espacio.variables), 23)
        self.assertEqual(len(CFG.espacio.familias), 11)
        self.assertEqual(CFG.espacio.total, 26_262)
        self.assertEqual(CFG.iv_min, 0.02)
        self.assertEqual(CFG.k, 6)

    def test_loader_solo_train(self):
        x, y = cargar_taiwan()
        self.assertEqual(x.shape, (24001, 23))
        self.assertEqual(len(y), 24001)
        self.assertEqual(set(x.columns), set(CFG.espacio.variables))

    def test_umbrales_se_aplican_y_restauran(self):
        previos = (evaluador_mod.IV_MIN, evaluador_mod.CORR_MAX,
                   evaluador_mod.VIF_MAX)
        fase3b.aplicar_umbrales(CFG)
        self.assertEqual(evaluador_mod.IV_MIN, 0.02)
        fase3b.restaurar_umbrales(previos)
        self.assertEqual(
            (evaluador_mod.IV_MIN, evaluador_mod.CORR_MAX,
             evaluador_mod.VIF_MAX), previos)

    def test_forward_greedy_usa_config(self):
        def fake(S, k):
            if len(S) != k:
                return None
            return sum(CFG.espacio.variables.index(v) for v in S) / 10000

        first = fase3b.forward_greedy(fake, CFG)
        second = fase3b.forward_greedy(fake, CFG)
        self.assertEqual(first, second)
        J, S, _ = first
        self.assertIsNotNone(J)
        self.assertTrue(CFG.espacio.valido(S))


if __name__ == "__main__":
    unittest.main()
