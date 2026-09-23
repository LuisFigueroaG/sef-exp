import unittest

from sef_exp.hostil import CFG, FAMILIAS_HOSTIL


class HostilTests(unittest.TestCase):
    def test_espacio_153090(self):
        from sef_exp.espacio import Espacio
        self.assertEqual(Espacio(FAMILIAS_HOSTIL, 6).total, 153090)
        self.assertEqual(CFG.presupuestos, (50, 100, 250))
        self.assertEqual(CFG.semillas, tuple(range(10)))
        self.assertEqual(len(CFG.espacio.variables), 30)

    def test_hgb_distringue_par_vivo(self):
        from sef_exp.evaluador_hgb import evaluar_hgb
        from sef_exp.hostil import cargar_hostil
        from sef_exp.evaluador import hacer_folds
        x, y = cargar_hostil()
        folds = hacer_folds(y)
        vivo = evaluar_hgb(("v1b", "v2a", "v3a", "v4a", "v5a", "v6a"),
                           x, y, folds)
        ruido = evaluar_hgb(("v1a", "v2a", "v3a", "v6a", "v8a", "v10a"),
                            x, y, folds)
        self.assertEqual(vivo["estado"], "ok")
        self.assertGreater(vivo["J"], 0.6)
        if ruido["estado"] == "ok":
            self.assertLess(ruido["J"], vivo["J"])


if __name__ == "__main__":
    unittest.main()
