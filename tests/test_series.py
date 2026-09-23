import unittest

from sef_exp.evaluador_series import decodificar, orden_diferencias
from sef_exp.series import CFG, FAMILIAS_SERIES, bits_series, familias_series
from sef_exp.series2 import CFG2, EXOGENAS2, LAGS2


class SeriesTests(unittest.TestCase):
    def test_espacio_130_con_10_sin_pq(self):
        self.assertEqual(len(CFG.espacio.variables), 17)
        self.assertEqual(len(CFG.espacio.familias), 6)
        self.assertEqual(CFG.espacio.total, 130)
        self.assertEqual(CFG.presupuestos, (25, 50, 75, 100))

    def test_decodificar_tokens(self):
        self.assertEqual(decodificar(("p2q1", "IPC", "TPM")),
                         (2, 1, ("IPC", "TPM")))
        p, q, ex = decodificar(("IPC", "TPM", "DOLAR"))
        self.assertIsNone(p)
        self.assertEqual(len(ex), 3)

    def test_bits_series_one_hot_uniforme(self):
        v = bits_series(("p2q1", "IPC", "TPM"), CFG)
        self.assertEqual(len(v), len(CFG.espacio.variables))
        self.assertEqual(sum(v), 3)
        w = bits_series(("IPC", "TPM", "DOLAR"), CFG)
        self.assertEqual(len(w), len(CFG.espacio.variables))
        self.assertEqual(sum(w), 3)

    def test_d_es_determinista_y_acotado(self):
        import numpy as np
        rng = np.random.default_rng(0)
        y = np.cumsum(rng.normal(size=150))
        self.assertEqual(orden_diferencias(y), orden_diferencias(y))
        self.assertLessEqual(orden_diferencias(y), 2)

    def test_series2_espacio_ampliado(self):
        self.assertEqual(len(CFG2.espacio.familias), 8)
        self.assertEqual(CFG2.espacio.total, 5481)
        self.assertEqual(CFG2.presupuestos, (100, 250, 500, 1000, 2000))
        self.assertEqual(CFG2.exogenas, EXOGENAS2)
        self.assertEqual(CFG2.lags, LAGS2)
        fams = familias_series(EXOGENAS2, LAGS2, 5, 3)
        self.assertEqual(len(fams[0][1]), 24)
        self.assertEqual(len(fams[1][1]), 3)


if __name__ == "__main__":
    unittest.main()
