import unittest

from sef_exp.evaluador_series import decodificar, orden_diferencias
from sef_exp.series import CFG, FAMILIAS_SERIES, bits_series, hk_forward


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

    def test_bits_series_mezcla_dummies_y_enteros(self):
        v = bits_series(("p2q1", "IPC", "TPM"), CFG)
        self.assertEqual(v, [1, 1, 0, 0, 0, 2, 1])

    def test_d_es_determinista_y_acotado(self):
        import numpy as np
        rng = np.random.default_rng(0)
        y = np.cumsum(rng.normal(size=150))
        self.assertEqual(orden_diferencias(y), orden_diferencias(y))
        self.assertLessEqual(orden_diferencias(y), 2)


if __name__ == "__main__":
    unittest.main()
