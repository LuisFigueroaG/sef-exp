import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from elegir_k import elegir


class ReglaTests(unittest.TestCase):
    def test_elige_menor_k_dentro_de_epsilon(self):
        medianas = {2: 0.70, 3: 0.755, 4: 0.7600, 5: 0.7615,
                    6: 0.7616, 7: 0.7616, 8: 0.7614}
        elegido, techo = elegir(medianas, 0.001)
        self.assertEqual(techo, 0.7616)
        self.assertEqual(elegido, 5)

    def test_empate_exacto_gana_chico(self):
        elegido, _ = elegir({2: 0.76, 3: 0.76}, 0.001)
        self.assertEqual(elegido, 2)


if __name__ == "__main__":
    unittest.main()
