import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from elegir_x import elegir


class ReglaXTests(unittest.TestCase):
    def test_elige_menor_n_dentro_de_epsilon(self):
        medianas = {0: -560.0, 1: -552.0, 2: -550.5, 3: -548.5,
                    4: -548.2, 5: -548.0}
        elegido, techo = elegir(medianas, 2.0)
        self.assertEqual(techo, -548.0)
        self.assertEqual(elegido, 3)


if __name__ == "__main__":
    unittest.main()
