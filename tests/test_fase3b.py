import csv
import io
import random
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

from sef_exp.espacio import Espacio
from sef_exp import fase3b
from sef_exp.fase3b import SPACE, TRIAL_COLUMNS, SUMMARY_COLUMNS, run_method


class EspacioTests(unittest.TestCase):
    def test_catalogo_completo_y_muestra_uniforme(self):
        self.assertEqual(len(SPACE.variables), 35)
        self.assertEqual(len(SPACE.familias), 27)
        self.assertEqual(SPACE.total, 1_251_023)
        toy = Espacio((("a", ("a1", "a2")), ("b", ("b1",)),
                       ("c", ("c1", "c2"))), k=2)
        self.assertEqual(toy.total, 8)
        rng = random.Random(7)
        counts = {}
        for _ in range(8000):
            S = toy.muestra(rng)
            self.assertTrue(toy.valido(S))
            counts[S] = counts.get(S, 0) + 1
        self.assertEqual(len(counts), 8)
        self.assertTrue(all(850 <= count <= 1150 for count in counts.values()))

    def test_operadores_mantienen_restricciones_con_familias_individuales(self):
        rng = random.Random(9)
        for _ in range(500):
            a, b = SPACE.muestra(rng), SPACE.muestra(rng)
            self.assertTrue(SPACE.valido(SPACE.cruza(a, b, rng)[0]))
            self.assertTrue(SPACE.valido(SPACE.muta(a, rng)))
            self.assertTrue(SPACE.valido(SPACE.muta(a, rng, dobles=False)))
            self.assertTrue(all(SPACE.valido(S) for S in SPACE.vecinos_simples(a)))


class FakeCache:
    def consulta(self, S, context):
        score = 0.6 + (abs(hash(S)) % 10000) / 100000
        return {"estado": "ok", "causa": "", "J": repr(score),
                "aucs": "[]", "pared_s": "0.001"}, False


class SearchTests(unittest.TestCase):
    def test_metodos_respetan_presupuesto_y_trials_unicos(self):
        iv = {v: 0.2 for v in SPACE.variables}
        for method in ("aleatorio", "local", "ga_local", "sef_completo"):
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=TRIAL_COLUMNS, lineterminator="\n")
            writer.writeheader()
            global_best = {"J": None, "S": None, "metodo": None, "semilla": None}
            with redirect_stdout(io.StringIO()):
                summaries = run_method(method, 0, 100, None, iv, FakeCache(),
                                       writer, global_best)
            rows = list(csv.DictReader(io.StringIO(output.getvalue())))
            self.assertEqual(len(rows), 100)
            self.assertEqual(len({r["S"] for r in rows}), 100)
            self.assertTrue(all(SPACE.valido(tuple(r["S"].split("|"))) for r in rows))
            self.assertEqual(len(summaries), 1)
            self.assertEqual(summaries[0]["presupuesto"], 100)
            if method == "sef_completo":
                self.assertGreater(int(summaries[0]["ajustes_sustituto"]), 0)

    def test_espacio_chico_termina_sin_bucle(self):
        toy = Espacio((("a", ("a1", "a2")), ("b", ("b1",)),
                       ("c", ("c1", "c2"))), k=2)
        self.assertEqual(toy.total, 8)
        stub = SimpleNamespace(espacio=toy, presupuestos=(4, 8), bits_fn=None)
        for method in ("aleatorio", "local", "ga_local", "sef_completo"):
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=TRIAL_COLUMNS, lineterminator="\n")
            writer.writeheader()
            global_best = {"J": None, "S": None, "metodo": None, "semilla": None}
            with redirect_stdout(io.StringIO()):
                run_method(method, 0, 8, None,
                           {v: 0.2 for v in toy.variables}, FakeCache(),
                           writer, global_best, stub)
            rows = list(csv.DictReader(io.StringIO(output.getvalue())))
            self.assertEqual(len(rows), 8)
            self.assertEqual(len({r["S"] for r in rows}), 8)

    def test_metricas_y_estabilidad_completas(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            summary = root / "resumen.csv"
            metrics = root / "metricas.csv"
            stability = root / "estabilidad.csv"
            with summary.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS, lineterminator="\n")
                writer.writeheader()
                for method in fase3b.METHODS:
                    for seed in fase3b.SEEDS:
                        for budget in fase3b.BUDGETS:
                            writer.writerow({
                                "metodo": method, "semilla": seed, "presupuesto": budget,
                                "mejor_J": 0.7 + seed / 10000,
                                "mejor_S": "|".join(SPACE.muestra(random.Random(seed))),
                                "propuestas": budget, "rechazados": 0,
                                "pared_modelo_s": 1, "pared_algoritmo_s": 0.1,
                            })
            stub = SimpleNamespace(
                espacio=SPACE, summary_path=summary,
                metrics_path=metrics, stability_path=stability,
                presupuestos=fase3b.BUDGETS, epsilon=0.001,
                etiqueta_y="AUC medio de cinco folds",
                semillas=fase3b.SEEDS)
            result = fase3b.metricas(stub)
            self.assertEqual(len(result), len(fase3b.METHODS) * len(fase3b.BUDGETS))
            with stability.open() as f:
                stability_rows = list(csv.DictReader(f))
            self.assertEqual(len(stability_rows),
                             len(result) * (len(SPACE.variables) + len(SPACE.familias)))


class ForwardTests(unittest.TestCase):
    def test_greedy_determinista_y_respeta_familias(self):
        def fake(S, k):
            if len(S) != k:
                return None
            return sum(SPACE.variables.index(v) for v in S) / 10000

        first = fase3b.forward_greedy(fake)
        second = fase3b.forward_greedy(fake)
        self.assertEqual(first, second)
        J, S, ensayos = first
        self.assertIsNotNone(J)
        self.assertTrue(SPACE.valido(S))
        self.assertLessEqual(ensayos, sum(
            len(SPACE.variables) - i for i in range(SPACE.k)))

    def test_greedy_bloqueado_devuelve_nulo(self):
        J, S, ensayos = fase3b.forward_greedy(lambda S, k: None)
        self.assertIsNone(J)
        self.assertIsNone(S)
        self.assertGreater(ensayos, 0)


if __name__ == "__main__":
    unittest.main()
