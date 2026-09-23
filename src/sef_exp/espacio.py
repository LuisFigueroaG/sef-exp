"""Structural search space with one selected variable per family."""

import itertools
import random


class Espacio:
    def __init__(self, familias, k=6):
        self.familias = tuple((name, tuple(choices)) for name, choices in familias)
        self.k = k
        self.variables = tuple(sorted(v for _, choices in self.familias for v in choices))
        self.familia_de = {v: name for name, choices in self.familias for v in choices}
        self.opciones = dict(self.familias)
        if len(self.variables) != len(self.familia_de):
            raise ValueError("Una variable aparece en mas de una familia")
        if len({name for name, _ in self.familias}) != len(self.familias):
            raise ValueError("Nombres de familia repetidos")
        if len(self.familias) < k or any(not choices for _, choices in self.familias):
            raise ValueError("Familias insuficientes o vacias")
        self.cola = [[0] * (k + 1) for _ in range(len(self.familias) + 1)]
        self.cola[-1][0] = 1
        for i in range(len(self.familias) - 1, -1, -1):
            self.cola[i][0] = 1
            for j in range(1, k + 1):
                self.cola[i][j] = (self.cola[i + 1][j]
                                   + len(self.familias[i][1]) * self.cola[i + 1][j - 1])
        self.total = self.cola[0][k]

    def valido(self, S):
        return len(S) == self.k and len(set(S)) == self.k and all(
            v in self.familia_de for v in S
        ) and len({self.familia_de[v] for v in S}) == self.k

    def muestra(self, rng):
        restantes = self.k
        selected = []
        for i, (_, choices) in enumerate(self.familias):
            if not restantes:
                break
            omitir = self.cola[i + 1][restantes]
            tomar = len(choices) * self.cola[i + 1][restantes - 1]
            if rng.randrange(omitir + tomar) >= omitir:
                selected.append(rng.choice(choices))
                restantes -= 1
        return tuple(sorted(selected))

    def guiada(self, iv, rng):
        ranked = []
        for name, choices in self.familias:
            chosen = max(choices, key=lambda v: (iv[v] + rng.gauss(0, 0.05), v))
            ranked.append((iv[chosen] + rng.gauss(0, 0.05), name, chosen))
        ranked.sort(reverse=True)
        return tuple(sorted(v for _, _, v in ranked[:self.k]))

    def vecinos_simples(self, S):
        selected = set(S)
        active = {self.familia_de[v] for v in S}
        for old in S:
            family = self.familia_de[old]
            for new in self.opciones[family]:
                if new != old:
                    yield tuple(sorted((selected - {old}) | {new}))
            for other, choices in self.familias:
                if other not in active:
                    for new in choices:
                        yield tuple(sorted((selected - {old}) | {new}))

    def muta(self, S, rng, dobles=True):
        selected = set(S)
        active = {self.familia_de[v] for v in S}
        same = [v for v in S if len(self.opciones[self.familia_de[v]]) > 1]
        inactive = [name for name, _ in self.familias if name not in active]
        modes = []
        if same:
            modes.append(("misma", 0.5))
        if inactive:
            modes.append(("familia", 0.3))
        if dobles:
            modes.append(("doble", 0.2))
        if not modes:
            return S
        mode = rng.choices([m for m, _ in modes], weights=[w for _, w in modes])[0]
        if mode == "doble":
            for _ in range(16):
                child = self.muta(self.muta(S, rng, False), rng, False)
                if len(set(S) - set(child)) == 2:
                    return child
            return child
        old = rng.choice(same if mode == "misma" else S)
        selected.remove(old)
        if mode == "misma":
            choices = [v for v in self.opciones[self.familia_de[old]] if v != old]
            selected.add(rng.choice(choices))
        else:
            selected.add(rng.choice(self.opciones[rng.choice(inactive)]))
        return tuple(sorted(selected))

    def cruza(self, a, b, rng):
        ga = {self.familia_de[v]: v for v in a}
        gb = {self.familia_de[v]: v for v in b}
        selected = {}
        for name, _ in self.familias:
            if ga.get(name) == gb.get(name):
                choice = ga.get(name)
            else:
                choice = rng.choice((ga.get(name), gb.get(name)))
            if choice is not None:
                selected[name] = choice
        repaired = len(selected) != self.k
        while len(selected) > self.k:
            removable = [name for name in selected if ga.get(name) != gb.get(name)]
            selected.pop(rng.choice(removable or list(selected)))
        while len(selected) < self.k:
            name = rng.choice([name for name, _ in self.familias if name not in selected])
            parents = [v for v in (ga.get(name), gb.get(name)) if v is not None]
            selected[name] = rng.choice(parents or self.opciones[name])
        return tuple(sorted(selected.values())), repaired

    def diversidad(self, population):
        if len(population) < 2:
            return None
        return sum(1 - len(set(a) & set(b)) / len(set(a) | set(b))
                   for a, b in itertools.combinations(population, 2)) / (
                       len(population) * (len(population) - 1) / 2)
