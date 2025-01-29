def merge_sets(sets):
    resultado = []
    while sets:
        primer, *resto = sets
        primer = set(primer)
        len_anterior = -1
        while len(primer) > len_anterior:
            len_anterior = len(primer)
            resto2 = []
            for s in resto:
                if primer & s:
                    primer |= s
                else:
                    resto2.append(s)
            resto = resto2
        resultado.append(primer)
        sets = resto
    return resultado