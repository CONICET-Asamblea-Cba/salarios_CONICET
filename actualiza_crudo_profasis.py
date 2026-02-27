import pandas as pd

# Leer datos
indice = pd.read_csv("datos/profasis_indice.csv", header=None, names=["fecha", "indice"])
bolsillo = pd.read_csv("datos/profasis_bolsillo.csv", header=None, names=["fecha", "bolsillo"])

# Obtener el valor de bolsillo (único valor)
valor_bolsillo = bolsillo["bolsillo"].iloc[0]

# Normalizar índice: dividir por el último valor disponible
ultimo_indice = indice["indice"].iloc[-1]
indice["salario"] = (valor_bolsillo * indice["indice"] / ultimo_indice).round(0).astype(int)

# Guardar crudo_profasis.csv
indice[["fecha", "salario"]].to_csv("datos/crudo_profasis.csv", index=False)

print(indice[["fecha", "salario"]].to_string(index=False))
print(f"\nBolsillo: {valor_bolsillo}")
print(f"Último índice: {ultimo_indice}")
