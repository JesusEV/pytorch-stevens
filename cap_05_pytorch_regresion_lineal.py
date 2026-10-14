import torch
import matplotlib.pyplot as plt


def modelo(x, parametros):
    w, b = parametros
    return w * x + b


def perdida(prediccion, objetivo):
    return torch.mean((prediccion - objetivo) ** 2)


# Datos
celsius = torch.tensor([
    0.5, 14.0, 15.0, 28.0, 11.0, 8.0,
    3.0, -4.0, 6.0, 13.0, 21.0
])

lectura = torch.tensor([
    35.7, 55.9, 58.2, 81.9, 56.3, 48.9,
    33.9, 21.8, 48.4, 60.4, 68.4
])


# Separar datos de entrenamiento y validación
torch.manual_seed(2026)

indices = torch.randperm(len(lectura))

indices_validacion = indices[:2]
indices_entrenamiento = indices[2:]

x_entrenamiento = lectura[indices_entrenamiento]
y_entrenamiento = celsius[indices_entrenamiento]

x_validacion = lectura[indices_validacion]
y_validacion = celsius[indices_validacion]


# Normalizar las entradas
x_entrenamiento = 0.1 * x_entrenamiento
x_validacion = 0.1 * x_validacion


# Parámetros iniciales
parametros = torch.tensor([1.0, 0.0], requires_grad=True)

optimizador = torch.optim.SGD([parametros], lr=0.01)



# Entrenamiento
for epoca in range(5000):

    # PREDICE -> MIDE EL ERROR -> CALCULAR GRADIENTES -> ACTUALIZAR LOS PESOS
    prediccion = modelo(x_entrenamiento, parametros)

    error = perdida(prediccion, y_entrenamiento)

    optimizador.zero_grad()

    error.backward()

    optimizador.step()

    if epoca % 500 == 0:

        with torch.no_grad():
            prediccion_validacion = modelo(x_validacion, parametros)
            error_validacion = perdida(
                prediccion_validacion,
                y_validacion
            )

        print(
            epoca,
            error.item(),
            error_validacion.item()
        )


# Parámetros aprendidos
print("\nParámetros:")
print(parametros)


# Gráfica
x_recta = torch.linspace(lectura.min(), lectura.max(), 200)

with torch.no_grad():
    y_recta = modelo(0.1 * x_recta, parametros)


plt.scatter(lectura, celsius, label="Datos")

plt.plot(
    x_recta,
    y_recta,
    label="Modelo"
)

plt.xlabel("Lectura del termómetro")
plt.ylabel("Temperatura (°C)")

plt.legend()
plt.grid()

plt.show()

