import torch
from torch import nn
import matplotlib.pyplot as plt


# Datos: una fila por muestra y una columna por característica.
# unsqueeze(1) cambia la forma de [11] a [11, 1].
celsius = torch.tensor([
    0.5, 14.0, 15.0, 28.0, 11.0, 8.0,
    3.0, -4.0, 6.0, 13.0, 21.0
]).unsqueeze(1)

lectura = torch.tensor([
    35.7, 55.9, 58.2, 81.9, 56.3, 48.9,
    33.9, 21.8, 48.4, 60.4, 68.4
]).unsqueeze(1)


# Separar datos de entrenamiento y validación
torch.manual_seed(2026)

indices = torch.randperm(len(lectura))

indices_validacion = indices[:2]
indices_entrenamiento = indices[2:]

x_entrenamiento = lectura[indices_entrenamiento]
y_entrenamiento = celsius[indices_entrenamiento]

x_validacion = lectura[indices_validacion]
y_validacion = celsius[indices_validacion]


# Reescalar las entradas, igual que en el capítulo 5
x_entrenamiento = 0.1 * x_entrenamiento
x_validacion = 0.1 * x_validacion


# Red neuronal: una entrada, ocho neuronas ocultas y una salida.
# Tanh permite aprender una curva en lugar de solamente una recta.
# La salida no lleva Tanh porque la temperatura no está limitada a [-1, 1].
modelo = nn.Sequential(
    nn.Linear(1, 8),
    nn.Tanh(),
    nn.Linear(8, 1)
)

perdida = nn.MSELoss()

# El modelo contiene los pesos y sesgos de las dos capas.
optimizador = torch.optim.SGD(modelo.parameters(), lr=0.001)

print(modelo)
print("\nÉpoca, error de entrenamiento, error de validación")


# Entrenamiento
for epoca in range(5000):

    # PREDECIR -> MEDIR EL ERROR -> CALCULAR GRADIENTES -> ACTUALIZAR
    prediccion = modelo(x_entrenamiento)

    error = perdida(prediccion, y_entrenamiento)

    optimizador.zero_grad()

    error.backward()

    optimizador.step()

    if (epoca + 1) % 500 == 0:

        # Evaluar ambos conjuntos con los pesos ya actualizados.
        # La validación no se utiliza para calcular gradientes.
        with torch.no_grad():
            error_entrenamiento = perdida(
                modelo(x_entrenamiento),
                y_entrenamiento
            )
            error_validacion = perdida(
                modelo(x_validacion),
                y_validacion
            )

        print(
            epoca + 1,
            error_entrenamiento.item(),
            error_validacion.item()
        )


# Parámetros aprendidos
print("\nPesos y sesgos:")
for nombre, parametro in modelo.named_parameters():
    print(nombre)
    print(parametro)


# Gráfica: aplicar el mismo reescalamiento a las nuevas entradas
x_curva = torch.linspace(lectura.min(), lectura.max(), 200).unsqueeze(1)

with torch.no_grad():
    y_curva = modelo(0.1 * x_curva)

plt.scatter(
    lectura[indices_entrenamiento],
    celsius[indices_entrenamiento],
    label="Entrenamiento"
)
plt.scatter(
    lectura[indices_validacion],
    celsius[indices_validacion],
    label="Validación"
)
plt.plot(x_curva, y_curva, label="Red neuronal")

plt.xlabel("Lectura del termómetro")
plt.ylabel("Temperatura (°C)")
plt.legend()
plt.grid()

plt.show()
