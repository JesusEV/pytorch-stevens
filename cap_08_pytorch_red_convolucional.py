import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


torch.manual_seed(2026)


# Convertir las imágenes a tensores y normalizar cada canal RGB.
# Las medias y desviaciones son las del entrenamiento de CIFAR-10.
transformacion = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4915, 0.4823, 0.4468),
        (0.2470, 0.2435, 0.2616)
    )
])


# Descargar CIFAR-10 la primera vez que se ejecuta el script.
datos_entrenamiento = datasets.CIFAR10(
    root="datos",
    train=True,
    download=True,
    transform=transformacion
)

# Igual que en el libro, usamos la partición de prueba como validación.
datos_validacion = datasets.CIFAR10(
    root="datos",
    train=False,
    download=True,
    transform=transformacion
)


# Conservar solamente aviones (0) y aves (2).
# Cambiar la etiqueta del ave a 1 para tener dos clases: 0 y 1.
entrenamiento = []
validacion = []

for imagen, etiqueta in datos_entrenamiento:
    if etiqueta == 0:
        entrenamiento.append((imagen, 0))
    elif etiqueta == 2:
        entrenamiento.append((imagen, 1))

for imagen, etiqueta in datos_validacion:
    if etiqueta == 0:
        validacion.append((imagen, 0))
    elif etiqueta == 2:
        validacion.append((imagen, 1))

print("Imágenes de entrenamiento:", len(entrenamiento))
print("Imágenes de validación:", len(validacion))


# DataLoader agrupa las imágenes en lotes de hasta 64 muestras.
lotes_entrenamiento = DataLoader(
    entrenamiento, batch_size=64, shuffle=True
)
lotes_validacion = DataLoader(
    validacion, batch_size=64, shuffle=False
)


# Usar la GPU si está disponible; en otro caso, usar la CPU.
if torch.cuda.is_available():
    dispositivo = torch.device("cuda")
else:
    dispositivo = torch.device("cpu")

print("Dispositivo:", dispositivo)


# Red convolucional del capítulo 8.
# B representa el número de imágenes del lote.
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # padding=1 conserva el alto y el ancho.
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(8 * 8 * 8, 32)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x):
        # [B, 3, 32, 32] -> [B, 16, 16, 16]
        x = self.conv1(x)
        x = torch.tanh(x)
        x = self.pool1(x)

        # [B, 16, 16, 16] -> [B, 8, 8, 8]
        x = self.conv2(x)
        x = torch.tanh(x)
        x = self.pool2(x)

        # Aplanar cada imagen: [B, 8, 8, 8] -> [B, 512].
        x = x.flatten(1)
        x = self.fc1(x)
        x = torch.tanh(x)
        x = self.fc2(x)
        return x


modelo = Net()

# Mover el modelo antes de crear el optimizador.
modelo = modelo.to(dispositivo)

# CrossEntropyLoss recibe logits y etiquetas enteras.
# No debemos agregar Softmax al final del modelo.
perdida = nn.CrossEntropyLoss()
optimizador = torch.optim.SGD(modelo.parameters(), lr=0.01)

errores_entrenamiento = []
errores_validacion = []

print(modelo)
print("\nÉpoca, error de entrenamiento, error de validación, exactitud")


# Entrenamiento
for epoca in range(100):
    modelo.train()
    suma_error = 0.0
    total = 0

    for imagenes, etiquetas in lotes_entrenamiento:
        # Las convoluciones reciben imágenes, no vectores aplanados.
        imagenes = imagenes.to(dispositivo)
        etiquetas = etiquetas.to(dispositivo)
        prediccion = modelo(imagenes)
        error = perdida(prediccion, etiquetas)

        optimizador.zero_grad()
        error.backward()
        optimizador.step()

        # Ponderar por el tamaño real del lote, incluido el último.
        suma_error += error.item() * len(etiquetas)
        total += len(etiquetas)

    errores_entrenamiento.append(suma_error / total)

    # Validar sin actualizar pesos ni calcular gradientes.
    modelo.eval()
    suma_error = 0.0
    aciertos = 0
    total = 0

    with torch.no_grad():
        for imagenes, etiquetas in lotes_validacion:
            imagenes = imagenes.to(dispositivo)
            etiquetas = etiquetas.to(dispositivo)
            prediccion = modelo(imagenes)
            error = perdida(prediccion, etiquetas)

            # Elegir la clase con la puntuación más alta de cada fila.
            clases = prediccion.argmax(dim=1)
            aciertos += (clases == etiquetas).sum().item()
            suma_error += error.item() * len(etiquetas)
            total += len(etiquetas)

    errores_validacion.append(suma_error / total)

    print(
        epoca + 1,
        errores_entrenamiento[-1],
        errores_validacion[-1],
        aciertos / total
    )


# Guardar los pesos aprendidos. Para cargarlos se necesita la misma red.
torch.save(modelo.state_dict(), "cap_08_pesos.pt")
print("Pesos guardados en cap_08_pesos.pt")


# Comparar las pérdidas para observar un posible sobreajuste.
# La de entrenamiento se acumula mientras los pesos van cambiando.
epocas = range(1, 101)
plt.plot(epocas, errores_entrenamiento, label="Entrenamiento")
plt.plot(epocas, errores_validacion, label="Validación")
plt.xlabel("Época")
plt.ylabel("Entropía cruzada")
plt.legend()
plt.grid()
plt.show()
