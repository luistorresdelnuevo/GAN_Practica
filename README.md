# Proyecto Final: Modelos de Generación de Imagen

Entrega final del módulo de Modelos de Generación de Imagen. Consiste en hacer un finetuning con **LoRA** del modelo [`CompVis/stable-diffusion-v1-4`](https://huggingface.co/CompVis/stable-diffusion-v1-4) usando el dataset de ilustraciones de libros antiguos [`gigant/oldbookillustrations`](https://huggingface.co/datasets/gigant/oldbookillustrations), para que el modelo genere imágenes con ese estilo.

## Contenido del repositorio

```
ProyectoFinal/
├── finetuning_Proyecto_Final.ipynb   # Notebook con todo el proceso (entregable 1)
├── link.txt                          # Enlace al modelo en Hugging Face (entregable 2)
├── Imagen_Antes_Después.png          # Misma imagen antes y después del finetuning (entregable 3)
└── docs/
    └── 1-enunciado_entrega_modelos_generacion_imagen.pdf   # Enunciado de la práctica
```

## Modelo en Hugging Face

La LoRA entrenada está publicada en:
**https://huggingface.co/estaluis/sd-v1-4-oldbookillustrations-lora**

## Resultado

Con el mismo prompt y la misma semilla, el modelo original genera una imagen en tonos sepia y con un trazo sencillo. Con la LoRA aparecen los tonos grises y el rayado típico de los grabados antiguos.

> *"an old book illustration of a knight fighting with animals including a lion, a bear and a wolf."*

![Antes y después del finetuning](ProyectoFinal/Imagen_Antes_Despu%C3%A9s.png)

## Qué hace el notebook

1. Genera una imagen con el modelo original, que sirve de referencia.
2. Carga el dataset y se queda con las columnas `1600px` (imagen) e `info_alt` (descripción). Usa un subconjunto de 800 imágenes.
3. Ajusta las imágenes, que no son cuadradas, a 512×512 con `Resize` + `CenterCrop` para no deformarlas.
4. Añade una LoRA a la UNet y entrena solo esos pesos (r = 6, 5 épocas, batch 3, learning rate 1e-4).
5. Genera la imagen de "después" con el mismo prompt y la misma semilla, y compara las dos.
6. Guarda la LoRA y la sube a Hugging Face.
7. Carga la LoRA desde Hugging Face y la usa para generar.

El entrenamiento se ejecutó en Google Colab con una GPU Tesla T4 y tardó aproximadamente 1 h 30 min.

## Cómo usar el modelo

```python
import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to("cuda")
pipe.load_lora_weights("estaluis/sd-v1-4-oldbookillustrations-lora")

image = pipe("an old book illustration of a knight fighting with animals including a lion, a bear and a wolf.").images[0]
image.save("resultado.png")
```
