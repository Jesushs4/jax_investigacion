# **Investigación de Jax**

## **¿Qué es?**

Es una biblioteca de Python diseñada por Google para computación numérica de alto rendimeinto y aprendizaje automático a gran escala.

---

## **Principales características**

* API: Utiliza una **API** similar a NumPy, facilitando el trabajo de los datos.

* XLA: Utiliza **XLA** (*Accelerated Linear Algebra*) para compilar y ejecutar programas de manera eficiente en CPUs, GPUs y TPUs

* Diferenciación automática: Puede diferenciar automáticamente funciones nativas de Python y NumPy.

* Vectorización automática (vmap): La transformación `vmap` permite convertir automáticamente funciones que operan sobre una sola muestra para que operen sobre lotes de datos.

* Paralelización (pmap): Permite distribuir y ejecuta cálculos en múltiples dispositivos de forma eficiente.

---

## **Ecosistema: librerías implementadas sobre JAX y otras herramientas que se integran bien con esta tecnología.**

### Redes neuronales
* Flax
* Equinox
* Keras

### Optimizadores
* Optax
* Optimistix
* Lineax
* Diffrax

### Carga de datos
* Grain
* TensorFlow Datasets
* Hugging Face Datasets

### Programación probabilística
* Blackjax
* Numpyro
* PyMC

### Modelado probabilístico
* TensorFlow Probability
* Distrax

### Físicas y simulación
* Jax MD
* Brax

### LLMs
* MaxText
* AXLearn
* Levanter
* EasyLM
* Marin

### Otras herramientas
* Orbax
* Chex
