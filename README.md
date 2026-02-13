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

---

## **Comparación de JAX con TensorFlow y PyTorch.**

| Aspecto                    | **JAX**                                             | **PyTorch**                | **TensorFlow**                |
| -------------------------- | --------------------------------------------------- | -------------------------- | ----------------------------- |
| **Modelo de ejecución**    | Funcional, basado en *tracing*                      | Imperativo (eager)         | Grafo + eager                 |
| **Definición del modelo**  | Funciones puras                                     | Clases (`nn.Module`)       | Clases (`tf.keras.Model`)     |
| **Autodiferenciación**     | Transformaciones de programa (`grad`, `vjp`, `jvp`) | Tape dinámico              | Tape dinámico                 |
| **Compilación**            | XLA por defecto (`jit`)                             | Opcional (`torch.compile`) | Opcional (XLA)                |
| **Mutabilidad**            | No mutable                                          | Mutable                    | Mutable                       |
| **Vectorización**          | Explícita (`vmap`)                                  | Implícita / manual         | Implícita                     |
| **Paralelismo**            | SPMD nativo (`pjit`)                                | DDP / FSDP                 | Estrategias (`tf.distribute`) |
| **TPU**                    | Soporte nativo                                      | Soporte parcial            | Soporte nativo                |
| **Debugging**              | Limitado bajo `jit`                                 | Directo                    | Intermedio                    |
| **Flexibilidad dinámica**  | Media                                               | Muy alta                   | Media                         |
| **Ecosistema ML**          | Más reducido                                        | Muy amplio                 | Muy amplio                    |
| **Producción**             | Posible, menos estándar                             | Común                      | Muy común                     |
| **Serialización / export** | Limitada                                            | Buena                      | Muy buena                     |

---

## **Ejemplo práctico**
[Google Colab – Ejemplo práctico](https://colab.research.google.com/drive/1mxeUKAeZHpbvIypMwdKl_zqBGGoSvALL?usp=sharing)

---

## **Fuentes**
https://eiposgrados.com/blog-python/jax-machine-learning/

https://es.eitca.org/artificial-intelligence/eitc-ai-gcml-google-cloud-machine-learning/google-cloud-ai-platform/introduction-to-jax/examination-review-introduction-to-jax/what-is-jax-and-how-does-it-speed-up-machine-learning-tasks/

https://docs.jax.dev/en/latest/

https://pytorch.org/docs/stable/index.html

https://www.tensorflow.org/guide
