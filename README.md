# AIOps para APIs y microservicios: clasificación de causa raíz y acción de resolución con ML + recomendaciones con LLM

Proyecto de la Maestría en Ingenieria y Analitica de datos (MIAD), Universidad de Bogotá Jorge Tadeo Lozano.
**Autor:** Brandon David Rambauth Céspedes

## Descripción

Un incidente en una API o microservicio exige saber **qué lo causó** y **qué hacer para resolverlo**. Este proyecto automatiza ese diagnóstico en dos capas:

1. **Capa de aprendizaje supervisado.** Clasifica dos variables objetivo del dataset AFID (*API Failure Incident Dataset*):
   - `root_cause`: causa raíz del incidente.
   - `resolution_action`: acción de resolución recomendada.
2. **Capa de LLM (Gemini).** Toma el diagnóstico del modelo y redacta una recomendación operativa para el ingeniero de guardia. Incluye un marco de evaluación propio (ver más abajo).

## Contenido del repositorio

| Archivo | Descripción |
|---|---|
| `Proyecto_MIAD_final.ipynb` | Notebook completo y documentado (EDA, auditoría de fuga, 12 modelos, estadística, balanceo, explicabilidad, LLM). |
| `api_error_f2.csv` | Dataset de trabajo: 50 000 registros y 22 columnas. Se incluye aquí porque ya no está disponible en Kaggle. |
| `Paper_MIAD_resultados.pdf` | Artículo con la metodología, resultados y discusión. |
| `README.md` | Este documento. |

> Si el nombre del CSV o del PDF en tu repositorio es distinto, ajusta la tabla.

## Metodología

1. **Auditoría de fuga de información.** V de Cramér corregida por sesgo y pureza para detectar variables que revelan el objetivo. Se evalúan conjuntos de variables A, B y C, más un conjunto de diagnóstico D que añade la causa real.
2. **Pipelines sin fuga.** `Pipeline` + `ColumnTransformer`: `OrdinalEncoder` para árboles; `OneHot` + `StandardScaler` para modelos lineales, KNN, SVC y MLP.
3. **12 modelos más una línea base (Dummy):** Regresión Logística, KNN, SVC (RBF), Árbol de Decisión, Random Forest, Extra-Trees, Gradient Boosting, HistGradientBoosting, AdaBoost, MLP, XGBoost y LightGBM.
4. **Búsqueda de hiperparámetros** con `RandomizedSearchCV`, métrica macro-F1, validación cruzada estratificada con particiones idénticas para todos los modelos. La configuración por defecto compite como candidata.
5. **Comparación estadística rigurosa** (basada en Bravo-Ortíz et al., 2025, *PeerJ Computer Science*): prueba de Friedman, Wilcoxon con corrección de Holm, t corregida de Nadeau-Bengio y prueba de Levene.
6. **Regla de selección.** Se elige el modelo estadísticamente equivalente al mejor (Wilcoxon-Holm p ≥ 0,05 y dentro de 1 error estándar) con menor tiempo de entrenamiento.
7. **Evaluación final única** en el conjunto de prueba, con intervalo de confianza bootstrap, importancia por permutación y SHAP.
8. **Manejo del desbalance:** `class_weight`, ROS, SMOTENC y RUS dentro de `ImbPipeline`.

## Resultados principales

Evaluación en el conjunto de prueba (una sola vez), con IC 95 % bootstrap:

| Objetivo | Modelo seleccionado | macro-F1 | IC 95 % | Exactitud | macro-F1 Dummy |
|---|---|---|---|---|---|
| Causa raíz | XGBoost | 0,825 | [0,805; 0,843] | 0,830 | 0,009 |
| Acción de resolución | XGBoost | 0,401 | [0,376; 0,421] | 0,408 | 0,002 |

- **Causa raíz:** los siete mejores modelos quedan entre 0,823 y 0,832 de macro-F1. XGBoost se elige por rendimiento equivalente con solo 1,08 s de entrenamiento.
- **Acción de resolución:** es una tarea mucho más difícil (macro-F1 ≈ 0,40), muy por encima de la línea base pero lejos de ser resuelta solo con las variables disponibles.
- Con pocas particiones de validación cruzada, el valor p mínimo exacto de Wilcoxon es 2/2^k, por lo que Holm no alcanza significancia con 6 particiones. Por eso la regla de selección añade el criterio de 1 error estándar.

Las tablas y figuras completas están en el notebook y en el PDF.

## Evaluación del LLM

El notebook incluye un marco de evaluación de las recomendaciones generadas, con seis dimensiones:

1. **Latencia** y costo.
2. **Alucinación:** heurísticas de verificación y consistencia tipo SelfCheck.
3. **Sesgo ético:** pruebas contrafactuales con Friedman + Holm.
4. **Seguridad:** comandos peligrosos, datos personales (PII) y tasa de éxito de inyección de prompt.
5. **Asertividad y calibración.**
6. **LLM como juez**, validado contra anotadores humanos (Spearman, kappa ponderado).

> **Estado actual:** por límites de cuota de la API de Gemini, la evaluación cuantitativa completa aún no se ha ejecutado. El repositorio incluye una demostración cualitativa. La ejecución completa (`MODO="completo"`) queda como trabajo pendiente y sus resultados se incorporarán al artículo.

## Cómo reproducir

### Entorno
Ejecutado en **Google Colab (plan gratuito)**. Los tiempos de entrenamiento y los límites de cuota dependen de ese entorno compartido. Las librerías (scikit-learn, XGBoost, LightGBM, imbalanced-learn, SHAP, statsmodels, google-genai) se instalan en la primera celda del notebook.

### Pasos
1. Abre `Proyecto_MIAD_final.ipynb` en Google Colab.
2. Sube `api_error_f2.csv` al entorno (o móntalo desde Drive y ajusta la ruta).
3. Ejecuta las secciones 0 a 13 (datos, modelos, estadística, explicabilidad). No requieren API key.
4. Para la sección del LLM (14 en adelante):
   - Guarda tu clave como secreto `GEMINI_API_KEY` en Colab.
   - Elimina la caché incompleta `resultados/llm_gemini-3.5-flash_rapido.joblib` si existe.
   - Usa `MODO="completo"` cuando tengas cuota suficiente. Con `MODO="rapido"` se obtiene una versión reducida.

## Dataset

AFID (*API Failure Incident Dataset*), publicado originalmente en Kaggle y ya no disponible allí. El archivo utilizado contiene 50 000 incidentes y 22 columnas, incluidas las dos variables objetivo `root_cause` y `resolution_action`.

## Referencias principales

- Bravo-Ortíz et al. (2025). *PeerJ Computer Science*, cs-3208. Base metodológica de la comparación estadística de modelos.
- Lista completa en la sección de referencias del notebook y en el PDF.

## Autor

**Brandon David Rambauth Céspedes**
Maestría en Inteligencia Artificial (MIAD), Universidad de Bogotá Jorge Tadeo Lozano
