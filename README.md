# Asistente Agroecológico de Huertas (RAG)

**Autora:** Marina Garcia
**Materia:** Procesamiento del Lenguaje Natural (PLN) — Laboratorio de PLN
**Institución:** Tecnicatura Superior en Ciencias de Datos e Inteligencia Artificial — IFTS N.º 24
**Docente:** Matías Barreto (2026)

---

## 1. Problema y destinatario

**Este sistema permite que cualquier persona interesada en hacer una huerta agroecológica consulte cómo preparar el suelo, hacer compost y fertilizantes naturales, controlar plagas sin químicos, y cuándo y cómo sembrar sus hortalizas, en un conjunto de manuales agroecológicos, para contar con un espacio de consulta que acompañe el manejo de su huerta.**

Es un espacio de consulta para llevar adelante una huerta: tratamientos naturales de plagas, fertilizantes naturales, preparación del suelo, fechas de siembra de cada hortaliza, clima y riego.

### Qué puede y qué no puede responder

* **Qué intenta responder:** los temas anteriores, según lo que traen los 4 manuales indexados.
* **Qué responde con dificultad:** preguntas que juntan datos de muchos cultivos, como "qué se puede sembrar en noviembre" (ver sección 8).
* **Qué no responde:** consultas ajenas a la huerta o datos que no están en los documentos.

El sistema es local: los documentos y las consultas no salen de la computadora.

---

## 2. Corpus de documentos

Cuatro publicaciones técnicas sobre huerta agroecológica, guardadas como copia exacta en `datos/`. Los cuatro son PDF con texto seleccionable (no escaneados), aunque algunas partes se extraen con defectos (ver sección 8).

### Documento 1: Guía Básica para la Planificación y Manejo Agroecológico de Cultivos

- **Autor u organismo:** Ministerio de la Producción de la Provincia de Santa Fe.
- **Origen:** Gobierno de la Provincia de Santa Fe.
- **Año de publicación:** 2019.
- **Documento:** (https://www.santafe.gov.ar/index.php/web/content/download/254524/1339209/file/Guia%20para%20el%20manejo%20agroec.%20de%20cultivos.pdf)
- **Licencia:** No identificada. No se encontró una declaración de licencia abierta verificable.

### Documento 2: Manual de Huerta (Programa Provincial Miryam Kita Gorban)

- **Autor u organismo:** Ministerio de Ambiente de la Provincia de Buenos Aires. Programa Provincial Miryam Kita Gorban.
- **Origen:** Gobierno de la Provincia de Buenos Aires.
- **Fecha / versión:** Enero de 2025 (según el nombre del archivo).
- **Documento:** (https://www.ambiente.gba.gob.ar/pdfs/009_Manual_huertas_ene25.pdf).
- **Licencia:** Creative Commons Atribución-NoComercial-CompartirIgual 4.0 Internacional (CC BY-NC-SA 4.0).

### Documento 3: Manual de cultivos para la Huerta Orgánica Familiar

- **Autor:** Enrique David Goites. **Edición literaria:** Janine Schonwald.
- **Organismo editor:** Instituto Nacional de Tecnología Agropecuaria (INTA), Ediciones INTA. **Programa:** Pro Huerta (INTA/MDS).
- **Fecha / versión:** Primera edición, Buenos Aires, 2008.
- **Documento:** (https://www.argentina.gob.ar/sites/default/files/2023/08/script-tmp-manual_de_cultivos_para_la_huerta_organica_familiar_-.pdf).
- **Licencia:** Creative Commons Atribución 4.0 Internacional (CC BY 4.0).

### Documento 4: Mi casa, mi huerta. Técnicas de agricultura urbana

- **Autores:** Janine Schonwald y Francisco José Pescio. **Edición literaria:** Daniela Novelli.
- **Organismo editor:** Instituto Nacional de Tecnología Agropecuaria (INTA), Ediciones INTA. **Programa:** Pro Huerta (INTA y Ministerio de Desarrollo Social de la Nación).
- **Fecha / versión:** Primera edición, Buenos Aires, 2015. **ISBN:** 978-987-521-591-7.
- **Documento:** (https://www.magyp.gob.ar/sitio/areas/nuestra-huerta/pdf/mi-casa-mi-huerta.pdf).
- **Licencia:** Creative Commons Atribución-NoComercial-CompartirIgual 4.0 Internacional (CC BY-NC-SA 4.0).

Los cuatro documentos se usan como fuentes de consulta del asistente.

---

## 3. Arquitectura del pipeline

```text
PDF en datos/ → PyPDFLoader (fuente + página) → limpiar() → fragmentos (800 / 150)
→ embeddings bge-m3 (Ollama) → Chroma (chroma_db_huertas/)

pregunta → embedding con el mismo modelo → búsqueda de K = 5 fragmentos
→ descarte de los que superan UMBRAL = 0.9 → contexto con fuente y página
→ prompt → granite4:latest (Ollama, temperatura 0) → respuesta + fragmentos (Gradio)
```

1. **Ingesta:** `PyPDFLoader` lee cada PDF página por página y conserva `source` y `page`. Se cargan 360 páginas. La página que muestra la app es la del visor de PDF; el número impreso en la hoja puede ser menor (en `mi-casa-mi-huerta.pdf`, una unidad menos; en el manual de cultivos, dos).
2. **Limpieza (`limpiar`):** une palabras cortadas con guion al final de renglón, reemplaza las ligaduras `ﬁ` y `ﬂ`, junta los renglones con una sola palabra y elimina espacios repetidos.
3. **Fragmentación:** `RecursiveCharacterTextSplitter`, `chunk_size = 800` y `chunk_overlap = 150` (miden caracteres). Resultan **831 fragmentos**.
4. **Indexación:** embeddings con `bge-m3` por Ollama, guardados en Chroma. Se procesa de a un fragmento porque en Windows Ollama se trababa con lotes. Al iniciar, la app imprime un fragmento para inspeccionar el texto extraído.
5. **Recuperación:** `similarity_search_with_score` con `K = 5` (empecé con 3 y lo subí a 5 durante las pruebas). 
6. **Filtro por umbral:** el umbral (0.9) es la distancia máxima aceptada entre la pregunta y un fragmento. Los más lejanos se descartan para que el modelo no reciba contexto sin relación con la pregunta. Si no queda ninguno, la app responde que no encuentra información y no consulta al modelo.
7. **Generación:** `granite4:latest`, temperatura 0, con instrucción de usar solo el contexto y de abstenerse si falta evidencia.
8. **Interfaz:** Gradio (`app.py`) muestra la respuesta y, por separado, los fragmentos usados con archivo, página y distancia.

### Versiones usadas

| Componente | Versión |
| --- | --- |
| Python | 3.12 |
| Ollama (programa) | 0.40.16 |
| granite4:latest | ID `4235724a127c`, 2.1 GB |
| bge-m3:latest | ID `790764642607`, 1.2 GB |
| gradio | 6.31.0 |
| langchain-community | 0.4.2 |
| langchain-core | 1.7.7 |
| langchain-text-splitters | 1.1.3 |
| langchain-chroma | 1.1.0 |
| langchain-ollama | 1.1.0 |
| chromadb | 1.5.9 |
| ollama (cliente Python) | 0.6.3 |
| pypdf | 7.0.1 |

---

## 4. Instalación y ejecución local

**Requisitos:** Python 3.12, [uv](https://docs.astral.sh/uv/) y [Ollama](https://ollama.com/) instalado y en ejecución.

**1. Descargar los modelos de Ollama**

```bash
ollama pull granite4:latest
ollama pull bge-m3
```

**2. Crear y activar el entorno con uv**

Windows (PowerShell):

```powershell
uv venv --python 3.12
.venv\Scripts\Activate.ps1
```

Linux / macOS:

```bash
uv venv --python 3.12
source .venv/bin/activate
```

**3. Instalar las dependencias**

```bash
uv pip install -r requirements.txt
```

**4. Iniciar la aplicación**

```bash
python app.py
```

La primera vez, la app lee los PDF de `datos/` y crea el índice (`chroma_db_huertas/`). Tarda unos minutos porque calcula los embeddings de los 831 fragmentos. Las siguientes veces carga el índice sin recalcular. La interfaz se abre en la dirección que muestra la consola, normalmente `http://127.0.0.1:7860` (si el puerto está ocupado, Gradio usa otro).

---

## 5. Reindexación

Si algo de lo que se indexó cambia, hay que **borrar a mano la carpeta `chroma_db_huertas/`**; si no, el sistema sigue usando el índice viejo sin avisar. Hay que reindexar cuando cambian:

* los documentos de `datos/`,
* `chunk_size` o `chunk_overlap`,
* la función `limpiar`,
* o el modelo de embeddings (no se deben mezclar vectores de modelos distintos).

No hace falta reindexar al cambiar `K`, `UMBRAL` o el prompt, porque se usan solo al consultar.

Pasos: detener la app (Ctrl+C), borrar `chroma_db_huertas/` y ejecutar de nuevo `python app.py`. La carpeta no se sube a GitHub: se reconstruye desde `datos/`.

---

## 6. Pruebas

Los registros crudos de Gradio (botón *Flag*) están en `pruebas/flags_unificado.csv`, con una columna `version`:

* `base`: K = 3, sin limpieza ni umbral.
* `mejorada`: K = 5, `UMBRAL = 0.9` y `limpiar` con ligaduras y guiones.
* `limpieza_ampliada`: lo anterior más la unión de renglones sueltos y espacios repetidos (831 fragmentos).

Gradio generó varios archivos CSV durante las pruebas (crea uno nuevo si el encabezado del existente no coincide), y se unieron en ese archivo.

### 6.1 Cinco casos de evaluación (versión `limpieza_ampliada`)

| # | Pregunta | Fuente esperada | Fragmentos recuperados (archivo, página PDF, distancia) | Respuesta del modelo | Qué observé |
| - | -------- | --------------- | ------------------------------------------------------- | -------------------- | ----------- |
| R1 | `cómo preparar compost` | Guía de Manejo Agroecológico (compostaje) | Guía p. 40 (0.681), p. 39 (0.683), p. 40 (0.783), p. 40 (0.791); mi-casa-mi-huerta p. 40 (0.839) | Pasos de armado de la compostadora: ubicación, base de ramas, capas, riego, fitoestimulantes, poste de aireación | Responde bien con la fuente correcta. Dos fragmentos de la p. 40 se solapan por el `chunk_overlap`. |
| R2 | `que temperatura requiere el cultivo del tomate` | Manual de cultivos, ficha del tomate | manual_de_cultivos p. 99 (0.587), p. 83 (0.631), p. 87 (0.835), p. 37 (0.843); Manual_huertas_ene25 p. 19 (0.807) | Germinación entre 15 y 24 °C; mejor desarrollo entre 21 y 26 °C | **Recuperación correcta, generación incorrecta.** El primer fragmento (p. 99) dice que la temperatura óptima de germinación del tomate es de 28-30 °C, y se comprobó en el original (p. 99 del archivo, 97 impresa). Los valores que dio el modelo vienen del fragmento de la p. 87, que corresponde a otro cultivo (el capítulo del tomate empieza en la p. 99). La respuesta fue la misma en todas las corridas. El fragmento de Manual_huertas_ene25 p. 19 tiene partes ilegibles (ver 8). |
| R3 | `como hacer una huerta vertical` | mi-casa-mi-huerta, sección "Huertas verticales" | mi-casa-mi-huerta p. 23 (0.539), p. 19 (0.632), p. 17 (0.759), p. 19 (0.760); Manual_huertas_ene25 p. 30 (0.797) | Pared soleada, caños de desagüe de 4" como eje, tramos laterales, cantero de madera o pallets en la base, sustrato, frutillera con envases reciclados | Mejoró con la limpieza ampliada (ver 6.2). Los pasos finales provienen de la frutillera (p. 19), una estructura vecina. Comprobado en el original (p. 23): el eje de caños de 4", los tramos laterales y el cantero de pallets coinciden. Los pasos de armado están en la p. 24, que no se recuperó en ninguna prueba. |
| S1 | `quien es le Presidente de Argentina` | No figura en el corpus | Ninguno pasó el umbral | "No encuentro esa información en los documentos." | Se abstuvo por el umbral: el modelo no fue consultado. No prueba por sí solo que el prompt funcione. |
| S2 | `cual es el corte mas caro de la vaca` | No figura en el corpus | Ninguno pasó el umbral | "No encuentro esa información en los documentos." | Igual que S1. |

### 6.2 Comparación entre etapas

| Pregunta | Etapa `base` | Etapa `mejorada` | Etapa `limpieza_ampliada` |
| -------- | ------------ | ---------------- | ------------------------- |
| `como hacer una huerta vertical` | No se probó | El fragmento de la p. 23 tenía una palabra por renglón. El modelo respondió con los pasos de "Bateas de cultivo" (p. 17), una técnica distinta | Fragmento legible; la respuesta usa el contenido de la huerta vertical |
| `como preparar materia organica` (y variante) | "No encuentro" | Según cómo estaba escrita la pregunta: con `como puedo preparar la materia organica` respondió con el extracto de ortiga, y con `como preparar materia organica` con pasos que empiezan por recolectar materias primas orgánicas (podas, hojas, ramas, cáscaras) | Respondió con la preparación de un biofertilizante (págs. 74, 65 y 70), no con compost |
| `que se puede sembrar en noviembre` | "No encuentro" (fragmentos de zapallito, pepino y un cultivo de invierno, 0.944 a 0.989) | "No encuentro", sin fragmentos | Igual |
| `cómo preparar compost` | No se probó | Responde bien | Responde bien (mismas fuentes y distancias) |

---

## 7. Mejoras y decisiones

| Cambio | Qué hace | Por qué |
| ------ | -------- | ------- |
| `K` de 3 a 5 | Trae más fragmentos por pregunta | Lo subí para ver si mejoraban las respuestas, porque a veces la respuesta está repartida en varios fragmentos. Me preocupaba que, al traer más, empezaran a venir fragmentos que no tienen nada que ver con la pregunta (ruido). |
| `UMBRAL = 0.9` | Descarta los fragmentos que están muy lejos de la pregunta | Fue la solución a ese ruido: si un fragmento está demasiado lejos de la pregunta, no se le pasa al modelo. |
| `limpiar()` (versión 1) | Une palabras cortadas con guion y reemplaza ligaduras | PyPDFLoader deja ese ruido en el texto |
| `limpiar()` (ampliada) | Junta los renglones con una sola palabra y elimina espacios repetidos | En `mi-casa-mi-huerta.pdf` algunos fragmentos salían con una palabra por renglón y el modelo mezclaba técnicas |
| Reindexado | Se regeneró `chroma_db_huertas/` en cada cambio de `limpiar` | El índice viejo tenía el texto sin limpiar |

**Método.** Fui corrigiendo los fallos a medida que aparecían en las pruebas: empecé con fragmentos de 500 caracteres y `K = 3`, sin limpieza ni umbral, y fui ajustando una cosa por vez cuando fue posible. Limitación del método: en la etapa `mejorada` se cambiaron tres cosas juntas (`K`, `UMBRAL` y la primera versión de `limpiar`), por lo que no se puede atribuir el efecto a cada una por separado. La etapa `limpieza_ampliada` fue un cambio aislado y permite comparar antes y después.

**Tamaño de fragmento.** Con `chunk_size = 500` se observó que el título de una preparación quedaba en el fragmento anterior a sus instrucciones. Se subió a `chunk_size = 800` y `chunk_overlap = 150` para evitarlo. El efecto no se midió de forma sistemática. Este cambio no resolvió las preguntas que fallaban por otros motivos (ver sección 8).

**Codificación.** Al abrir el CSV de flags con una codificación equivocada se veía `PreparaciÃ³n`. Se comprobó imprimiendo un fragmento en la consola que el texto indexado es correcto (con tildes y ñ); era un problema de visualización del CSV.

---

## 8. Limitaciones observadas

1. **Depende de cómo se escribe la pregunta.** "Materia orgánica" es ambigua (es lo que se composta, no el compost) y los manuales hablan de compost y abono. El sistema solo compara los números de la pregunta con los de cada fragmento, sin el contexto que una persona sobreentiende. Dos redacciones casi iguales trajeron fragmentos y respuestas distintas, mientras que `cómo preparar compost` funcionó bien. Mejora posible: reescribir la pregunta con los términos de los manuales antes de buscar.
2. **"Qué se puede sembrar en noviembre" no se resuelve.** El dato existe, pero repartido en la ficha de cada cultivo, y la búsqueda no puede filtrar por mes. Ese "No encuentro" es un fallo de la búsqueda, no una abstención correcta. Mejora posible: una tabla de cultivos con sus meses de siembra, consultada por código.
3. **Algunos PDF se leen mal.** Hay palabras con espacios adentro (`la huer ta`), partes con símbolos ilegibles (`Manual_huertas_ene25.pdf`, p. 19; haría falta OCR) y tablas que pierden su estructura (la de variedades de tomate). `limpiar` no corrige esos casos. Además, la p. 24 de `mi-casa-mi-huerta.pdf`, con los pasos de armado de la huerta vertical, no apareció en ninguna prueba.
4. **Los fragmentos se cortan.** El solapamiento repite texto y ocupa varios de los `K` lugares. Y como se corta página por página, una explicación que sigue en la hoja siguiente queda en dos fragmentos y, según la pregunta, puede llegar solo uno (por ejemplo, el biofertilizante de las págs. 73 y 74 de la Guía, en la versión base: una pregunta trajo solo la p. 74 y otra trajo las dos).
5. **El modelo puede equivocarse y la abstención no está garantizada.** En S1 y S2 se abstuvo el umbral y no el modelo, y no probé una pregunta de huerta ausente del corpus que pase el umbral. Granite puede mezclar fragmentos (en la huerta vertical mezcló dos técnicas antes de la limpieza ampliada; en el tomate usó la temperatura de otro cultivo) o cambiar palabras ("carpintería" por "carpida"). La fuente que muestra la app es lo que recibió el modelo, no prueba que lo haya interpretado bien, y la distancia no mide la calidad del fragmento.
6. **No comprobé si el cuarto manual mejoró las respuestas.** Sumé `Manual_huertas_ene25.pdf` pensando que más información mejoraría las respuestas, pero lo asumí sin comparar el sistema con y sin ese documento. Vi que aportó un fragmento sin relación con la pregunta (p. 30, en la huerta vertical) y otro con partes ilegibles (p. 19, en el tomate). Eso no prueba que empeore, pero muestra que importa más la calidad del texto que la 
7 ** No comprobé si el cuarto manual mejoró las respuestas.** El trabajo pide tres documentos y sumé un cuarto (`Manual_huertas_ene25.pdf`) pensando que más información mejoraría la calidad de las respuestas. Lo asumí como cierto sin plantearlo como hipótesis y sin comparar el sistema con y sin ese documento. Lo que sí vi es que ese manual aportó un fragmento sin relación con la pregunta (en `como hacer una huerta vertical`, la p. 30, sobre sembrar "a chorrillo") y otro con partes ilegibles (en la pregunta del tomate, la p. 19). Eso no demuestra que empeore las respuestas, pero muestra que sumar documentos no alcanza: importa también la calidad del texto que entra.


**Trabajo futuro:** ordenar el texto antes de indexarlo .

**Hipótesis para una segunda versión (no probada):** un paso previo en el que Granite lea los textos, deje lo útil, los reescriba claros y les ponga una etiqueta de tema (plagas, siembra, suelo, compost, clima), guardando el texto original y su página. Con eso se armaría un RAG nuevo, donde la pregunta se busque en su tema. Se evaluaría con las mismas preguntas de este trabajo, con y sin ese paso, y con tres y con cuatro documentos.

---

## 9. Estructura del repositorio

```text
TP_FINAL_RAG/
├── .gitignore
├── README.md
├── requirements.txt
├── app.py
├── pruebas/
│   └── flags_unificado.csv
└── datos/
    ├── Guia_Manejo_Agroecologico.pdf
    ├── Manual_huertas_ene25.pdf
    ├── manual_de_cultivos_para_la_huerta_organica_familiar_-.pdf
    └── mi-casa-mi-huerta.pdf
```

`.gitignore` excluye `.venv/`, `.env`, `.gradio/`, `chroma_db_huertas/` y `__pycache__/`. El índice se reconstruye desde `datos/` como se explica en la sección 5.

---

## 10. Uso de inteligencia artificial

* **Modelos dentro del sistema:** `granite4:latest` (generación) y `bge-m3` (embeddings), ambos locales por Ollama.
* **Asistente usado para construir el proyecto:** Claude (Anthropic), en noviembre de 2026.
* **Para qué lo usé:**
  * Entender por qué Ollama se trababa en Windows y pedir el código para procesar de a un fragmento (`SafeOllamaEmbeddings`).
  * Entender por qué fallaban las respuestas, a partir de los fragmentos recuperados, y por qué fallaban también algunas de mis preguntas.
  * Subir `K` a 5 (en la materia se usa 3) . A mí me preocupó que un `K` mayor trajera ruido, incorporar el umbral como solución.
  * Escribir la función `limpiar` (las dos versiones). La materia marca la limpieza como una consideración importante; yo no la hice de entrada y lo noté con las respuestas.
  * Ordenar y reescribir mis ideas de forma clara en este README.
* **Qué verifiqué personalmente:** que el código corriera en mi entorno Windows; revisé los fragmentos recuperados en la consola y en Gradio, reindexé después de cada cambio, repetí las preguntas de prueba y comprobé las respuestas contra los PDF. Para incorporar mejoras, comparé escenarios.
* **Cambios que probé durante el proyecto:** `K` de 3 a 5, el umbral de distancia, la función `limpiar` (dos versiones) y el procesamiento de a un fragmento (`SafeOllamaEmbeddings`), que se incorporó porque en Windows Ollama se trababa al recibir todos los fragmentos juntos.
