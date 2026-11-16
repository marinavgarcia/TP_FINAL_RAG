
# IMPORTAR LIBRERIAS
from pathlib import Path
import gradio as gr
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import re

# =========================================================
# CONFIGURACIÓN GENERAL 
# =========================================================

MODELO_GENERADOR = "granite4:latest"
MODELO_EMBEDDINGS = "bge-m3"

CARPETA_DATOS = Path("datos")
CARPETA_INDICE = "chroma_db_huertas"
NOMBRE_COLECCION = "huertas_rag"

K = 5 # Cantidad de fragmentos a buscar por pregunta

UMBRAL= 0.90 #Distancia máxima aceptada(más baja= mas parecido)
# ---------------------------------------------------------
# Solución para Windows: procesa de a 1 para que Ollama
# no se trabe
# ---------------------------------------------------------

class SafeOllamaEmbeddings(OllamaEmbeddings):

    def embed_documents(self, texts):
        embeddings = []
        total = len(texts)

        print(f"\nCreando embeddings para {total} fragmentos en total...")

        for i, text in enumerate(texts, 1):

            # Usamos super() para llamar a Ollama directamente
            # y evitar el bucle infinito.
            emb = super().embed_documents([text])[0]

            embeddings.append(emb)

            # Muestra el avance en la consola cada 50 fragmentos
            if i % 50 == 0 or i == total:
                print(f" [+] Procesados {i} de {total} fragmentos...")

        return embeddings
    

def limpiar(texto):
    texto = texto.replace("ﬁ", "fi").replace("ﬂ", "fl")
    texto = re.sub(r"-\n(\w)", r"\1", texto)        # palabras cortadas con guion
    texto = re.sub(r"\n[ \t]+\n", " ", texto)       # una palabra por renglón
    texto = re.sub(r"[ \t]{2,}", " ", texto)        # espacios repetidos
    return texto


    
# =========================================================
# PASO 1: Cargar los archivos PDF
# =========================================================

archivos_pdf = list(CARPETA_DATOS.glob("*.pdf"))

print(f"Archivos PDF encontrados: {len(archivos_pdf)}")

if not archivos_pdf:
    raise FileNotFoundError(
        "No se encontraron archivos PDF en la carpeta 'datos'."
    )

documentos = []

for pdf in archivos_pdf:

    cargador = PyPDFLoader(str(pdf))
    paginas = cargador.load()

    for p in paginas:
        p.page_content = limpiar(p.page_content)

    documentos.extend(paginas)

print(f"Total de páginas cargadas: {len(documentos)}")


# =========================================================
# PASO 2: Cortar los textos en fragmentos (chunks)
# =========================================================

divisor = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)

fragmentos = divisor.split_documents(documentos)

print(f"Total de fragmentos creados: {len(fragmentos)}")

print(fragmentos[10].page_content[:300])
# =========================================================
# PASO 3: Guardar los vectores en ChromaDB con bge-m3
# =========================================================

embeddings = SafeOllamaEmbeddings(
    model=MODELO_EMBEDDINGS,
    base_url="http://localhost:11434"
)

vector_store = Chroma(
    collection_name=NOMBRE_COLECCION,
    embedding_function=embeddings,
    persist_directory=CARPETA_INDICE
)

# Si la carpeta ya existe, la carga directamente
# sin volver a procesar los documentos.

if len(vector_store.get()["ids"]) == 0:

    vector_store.add_documents(fragmentos)

    print(
        f'\n¡Base de datos vectorial creada en "{CARPETA_INDICE}"!'
    )

else:

    print(
        f'\n¡Base de datos cargada directamente desde "{CARPETA_INDICE}"!'
    )


# =========================================================
# PASO 4: Armar la cadena RAG (Prompt + Granite 4)
# =========================================================

plantilla = ChatPromptTemplate.from_messages([

    (
        "system",
        "Sos un asistente experto en huertas agroecológicas. "
        "Respondé la pregunta usando ÚNICAMENTE el contexto "
        "provisto de los manuales. Si la información no está "
        "en el contexto, indicá: No encuentro esa información "
        "en los documentos."
    ),

    (
        "human",
        "Contexto:\n{contexto}\n\nPregunta: {pregunta_usuario}"
    )

])


llm = ChatOllama(
    model=MODELO_GENERADOR,
    base_url="http://localhost:11434",
    temperature=0
)


cadena_rag = plantilla | llm | StrOutputParser()


# =========================================================
# PASO 5: Función de consulta e Interfaz Gradio
# =========================================================

def consultar(pregunta_usuario: str):

    if not pregunta_usuario.strip():

        return (
            "Por favor escribí una pregunta.",
            "No se buscaron fuentes."
        )

    hallazgos = vector_store.similarity_search_with_score(
        query=pregunta_usuario,
        k=K
    )
    hallazgos = [h for h in hallazgos if h[1] <= UMBRAL]

    if not hallazgos:
        return (
            "No encuentro esa información en los documentos.",
            "No se encontraron fragmentos suficientemente relevantes."
        )
    contexto = ""
    fuentes = ""

    for numero, (doc, distancia) in enumerate(hallazgos, start=1):

        fuente = Path(
            doc.metadata.get("source", "desconocido")
        ).name

        pagina = doc.metadata.get("page", 0) + 1

        referencia = f"{fuente}, página {pagina}"

        contexto += (
            f"[{numero}] {referencia}\n"
            f"{doc.page_content}\n\n"
        )

        fuentes += (
            f"**[{numero}] {referencia}** "
            f"(distancia: {distancia:.3f})\n\n"
        )

        fuentes += (
            f"```text\n"
            f"{doc.page_content}\n"
            f"```\n\n"
        )

    respuesta = cadena_rag.invoke({

        "contexto": contexto,
        "pregunta_usuario": pregunta_usuario

    })

    return respuesta, fuentes


interfaz = gr.Interface(

    fn=consultar,

    inputs=gr.Textbox(
        label="Pregunta sobre los manuales de huerta",
        placeholder="¿Cómo preparar el compost o sembrar lechuga?"
    ),

    outputs=[

        gr.Textbox(
            label="Respuesta generada (Granite 4)"
        ),

        gr.Markdown(
            label="Fuentes y fragmentos consultados"
        )

    ],

    title="Asistente RAG de Huertas Agroecológicas",

    description=(
        "Consultá los manuales de huerta en tiempo real "
        "con un RAG local totalmente privativo."
    )

)


# =========================================================
# EJECUCIÓN PRINCIPAL
# =========================================================

if __name__ == "__main__":

    interfaz.launch()