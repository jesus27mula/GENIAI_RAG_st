from __future__ import annotations

import hashlib
import math
import os
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.tools import BaseTool, tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import NotRequired, TypedDict


# =============================================================================
# Configuración
# =============================================================================


def _env_int(name: str, default: int, *, minimum: int = 0) -> int:
    raw_value = os.getenv(name)

    if raw_value is None or not raw_value.strip():
        return default

    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"La variable {name} debe ser un entero; se recibió {raw_value!r}."
        ) from exc

    if value < minimum:
        raise ValueError(
            f"La variable {name} debe ser >= {minimum}; se recibió {value}."
        )

    return value


def _env_float(name: str, default: float) -> float:
    raw_value = os.getenv(name)

    if raw_value is None or not raw_value.strip():
        return default

    try:
        return float(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"La variable {name} debe ser numérica; se recibió {raw_value!r}."
        ) from exc


def _resolve_project_path(project_root: Path, raw_value: str | None, default: Path) -> Path:
    if raw_value is None or not raw_value.strip():
        return default.resolve()

    candidate = Path(raw_value).expanduser()

    if not candidate.is_absolute():
        candidate = project_root / candidate

    return candidate.resolve()


@dataclass(frozen=True, slots=True)
class FiscalSettings:
    project_root: Path
    vectorstore_dir: Path
    collection_name: str
    embedding_model: str
    chat_model: str
    retrieval_k: int
    priority_k_per_source: int
    max_doc_chars: int
    max_context_chars: int
    max_output_tokens: int
    thinking_budget: int
    temperature: float
    timeout_seconds: int

    @classmethod
    def from_env(cls) -> "FiscalSettings":
        project_root = Path(__file__).resolve().parents[1]
        default_vectorstore = project_root / "data" / "vectorstore" / "chroma"

        return cls(
            project_root=project_root,
            vectorstore_dir=_resolve_project_path(
                project_root,
                os.getenv("VECTORSTORE_DIR"),
                default_vectorstore,
            ),
            collection_name=os.getenv(
                "CHROMA_COLLECTION_NAME",
                "rag_fiscal_startups",
            ).strip(),
            embedding_model=os.getenv(
                "EMBEDDING_MODEL",
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            ).strip(),
            chat_model=os.getenv("CHAT_MODEL", "gemini-2.5-flash").strip(),
            retrieval_k=_env_int("RETRIEVAL_K", 4, minimum=1),
            priority_k_per_source=_env_int(
                "PRIORITY_K_PER_SOURCE",
                2,
                minimum=1,
            ),
            max_doc_chars=_env_int("MAX_DOC_CHARS", 1_400, minimum=300),
            max_context_chars=_env_int(
                "MAX_CONTEXT_CHARS",
                5_000,
                minimum=1_000,
            ),
            max_output_tokens=_env_int(
                "MAX_AGENT_OUTPUT_TOKENS",
                500,
                minimum=100,
            ),
            thinking_budget=_env_int("THINKING_BUDGET", 0, minimum=0),
            temperature=_env_float("MODEL_TEMPERATURE", 0.2),
            timeout_seconds=_env_int("MODEL_TIMEOUT_SECONDS", 60, minimum=1),
        )


@dataclass(slots=True)
class RetrievalResources:
    settings: FiscalSettings
    embeddings: HuggingFaceEmbeddings
    chroma_client: Any
    vectorstore: Chroma
    collection_count: int


@dataclass(slots=True)
class FiscalRuntime:
    settings: FiscalSettings
    resources: RetrievalResources
    llm: ChatGoogleGenerativeAI
    search_tax_corpus: BaseTool
    graph: Any


# =============================================================================
# Carga de embeddings y Chroma
# =============================================================================


def list_chroma_collection_names(client: Any) -> list[str]:
    names: list[str] = []

    for collection in client.list_collections():
        if isinstance(collection, str):
            names.append(collection)
            continue

        name = getattr(collection, "name", None)
        if name:
            names.append(str(name))

    return names


def load_retrieval_resources(
    settings: FiscalSettings | None = None,
) -> RetrievalResources:
    settings = settings or FiscalSettings.from_env()

    if not settings.vectorstore_dir.exists():
        raise FileNotFoundError(
            "No existe el vectorstore persistido. Copia la carpeta Chroma en:\n"
            f"{settings.vectorstore_dir}"
        )

    embeddings = HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        encode_kwargs={"normalize_embeddings": True},
    )

    chroma_client = chromadb.PersistentClient(
        path=str(settings.vectorstore_dir)
    )

    collection_names = list_chroma_collection_names(chroma_client)

    if settings.collection_name not in collection_names:
        raise RuntimeError(
            f"No existe la colección {settings.collection_name!r}. "
            f"Colecciones disponibles: {collection_names or 'ninguna'}."
        )

    vectorstore = Chroma(
        client=chroma_client,
        collection_name=settings.collection_name,
        embedding_function=embeddings,
        create_collection_if_not_exists=False,
    )

    collection_count = vectorstore._collection.count()

    if collection_count <= 0:
        raise RuntimeError(
            f"La colección {settings.collection_name!r} está vacía."
        )

    return RetrievalResources(
        settings=settings,
        embeddings=embeddings,
        chroma_client=chroma_client,
        vectorstore=vectorstore,
        collection_count=collection_count,
    )


# =============================================================================
# Utilidades de texto y mensajes
# =============================================================================


def normalize_text(text: str | None) -> str:
    if text is None:
        return ""

    normalized = str(text).lower()
    normalized = "".join(
        char
        for char in unicodedata.normalize("NFD", normalized)
        if unicodedata.category(char) != "Mn"
    )
    normalized = re.sub(r"\s+", " ", normalized)

    return normalized.strip()


def message_to_text(message: Any) -> str:
    if message is None:
        return ""

    if isinstance(message, dict):
        content = message.get("content", "")
    else:
        content = getattr(message, "content", message)

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts: list[str] = []

        for block in content:
            if isinstance(block, str):
                parts.append(block)
                continue

            if isinstance(block, dict):
                text = (
                    block.get("text")
                    or block.get("content")
                    or block.get("output")
                )
                if text:
                    parts.append(str(text))
                continue

            text = getattr(block, "text", None)
            if text:
                parts.append(str(text))

        return "\n".join(parts).strip()

    return str(content).strip()


def extract_final_answer(state: dict[str, Any]) -> str:
    messages = state.get("messages", []) or []

    for message in reversed(messages):
        if getattr(message, "type", None) == "ai":
            text = message_to_text(message)
            if text:
                return text

        if isinstance(message, dict) and message.get("role") in {
            "assistant",
            "ai",
        }:
            text = message_to_text(message)
            if text:
                return text

    return ""


def extract_usage_metadata(state: dict[str, Any]) -> dict[str, Any]:
    messages = state.get("messages", []) or []

    for message in reversed(messages):
        if getattr(message, "type", None) != "ai":
            continue

        usage = getattr(message, "usage_metadata", None)
        if usage:
            return dict(usage)

    return {}


# =============================================================================
# Retrieval y contexto compacto
# =============================================================================


PRIORITY_SOURCE_RULES: tuple[
    tuple[tuple[str, ...], tuple[str, ...]], ...
] = (
    (
        (
            "39.2",
            "abono",
            "sin limite",
            "excluidas del limite",
            "sin cuota",
            "no tengo cuota",
            "cobrar la deduccion",
            "recuperar la deduccion",
            "monetizar",
            "devolucion de la deduccion",
        ),
        (
            "aeat_idi_2025_art_39_2_lis",
            "boe_lis_ley_27_2014",
        ),
    ),
    (
        (
            "35.2",
            "innovacion tecnologica",
            "mejora tecnologica",
            "mejorar un producto",
            "producto nuevo",
            "proceso nuevo",
            "mejora sustancial",
        ),
        (
            "aeat_it_2025_art_35_2_lis",
            "boe_lis_ley_27_2014",
        ),
    ),
    (
        (
            "35.1",
            "investigacion y desarrollo",
            "i+d",
            "desarrollo de software",
            "software propio",
            "proyecto tecnologico",
            "personal investigador",
            "gastos de desarrollo",
            "nuevo producto",
        ),
        (
            "aeat_id_2025_art_35_1_lis",
            "boe_lis_ley_27_2014",
        ),
    ),
    (
        (
            "empresa emergente",
            "ley de startups",
            "certificacion",
            "enisa",
            "startup",
            "tipo reducido",
            "15%",
            "15 %",
            "empresa nueva",
            "sociedad nueva",
            "recien creada",
            "empresa joven",
            "negocio innovador",
            "pagar menos impuestos",
            "beneficio fiscal",
            "ventaja fiscal",
        ),
        (
            "boe_startups_ley_28_2022",
            "boe_orden_pcm_825_2023_certificacion_startups",
        ),
    ),
)

def infer_priority_source_ids(query: str) -> list[str]:
    query_norm = normalize_text(query)
    selected: list[str] = []

    for markers, source_ids in PRIORITY_SOURCE_RULES:
        if not any(normalize_text(marker) in query_norm for marker in markers):
            continue

        for source_id in source_ids:
            if source_id not in selected:
                selected.append(source_id)

    return selected


def merge_retrieval_results(
    result_groups: list[list[tuple[Document, float]]],
    limit: int,
) -> list[tuple[Document, float]]:
    merged: list[tuple[Document, float]] = []
    seen_ids: set[str] = set()

    for group in result_groups:
        for document, score in group:
            unique_id = str(
                document.metadata.get("chunk_id")
                or document.metadata.get("record_id")
                or hashlib.sha1(
                    document.page_content.encode("utf-8")
                ).hexdigest()
            )

            if unique_id in seen_ids:
                continue

            seen_ids.add(unique_id)
            merged.append((document, score))

            if len(merged) >= limit:
                return merged

    return merged


def retrieve_tax_results(
    resources: RetrievalResources,
    query: str,
    k: int | None = None,
) -> list[tuple[Document, float]]:
    clean_query = str(query).strip()

    if not clean_query:
        return []

    k = k or resources.settings.retrieval_k

    global_results = resources.vectorstore.similarity_search_with_score(
        clean_query,
        k=k,
    )

    priority_groups: list[list[tuple[Document, float]]] = []

    for source_id in infer_priority_source_ids(clean_query):
        filtered_results = resources.vectorstore.similarity_search_with_score(
            clean_query,
            k=resources.settings.priority_k_per_source,
            filter={"source_id": source_id},
        )
        priority_groups.append(filtered_results)

    return merge_retrieval_results(
        [*priority_groups, global_results],
        limit=k,
    )


def truncate_document_text(text: str, max_chars: int) -> str:
    compact = re.sub(r"\s+", " ", str(text)).strip()

    if len(compact) <= max_chars:
        return compact

    truncated = compact[:max_chars]
    last_period = truncated.rfind(". ")

    if last_period >= int(max_chars * 0.60):
        truncated = truncated[: last_period + 1]

    return truncated.rstrip() + " […]"


def format_doc_for_context(
    document: Document,
    index: int,
    settings: FiscalSettings,
) -> str:
    metadata = document.metadata or {}
    source = metadata.get("source", "Fuente no indicada")
    title = metadata.get("title", "Título no indicado")
    article = metadata.get("article")
    page = metadata.get("page")
    url = metadata.get("url", "")

    location_parts: list[str] = []

    if article:
        location_parts.append(f"artículo {article}")

    if page:
        location_parts.append(f"página {page}")

    location = (
        ", ".join(location_parts)
        if location_parts
        else "sin localización específica"
    )

    content = truncate_document_text(
        document.page_content,
        settings.max_doc_chars,
    )

    return (
        f"[DOCUMENTO {index}]\n"
        f"Fuente: {source}\n"
        f"Título: {title}\n"
        f"Localización: {location}\n"
        f"URL: {url}\n\n"
        f"Contenido:\n{content}"
    )


def format_context_from_results(
    results: list[tuple[Document, float]],
    settings: FiscalSettings,
) -> str:
    if not results:
        return "No se encontraron documentos relevantes en el corpus."

    blocks: list[str] = []
    current_chars = 0

    for index, (document, _) in enumerate(results, start=1):
        block = format_doc_for_context(document, index, settings)
        remaining = settings.max_context_chars - current_chars

        if remaining <= 300:
            break

        if len(block) > remaining:
            blocks.append(
                block[:remaining].rstrip() + "\n[CONTEXTO TRUNCADO]"
            )
            break

        blocks.append(block)
        current_chars += len(block)

    return "\n\n".join(blocks)


def make_search_tax_corpus_tool(
    resources: RetrievalResources,
) -> BaseTool:
    @tool
    def search_tax_corpus(query: str) -> str:
        """
        Busca información en el corpus fiscal oficial del proyecto.

        Ámbito:
        - Impuesto sobre Sociedades.
        - Empresas emergentes y Ley de Startups.
        - Deducciones por I+D+i.
        - Artículos 35 y 39 LIS.
        - Fuentes BOE y AEAT incorporadas al vectorstore.
        """
        clean_query = str(query).strip()

        if not clean_query:
            return "La consulta de búsqueda está vacía."

        results = retrieve_tax_results(resources, clean_query)

        return format_context_from_results(
            results,
            resources.settings,
        )

    return search_tax_corpus


# =============================================================================
# Guardrails y routing
# =============================================================================


STRONG_IN_SCOPE_KEYWORDS = (
    "impuesto sobre sociedades",
    "impuesto de sociedades",
    "ley 27/2014",
    "lis",
    "empresa emergente",
    "empresas emergentes",
    "ley de startups",
    "ley 28/2022",
    "orden pcm/825/2023",
    "enisa",
    "tipo reducido",
    "15%",
    "15 %",
    "entidad de nueva creacion",
    "i+d",
    "i+d+i",
    "investigacion y desarrollo",
    "innovacion tecnologica",
    "articulo 35",
    "35.1",
    "35.2",
    "articulo 39",
    "39.2",
    "abono de deducciones",
    "deducciones excluidas del limite",
)

BROAD_ENTITY_KEYWORDS = (
    "startup",
    "startups",
    "pyme",
    "pymes",
)

FISCAL_CONTEXT_KEYWORDS = (
    "fiscal",
    "fiscales",
    "tributar",
    "tributacion",
    "impuesto",
    "sociedades",
    "deduccion",
    "deducciones",
    "incentivo",
    "incentivos",
    "tipo impositivo",
    "certificacion",
)

OUT_OF_SCOPE_KEYWORDS = (
    "iva",
    "irpf",
    "modelo 303",
    "modelo 130",
    "modelo 111",
    "retenciones",
    "nomina",
    "contrato laboral",
    "despido",
    "seguridad social",
    "autonomo",
    "cotizacion",
    "consulta vinculante",
    "direccion general de tributos",
    "dgt",
    "herencia",
    "sucesiones",
    "donaciones",
    "ibi",
    "itp",
    "aduanas",
)

FOLLOW_UP_MARKERS = (
    "y ",
    "entonces ",
    "en ese caso",
    "respecto a eso",
    "sobre eso",
    "esa empresa",
    "esa condicion",
    "ese requisito",
    "esa deduccion",
    "lo anterior",
    "durante cuanto",
    "cuanto tiempo",
    "que limite",
    "que porcentaje",
    "que plazo",
)

NO_RAG_EXACT_REQUESTS = {
    "hola",
    "buenos dias",
    "buenas tardes",
    "buenas noches",
    "gracias",
    "muchas gracias",
    "de acuerdo",
    "entendido",
    "adios",
    "hasta luego",
    "que puedes hacer",
    "en que puedes ayudarme",
}

NO_RAG_TRANSFORMATION_MARKERS = (
    "resume la respuesta",
    "resume lo anterior",
    "resumelo",
    "hazlo mas breve",
    "hazla mas breve",
    "explicalo mas sencillo",
    "explicalo de forma sencilla",
    "reformula",
    "ponlo en una tabla",
    "muestralo en una tabla",
    "dame una version mas corta",
)

# =============================================================================
# Ejemplos semánticos para flexibilizar el guardrail
# =============================================================================


IN_SCOPE_SEMANTIC_EXAMPLES = (
    # Impuesto sobre Sociedades
    "Mi empresa acaba de empezar y quiero saber cómo tributa.",
    "¿Puede una sociedad nueva pagar menos impuestos?",
    "¿Qué tipo impositivo se aplica a una empresa recién creada?",
    "¿Qué beneficios fiscales puede tener una pyme?",
    "¿Cómo tributa una empresa joven en el Impuesto sobre Sociedades?",
    "¿Durante cuánto tiempo se puede aplicar un tipo reducido?",

    # Empresas emergentes
    "¿Qué condiciones tiene que cumplir una startup?",
    "¿Cómo sé si mi negocio puede considerarse empresa emergente?",
    "¿Qué ventajas tiene una empresa joven e innovadora?",
    "¿Cómo se obtiene la certificación de startup?",
    "¿Qué ocurre si una empresa deja de cumplir los requisitos?",
    "¿Durante cuánto tiempo se mantiene la condición de startup?",

    # I+D
    "Mi empresa desarrolla software propio, ¿puede aplicar una deducción?",
    "¿Qué gastos de un proyecto tecnológico puedo deducir?",
    "¿El desarrollo de un producto nuevo puede considerarse investigación?",
    "¿Qué incentivo existe para proyectos de investigación?",
    "¿Se puede deducir el salario de los investigadores?",
    "¿Qué requisitos debe cumplir un proyecto de desarrollo de software?",

    # Innovación tecnológica
    "¿Una mejora importante de un producto puede generar una deducción?",
    "¿Qué diferencia hay entre investigar e innovar?",
    "¿El desarrollo tecnológico de un producto puede ser deducible?",
    "¿Qué incentivo fiscal existe para la innovación?",

    # Artículo 39.2
    "¿Puedo cobrar la deducción si no tengo suficiente cuota?",
    "¿Se puede monetizar una deducción por investigación?",
    "¿Cómo puedo recuperar una deducción que no puedo aplicar?",
    "¿Existe alguna devolución para deducciones de I+D?",
    "¿Qué condiciones hay que cumplir para cobrar la deducción?",
)


SEMANTIC_IN_SCOPE_THRESHOLD = 0.34

def _keyword_present(normalized_text: str, keyword: str) -> bool:
    normalized_keyword = normalize_text(keyword)

    if not normalized_keyword:
        return False

    if re.fullmatch(r"[a-z0-9]+", normalized_keyword):
        pattern = (
            rf"(?<![a-z0-9]){re.escape(normalized_keyword)}(?![a-z0-9])"
        )
        return bool(re.search(pattern, normalized_text))

    return normalized_keyword in normalized_text


def contains_any_keyword(text: str, keywords: tuple[str, ...]) -> bool:
    normalized = normalize_text(text)
    return any(_keyword_present(normalized, keyword) for keyword in keywords)

def cosine_similarity(
    left: list[float],
    right: list[float],
) -> float:
    """
    Calcula la similitud coseno entre dos embeddings.
    """
    if not left or not right:
        return 0.0

    dot_product = sum(
        left_value * right_value
        for left_value, right_value
        in zip(left, right)
    )

    left_norm = math.sqrt(
        sum(
            value * value
            for value in left
        )
    )

    right_norm = math.sqrt(
        sum(
            value * value
            for value in right
        )
    )

    if left_norm == 0 or right_norm == 0:
        return 0.0

    return dot_product / (
        left_norm * right_norm
    )


def max_semantic_similarity(
    query_vector: list[float],
    reference_vectors: list[list[float]],
) -> float:
    """
    Devuelve la similitud máxima de una consulta
    respecto a los ejemplos válidos.
    """
    if not reference_vectors:
        return 0.0

    return max(
        cosine_similarity(
            query_vector,
            reference_vector,
        )
        for reference_vector
        in reference_vectors
    )


def looks_like_contextual_follow_up(
    question: str,
) -> bool:
    """
    Detecta preguntas que probablemente dependen
    de un turno anterior.
    """
    question_norm = normalize_text(
        question
    )

    additional_follow_up_markers = (
        "y si",
        "pero entonces",
        "tambien",
        "ademas",
        "en mi caso",
        "para esa empresa",
        "para ese proyecto",
        "cuanto dura",
        "que ocurre despues",
        "que pasa si",
        "se puede aplicar",
        "puedo aplicarlo",
        "puedo cobrarlo",
        "cuales son las condiciones",
    )

    return (
        any(
            marker in question_norm
            for marker in FOLLOW_UP_MARKERS
        )
        or any(
            marker in question_norm
            for marker
            in additional_follow_up_markers
        )
    )


def build_out_of_scope_answer(question: str) -> str:
    return (
        "No puedo responder con base suficiente usando el corpus cargado.\n\n"
        f"Pregunta: {question}\n\n"
        "El corpus está limitado a Impuesto sobre Sociedades, empresas "
        "emergentes, Ley de Startups y deducciones por I+D+i de los "
        "artículos 35 y 39 LIS. Las consultas vinculantes de la DGT "
        "todavía no están integradas."
    )


def is_question_in_scope(
    question: str,
    conversation_context: str = "",
) -> bool:
    question_norm = normalize_text(question)
    context_norm = normalize_text(conversation_context)

    if not question_norm:
        return False

    if contains_any_keyword(question_norm, OUT_OF_SCOPE_KEYWORDS):
        return False

    if contains_any_keyword(question_norm, STRONG_IN_SCOPE_KEYWORDS):
        return True

    broad_entity = contains_any_keyword(
        question_norm,
        BROAD_ENTITY_KEYWORDS,
    )
    fiscal_context = contains_any_keyword(
        question_norm,
        FISCAL_CONTEXT_KEYWORDS,
    )

    if broad_entity and fiscal_context:
        return True

    prior_context_is_fiscal = contains_any_keyword(
        context_norm,
        STRONG_IN_SCOPE_KEYWORDS,
    )
    looks_like_follow_up = any(
        marker in question_norm for marker in FOLLOW_UP_MARKERS
    )

    return prior_context_is_fiscal and looks_like_follow_up


def is_no_rag_request(
    question: str,
    conversation_context: str = "",
) -> bool:
    question_norm = normalize_text(question)
    context_norm = normalize_text(conversation_context)

    if question_norm in NO_RAG_EXACT_REQUESTS:
        return True

    if question_norm.startswith("gracias") and len(question_norm.split()) <= 8:
        return True

    transformation = any(
        marker in question_norm
        for marker in NO_RAG_TRANSFORMATION_MARKERS
    )

    return transformation and bool(context_norm)


# =============================================================================
# Prompts
# =============================================================================


RAG_SYSTEM_PROMPT = """
Eres un asistente fiscal práctico especializado en Impuesto sobre
Sociedades español, empresas emergentes y deducciones por I+D+i.

Los documentos recuperados son la base y la evidencia de tu respuesta,
pero no son el formato de la respuesta.

OBJETIVO PRINCIPAL:

Transforma el contenido legal y administrativo en una explicación útil,
clara y aplicable para el usuario. No te limites a copiar, parafrasear
artículos o hacer un resumen literal de la ley.

REGLAS DE RESPUESTA:

- Empieza contestando directamente a la pregunta.
- Utiliza lenguaje claro para una startup o pyme.
- Explica qué significa la norma en la práctica.
- Cuando el usuario pregunte "¿puedo...?", responde:
    "sí", "no" o "depende", y explica las condiciones.
- Señala qué datos concretos tendría que comprobar la empresa.
- Puedes aplicar de forma prudente la norma a los hechos que aporte
    el usuario, pero presenta la conclusión como orientación condicionada.
- Si faltan datos del caso, enumera cuáles son.
- Puedes utilizar un ejemplo hipotético breve cuando ayude a entenderlo.
- Distingue siempre el ejemplo de la norma.
- No inventes artículos, porcentajes, plazos, requisitos ni límites.
- No reproduzcas párrafos completos de los documentos.
- No empieces la respuesta recitando literalmente un artículo.
- Si el contexto no permite resolver la cuestión, explica exactamente
    qué información falta o qué fuente adicional sería necesaria.
- Cita al final las fuentes utilizadas: BOE o AEAT, título del documento
    y artículo o página disponible.
- Las consultas vinculantes de la DGT no están integradas.
- No sustituyes el asesoramiento fiscal profesional.

ESTRUCTURA RECOMENDADA:

**Respuesta directa**

Una contestación clara de entre una y tres frases.

**Qué significa en la práctica**

Explicación aplicada a una startup o pyme.

**Qué debes comprobar**

Condiciones, requisitos o datos del caso que podrían cambiar la respuesta.

**Fuentes**

Norma o documento oficial utilizado.

**Cautela**

Limitaciones de la respuesta cuando sean relevantes.
""".strip()


DIRECT_RESPONSE_SYSTEM_PROMPT = """
Eres un asistente fiscal conversacional.
No añadas información fiscal nueva. Limítate a resumir, reformular o cambiar
el formato de la respuesta anterior. Conserva su significado, responde en
español y sé conciso.
""".strip()


DOMAIN_QUERY_HINTS: tuple[
    tuple[tuple[str, ...], str],
    ...
] = (
    (
        (
            "39.2",
            "abono",
            "sin limite",
            "sin cuota",
            "no tengo cuota",
            "cobrar la deduccion",
            "recuperar la deduccion",
            "monetizar",
            "devolucion de la deduccion",
        ),
        (
            "Deducciones I+D+i excluidas del límite, "
            "artículo 39.2 LIS, descuento del 20 %, "
            "abono, monetización e insuficiencia de cuota."
        ),
    ),
    (
        (
            "35.2",
            "innovacion tecnologica",
            "mejora tecnologica",
            "mejorar un producto",
            "producto nuevo",
            "proceso nuevo",
            "mejora sustancial",
        ),
        (
            "Concepto, base, gastos y porcentaje de "
            "innovación tecnológica del artículo 35.2 LIS."
        ),
    ),
    (
        (
            "35.1",
            "investigacion y desarrollo",
            "i+d",
            "desarrollo de software",
            "software propio",
            "proyecto tecnologico",
            "personal investigador",
            "gastos de desarrollo",
            "nuevo producto",
        ),
        (
            "Concepto, base, gastos y porcentajes de "
            "investigación y desarrollo del artículo 35.1 LIS."
        ),
    ),
    (
        (
            "empresa emergente",
            "ley de startups",
            "enisa",
            "startup",
            "empresa nueva",
            "sociedad nueva",
            "recien creada",
            "empresa joven",
            "negocio innovador",
            "pagar menos impuestos",
            "ventaja fiscal",
            "ventajas fiscales",
            "beneficio fiscal",
            "beneficios fiscales",
            "tipo reducido",
        ),
        (
            "Requisitos, certificación, pérdida de la condición "
            "y tipo reducido de las empresas emergentes conforme "
            "a la Ley 28/2022 y la Orden PCM/825/2023."
        ),
    ),
)

def build_retrieval_query(
    user_query: str,
    previous_human_query: str = "",
) -> str:
    query_norm = normalize_text(user_query)
    hints = [
        hint
        for markers, hint in DOMAIN_QUERY_HINTS
        if any(normalize_text(marker) in query_norm for marker in markers)
    ]

    parts = [f"Pregunta actual: {user_query}"]

    if previous_human_query:
        parts.append(
            f"Pregunta fiscal anterior del hilo: {previous_human_query}"
        )

    if hints:
        parts.append("Términos de recuperación: " + " ".join(hints))

    return "\n".join(parts)


# =============================================================================
# Estado y utilidades de conversación
# =============================================================================


class FiscalGraphState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

    user_query: NotRequired[str]
    conversation_context: NotRequired[str]
    previous_human_query: NotRequired[str]

    in_scope: NotRequired[bool]
    scope_reason: NotRequired[str]
    scope_score: NotRequired[float]
    
    needs_rag: NotRequired[bool]

    retrieval_query: NotRequired[str]
    retrieved_context: NotRequired[str]
    retrieval_status: NotRequired[str]
    tool_used: NotRequired[bool]
    graph_path: NotRequired[str]



def get_last_human_message_text(state: dict[str, Any]) -> str:
    messages = state.get("messages", []) or []

    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") in {
            "user",
            "human",
        }:
            return message_to_text(message)

        if getattr(message, "type", None) == "human":
            return message_to_text(message)

    return ""


def messages_before_current_human(state: dict[str, Any]) -> list[Any]:
    messages = list(state.get("messages", []) or [])

    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        is_human = (
            getattr(message, "type", None) == "human"
            or (
                isinstance(message, dict)
                and message.get("role") in {"user", "human"}
            )
        )

        if is_human:
            return messages[:index]

    return messages


def get_recent_conversation_text(
    state: dict[str, Any],
    n_messages: int = 6,
    max_chars: int = 3_000,
) -> str:
    prior_messages = messages_before_current_human(state)[-n_messages:]
    text = "\n".join(
        message_to_text(message)
        for message in prior_messages
        if message_to_text(message)
    )

    return text[-max_chars:]


def get_previous_human_query(state: dict[str, Any]) -> str:
    prior_messages = messages_before_current_human(state)

    for message in reversed(prior_messages):
        if getattr(message, "type", None) == "human":
            return message_to_text(message)

        if isinstance(message, dict) and message.get("role") in {
            "user",
            "human",
        }:
            return message_to_text(message)

    return ""


def get_last_ai_message_text(state: dict[str, Any]) -> str:
    prior_messages = messages_before_current_human(state)

    for message in reversed(prior_messages):
        if getattr(message, "type", None) == "ai":
            return message_to_text(message)

        if isinstance(message, dict) and message.get("role") in {
            "assistant",
            "ai",
        }:
            return message_to_text(message)

    return ""


def build_fixed_direct_answer(question: str) -> str | None:
    question_norm = normalize_text(question)

    if question_norm in {
        "hola",
        "buenos dias",
        "buenas tardes",
        "buenas noches",
    }:
        return (
            "Hola. Puedo ayudarte con Impuesto sobre Sociedades, empresas "
            "emergentes y deducciones por I+D+i."
        )

    if question_norm.startswith("gracias"):
        return "De nada."

    if question_norm in {"adios", "hasta luego"}:
        return "Hasta luego."

    if question_norm in {"que puedes hacer", "en que puedes ayudarme"}:
        return (
            "Puedo consultar fuentes BOE y AEAT sobre Impuesto sobre "
            "Sociedades, empresas emergentes y deducciones por I+D+i."
        )

    return None


# =============================================================================
# LLM, nodos y compilación del grafo
# =============================================================================


def create_llm(
    settings: FiscalSettings,
    api_key: str | None = None,
) -> ChatGoogleGenerativeAI:
    resolved_api_key = (
        api_key
        or os.getenv("GOOGLE_API_KEY")
        or os.getenv("GEMINI_API_KEY")
    )

    if not resolved_api_key:
        raise RuntimeError(
            "No se encontró GOOGLE_API_KEY ni GEMINI_API_KEY."
        )

    return ChatGoogleGenerativeAI(
        model=settings.chat_model,
        api_key=resolved_api_key,
        vertexai=False,
        temperature=settings.temperature,
        max_tokens=settings.max_output_tokens,
        thinking_budget=settings.thinking_budget,
        timeout=settings.timeout_seconds,
        max_retries=1,
    )


def build_fiscal_runtime(
    resources: RetrievalResources | None = None,
    *,
    settings: FiscalSettings | None = None,
    api_key: str | None = None,
) -> FiscalRuntime:
    """
    Construye el runtime completo del asistente fiscal.

    El guardrail mantiene las exclusiones explícitas y añade una capa
    semántica local para aceptar preguntas formuladas de manera natural.
    Los embeddings semánticos se crean aquí, cuando ``resources`` ya existe.
    """
    settings = settings or (
        resources.settings
        if resources is not None
        else FiscalSettings.from_env()
    )
    resources = resources or load_retrieval_resources(settings)
    llm = create_llm(settings, api_key=api_key)
    search_tax_corpus = make_search_tax_corpus_tool(resources)

    # Se vectorizan una sola vez al construir el runtime de la sesión.
    # Esta operación usa embeddings locales y no consume Gemini.
    in_scope_reference_vectors = resources.embeddings.embed_documents(
        list(IN_SCOPE_SEMANTIC_EXAMPLES)
    )

    def calculate_semantic_scope_score(
        question: str,
        previous_human_query: str = "",
    ) -> float:
        """
        Calcula la similitud máxima con ejemplos válidos del dominio fiscal.
        Para preguntas de seguimiento, incorpora el turno humano anterior.
        """
        semantic_query = str(question).strip()

        if (
            previous_human_query
            and looks_like_contextual_follow_up(question)
        ):
            semantic_query = (
                "Consulta fiscal anterior:\n"
                f"{previous_human_query}\n\n"
                "Pregunta de seguimiento:\n"
                f"{question}"
            )

        query_vector = resources.embeddings.embed_query(semantic_query)

        return max_semantic_similarity(
            query_vector=query_vector,
            reference_vectors=in_scope_reference_vectors,
        )

    def prepare_query_node(
        state: FiscalGraphState,
    ) -> dict[str, Any]:
        user_query = get_last_human_message_text(state)

        if not user_query:
            raise ValueError(
                "No se encontró una pregunta humana en el estado."
            )

        return {
            "user_query": user_query,
            "conversation_context": get_recent_conversation_text(state),
            "previous_human_query": get_previous_human_query(state),
            "in_scope": False,
            "scope_reason": "not_checked",
            "scope_score": 0.0,
            "needs_rag": False,
            "retrieval_query": "",
            "retrieved_context": "",
            "retrieval_status": "not_run",
            "tool_used": False,
            "graph_path": "prepared",
        }

    def scope_guardrail_node(
        state: FiscalGraphState,
    ) -> dict[str, Any]:
        """
        Guardrail híbrido.

        Orden de decisión:
        1. Bloquea las exclusiones explícitas existentes.
        2. Admite saludos y transformaciones conversacionales.
        3. Admite coincidencias fiscales explícitas.
        4. Admite paráfrasis mediante similitud semántica local.
        """
        question = state.get("user_query", "")
        context = state.get("conversation_context", "")
        previous_human_query = state.get("previous_human_query", "")

        # Las exclusiones duras mantienen prioridad absoluta.
        if contains_any_keyword(question, OUT_OF_SCOPE_KEYWORDS):
            return {
                "in_scope": False,
                "scope_reason": "explicit_out_of_scope_keyword",
                "scope_score": 0.0,
                "graph_path": "scope_checked",
            }

        if is_no_rag_request(question, context):
            return {
                "in_scope": True,
                "scope_reason": "conversational_request",
                "scope_score": 1.0,
                "graph_path": "scope_checked",
            }

        if is_question_in_scope(question, context):
            return {
                "in_scope": True,
                "scope_reason": "keyword_in_scope",
                "scope_score": 1.0,
                "graph_path": "scope_checked",
            }

        semantic_score = calculate_semantic_scope_score(
            question=question,
            previous_human_query=previous_human_query,
        )
        semantic_in_scope = (
            semantic_score >= SEMANTIC_IN_SCOPE_THRESHOLD
        )

        return {
            "in_scope": semantic_in_scope,
            "scope_reason": (
                "semantic_in_scope"
                if semantic_in_scope
                else "semantic_out_of_scope"
            ),
            "scope_score": round(semantic_score, 4),
            "graph_path": "scope_checked",
        }

    def route_after_scope_guardrail(
        state: FiscalGraphState,
    ) -> Literal["rag_router", "out_of_scope_response"]:
        return (
            "rag_router"
            if state.get("in_scope", False)
            else "out_of_scope_response"
        )

    def out_of_scope_response_node(
        state: FiscalGraphState,
    ) -> dict[str, Any]:
        answer = build_out_of_scope_answer(
            state.get("user_query", "")
        )

        return {
            "messages": [AIMessage(content=answer)],
            "needs_rag": False,
            "retrieval_status": "not_run",
            "tool_used": False,
            "graph_path": "out_of_scope",
        }

    def rag_router_node(
        state: FiscalGraphState,
    ) -> dict[str, Any]:
        needs_rag = not is_no_rag_request(
            state.get("user_query", ""),
            state.get("conversation_context", ""),
        )

        return {
            "needs_rag": needs_rag,
            "graph_path": (
                "rag_required"
                if needs_rag
                else "direct_response"
            ),
        }

    def route_after_rag_router(
        state: FiscalGraphState,
    ) -> Literal["retrieve_documents", "direct_response"]:
        return (
            "retrieve_documents"
            if state.get("needs_rag", True)
            else "direct_response"
        )

    def direct_response_node(
        state: FiscalGraphState,
    ) -> dict[str, Any]:
        question = state.get("user_query", "")
        fixed_answer = build_fixed_direct_answer(question)

        if fixed_answer is not None:
            return {
                "messages": [AIMessage(content=fixed_answer)],
                "retrieval_status": "not_run",
                "tool_used": False,
                "graph_path": "direct_response",
            }

        previous_answer = get_last_ai_message_text(state)

        if not previous_answer:
            return {
                "messages": [
                    AIMessage(
                        content=(
                            "No existe una respuesta anterior que "
                            "pueda reformular o resumir."
                        )
                    )
                ],
                "retrieval_status": "not_run",
                "tool_used": False,
                "graph_path": "direct_response",
            }

        response = llm.invoke(
            [
                SystemMessage(content=DIRECT_RESPONSE_SYSTEM_PROMPT),
                HumanMessage(
                    content=(
                        "RESPUESTA ANTERIOR:\n"
                        f"{previous_answer[-3_500:]}\n\n"
                        "PETICIÓN ACTUAL:\n"
                        f"{question}"
                    )
                ),
            ]
        )

        return {
            "messages": [response],
            "retrieval_status": "not_run",
            "tool_used": False,
            "graph_path": "direct_response",
        }

    def retrieve_documents_node(
        state: FiscalGraphState,
    ) -> dict[str, Any]:
        retrieval_query = build_retrieval_query(
            user_query=state.get("user_query", ""),
            previous_human_query=state.get(
                "previous_human_query",
                "",
            ),
        )

        retrieved_context = search_tax_corpus.invoke(
            {"query": retrieval_query}
        )

        return {
            "retrieval_query": retrieval_query,
            "retrieved_context": retrieved_context,
            "retrieval_status": "success",
            "tool_used": True,
            "graph_path": "rag",
        }

    def answer_with_context_node(
        state: FiscalGraphState,
    ) -> dict[str, Any]:
        retrieved_context = state.get("retrieved_context", "")

        if not retrieved_context:
            return {
                "messages": [
                    AIMessage(
                        content=(
                            "No se recuperó contexto documental "
                            "suficiente para responder."
                        )
                    )
                ],
                "graph_path": "rag_without_context",
            }

        conversation_context = state.get(
            "conversation_context",
            "",
        )[-1_800:]

        response = llm.invoke(
            [
                SystemMessage(content=RAG_SYSTEM_PROMPT),
                HumanMessage(
                    content=(
                        "CONTEXTO CONVERSACIONAL PREVIO:\n"
                        f"{conversation_context or 'Sin contexto previo.'}"
                        "\n\n"
                        "PREGUNTA ACTUAL DEL USUARIO:\n"
                        f"{state.get('user_query', '')}\n\n"
                        "TAREA:\n"
                        "Responde a la intención práctica de la pregunta. "
                        "No hagas un resumen general de los documentos. "
                        "Utiliza los documentos para fundamentar una "
                        "respuesta clara, útil y condicionada a los hechos "
                        "disponibles.\n\n"
                        "CONTEXTO DOCUMENTAL RECUPERADO:\n"
                        f"{retrieved_context}"
                    )
                ),
            ]
        )

        return {
            "messages": [response],
            "graph_path": "rag",
        }

    graph_builder = StateGraph(FiscalGraphState)

    graph_builder.add_node("prepare_query", prepare_query_node)
    graph_builder.add_node("scope_guardrail", scope_guardrail_node)
    graph_builder.add_node(
        "out_of_scope_response",
        out_of_scope_response_node,
    )
    graph_builder.add_node("rag_router", rag_router_node)
    graph_builder.add_node("direct_response", direct_response_node)
    graph_builder.add_node(
        "retrieve_documents",
        retrieve_documents_node,
    )
    graph_builder.add_node(
        "answer_with_context",
        answer_with_context_node,
    )

    graph_builder.add_edge(START, "prepare_query")
    graph_builder.add_edge("prepare_query", "scope_guardrail")

    graph_builder.add_conditional_edges(
        "scope_guardrail",
        route_after_scope_guardrail,
        {
            "rag_router": "rag_router",
            "out_of_scope_response": "out_of_scope_response",
        },
    )

    graph_builder.add_edge("out_of_scope_response", END)

    graph_builder.add_conditional_edges(
        "rag_router",
        route_after_rag_router,
        {
            "retrieve_documents": "retrieve_documents",
            "direct_response": "direct_response",
        },
    )

    graph_builder.add_edge("direct_response", END)
    graph_builder.add_edge(
        "retrieve_documents",
        "answer_with_context",
    )
    graph_builder.add_edge("answer_with_context", END)

    graph = graph_builder.compile(
        checkpointer=InMemorySaver()
    )

    return FiscalRuntime(
        settings=settings,
        resources=resources,
        llm=llm,
        search_tax_corpus=search_tax_corpus,
        graph=graph,
    )


# =============================================================================
# API pública para la interfaz
# =============================================================================


def make_thread_config(thread_id: str) -> dict[str, Any]:
    clean_thread_id = str(thread_id).strip()

    if not clean_thread_id:
        raise ValueError("thread_id no puede estar vacío.")

    return {"configurable": {"thread_id": clean_thread_id}}


def invoke_fiscal_runtime(
    runtime: FiscalRuntime,
    question: str,
    thread_id: str,
) -> dict[str, Any]:
    clean_question = str(question).strip()

    if not clean_question:
        raise ValueError("La pregunta no puede estar vacía.")

    state = runtime.graph.invoke(
        {"messages": [HumanMessage(content=clean_question)]},
        config=make_thread_config(thread_id),
    )

    return {
        "question": clean_question,
        "answer": extract_final_answer(state),
        "thread_id": thread_id,
        "state": state,
        "in_scope": state.get("in_scope"),
        "scope_reason": state.get("scope_reason"),
        "scope_score": state.get("scope_score"),
        "needs_rag": state.get("needs_rag"),
        "tool_used": state.get("tool_used"),
        "retrieval_status": state.get("retrieval_status"),
        "graph_path": state.get("graph_path"),
        "usage_metadata": extract_usage_metadata(state),
    }


def stream_fiscal_runtime(
    runtime: FiscalRuntime,
    question: str,
    thread_id: str,
):
    clean_question = str(question).strip()

    if not clean_question:
        raise ValueError("La pregunta no puede estar vacía.")

    yield from runtime.graph.stream(
        {"messages": [HumanMessage(content=clean_question)]},
        config=make_thread_config(thread_id),
        stream_mode="updates",
    )


def get_thread_state(
    runtime: FiscalRuntime,
    thread_id: str,
) -> dict[str, Any]:
    snapshot = runtime.graph.get_state(
        make_thread_config(thread_id)
    )
    values = getattr(snapshot, "values", None)

    return dict(values or {})


def unpack_stream_updates(part: Any) -> list[tuple[str, Any]]:
    """Admite tanto el formato de streaming v1 como el unificado v2."""
    if part is None:
        return []

    if isinstance(part, tuple) and len(part) == 2:
        part = part[1]

    if not isinstance(part, dict):
        return []

    if part.get("type") == "updates" and isinstance(part.get("data"), dict):
        return list(part["data"].items())

    return list(part.items())


def classify_runtime_error(exc: Exception) -> str:
    error_text = normalize_text(str(exc))

    api_markers = (
        "429",
        "resource_exhausted",
        "quota",
        "rate limit",
        "too many requests",
    )

    if any(marker in error_text for marker in api_markers):
        return "api_error"

    if isinstance(exc, FileNotFoundError) or "vectorstore" in error_text:
        return "vectorstore_error"

    return "runtime_error"