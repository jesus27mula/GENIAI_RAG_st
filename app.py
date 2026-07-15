from __future__ import annotations

import os
from typing import Any
from uuid import uuid4

import streamlit as st
from dotenv import load_dotenv

from src.fiscal_runtime import (
    FiscalSettings,
    build_fiscal_runtime,
    classify_runtime_error,
    extract_final_answer,
    extract_usage_metadata,
    get_thread_state,
    load_retrieval_resources,
    stream_fiscal_runtime,
    unpack_stream_updates,
)


# =============================================================================
# Página y credenciales
# =============================================================================


st.set_page_config(
    page_title="Asistente fiscal RAG",
    page_icon="⚖️",
    layout="centered",
    initial_sidebar_state="expanded",
)

load_dotenv(override=False)


def get_google_api_key() -> str | None:
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

    if api_key:
        return api_key

    try:
        return st.secrets.get("GOOGLE_API_KEY") or st.secrets.get(
            "GEMINI_API_KEY"
        )
    except Exception:
        # Streamlit lanza su propia excepción cuando no existe
        # `.streamlit/secrets.toml`.
        return None


GOOGLE_API_KEY = get_google_api_key()

if not GOOGLE_API_KEY:
    st.error(
        "No se ha configurado la clave de Gemini. Añade GOOGLE_API_KEY "
        "en `.streamlit/secrets.toml` o en un archivo `.env`."
    )
    st.code('GOOGLE_API_KEY = "tu_clave"', language="toml")
    st.stop()

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY


# =============================================================================
# Recursos compartidos y runtime aislado por sesión
# =============================================================================


@st.cache_resource(show_spinner=False)
def get_shared_retrieval_resources():
    settings = FiscalSettings.from_env()
    return load_retrieval_resources(settings)


@st.cache_resource(show_spinner=False, scope="session")
def get_session_runtime():
    resources = get_shared_retrieval_resources()
    return build_fiscal_runtime(
        resources=resources,
        api_key=GOOGLE_API_KEY,
    )


try:
    with st.spinner("Cargando embeddings, Chroma y LangGraph…"):
        retrieval_resources = get_shared_retrieval_resources()
        fiscal_runtime = get_session_runtime()
except Exception as exc:
    st.error("No se pudo inicializar el asistente fiscal.")
    st.exception(exc)
    st.stop()


# =============================================================================
# Estado visual de la sesión
# =============================================================================


def new_thread_id() -> str:
    return f"streamlit_{uuid4().hex}"


if "thread_id" not in st.session_state:
    st.session_state.thread_id = new_thread_id()

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

if "last_run_metadata" not in st.session_state:
    st.session_state.last_run_metadata = {}


# =============================================================================
# Utilidades de interfaz
# =============================================================================


NODE_LABELS = {
    "prepare_query": "Preparando la consulta",
    "scope_guardrail": "Comprobando el ámbito del corpus",
    "rag_router": "Decidiendo si es necesario consultar el RAG",
    "retrieve_documents": "Consultando fuentes BOE y AEAT",
    "answer_with_context": "Generando la respuesta documentada",
    "direct_response": "Preparando una respuesta conversacional",
    "out_of_scope_response": "Aplicando el guardrail de ámbito",
}


def safe_update_metadata(update: Any) -> dict[str, Any]:
    if not isinstance(update, dict):
        return {}

    allowed_keys = {
        "in_scope",
        "scope_reason",
        "scope_score",
        "needs_rag",
        "retrieval_status",
        "tool_used",
        "graph_path",
        "retrieval_query",
    }

    return {
        key: value
        for key, value in update.items()
        if key in allowed_keys
    }


def build_run_metadata(state: dict[str, Any]) -> dict[str, Any]:
    retrieved_context = state.get("retrieved_context", "") or ""

    return {
        "graph_path": state.get("graph_path"),
        "in_scope": state.get("in_scope"),
        "needs_rag": state.get("needs_rag"),
        "tool_used": state.get("tool_used"),
        "retrieval_status": state.get("retrieval_status"),
        "retrieved_context_chars": len(retrieved_context),
        "usage_metadata": extract_usage_metadata(state),
    }


def final_status_label(state: dict[str, Any]) -> str:
    if state.get("in_scope") is False:
        return "Consulta fuera del ámbito documental"

    if state.get("tool_used"):
        return "Respuesta generada con el corpus fiscal"

    return "Respuesta preparada"


# =============================================================================
# Cabecera y barra lateral
# =============================================================================


st.title("⚖️ Asistente fiscal")
st.caption(
    "Gemini + embeddings locales + Chroma + tool RAG + LangGraph"
)

st.info(
    "Ámbito: Impuesto sobre Sociedades, empresas emergentes, Ley de "
    "Startups y deducciones por I+D+i. Las respuestas se basan en las "
    "fuentes BOE y AEAT incorporadas al corpus y no sustituyen el "
    "asesoramiento profesional."
)

with st.sidebar:
    st.header("Conversación")

    if st.button(
        "Nueva conversación",
        icon=":material/add_comment:",
        use_container_width=True,
    ):
        st.session_state.thread_id = new_thread_id()
        st.session_state.chat_messages = []
        st.session_state.last_run_metadata = {}
        st.rerun()

    show_steps = st.toggle(
        "Mostrar pasos del grafo",
        value=True,
        help="Muestra los nombres de los nodos, pero nunca el corpus completo.",
    )

    show_technical_details = st.toggle(
        "Mostrar detalles técnicos",
        value=False,
    )

    st.divider()
    st.subheader("Runtime")
    st.write("**Modelo:**", fiscal_runtime.settings.chat_model)
    st.write("**Vectorstore:** Chroma")
    st.write(
        "**Colección:**",
        fiscal_runtime.settings.collection_name,
    )
    st.write(
        "**Vectores:**",
        f"{retrieval_resources.collection_count:,}".replace(",", "."),
    )
    st.write("**Top-k:**", fiscal_runtime.settings.retrieval_k)

    if show_technical_details:
        st.code(st.session_state.thread_id, language=None)
        st.write("**Ruta Chroma:**")
        st.code(str(fiscal_runtime.settings.vectorstore_dir), language=None)

        if st.session_state.last_run_metadata:
            st.write("**Última ejecución:**")
            st.json(st.session_state.last_run_metadata)


with st.expander("Ejemplos de preguntas"):
    st.markdown(
        """
- ¿Puede una empresa emergente aplicar el tipo reducido del 15 %?
- ¿Qué requisitos debe cumplir una empresa para ser considerada emergente?
- ¿Qué se considera I+D a efectos del artículo 35.1 LIS?
- ¿Qué regula el artículo 35.2 sobre innovación tecnológica?
- ¿Cómo funciona el artículo 39.2 LIS para las deducciones por I+D+i?
        """
    )


# =============================================================================
# Historial visual
# =============================================================================


for chat_message in st.session_state.chat_messages:
    with st.chat_message(chat_message["role"]):
        st.markdown(chat_message["content"])

        metadata = chat_message.get("metadata")
        if show_technical_details and metadata:
            with st.expander("Detalles de esta respuesta"):
                st.json(metadata)


# =============================================================================
# Turno interactivo
# =============================================================================


prompt = st.chat_input(
    "Escribe tu consulta fiscal…",
    max_chars=2_000,
    submit_mode="disable",
)

if prompt:
    st.session_state.chat_messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with st.status(
                "Analizando la consulta…",
                expanded=show_steps,
            ) as status:
                trace: dict[str, Any] = {}

                for part in stream_fiscal_runtime(
                    fiscal_runtime,
                    question=prompt,
                    thread_id=st.session_state.thread_id,
                ):
                    for node_name, update in unpack_stream_updates(part):
                        label = NODE_LABELS.get(
                            node_name,
                            f"Ejecutando nodo: {node_name}",
                        )

                        status.update(label=label)

                        if show_steps:
                            status.write(f"✓ `{node_name}`")

                            if node_name == "retrieve_documents":
                                context_size = len(
                                    (update or {}).get(
                                        "retrieved_context",
                                        "",
                                    )
                                )
                                status.caption(
                                    "Tool ejecutada: `search_tax_corpus` · "
                                    f"{context_size} caracteres recuperados · "
                                    "contenido oculto"
                                )

                        trace.update(safe_update_metadata(update))

                final_state = get_thread_state(
                    fiscal_runtime,
                    st.session_state.thread_id,
                )
                answer = extract_final_answer(final_state)

                if not answer:
                    answer = "No se ha podido generar una respuesta."

                status.update(
                    label=final_status_label(final_state),
                    state="complete",
                    expanded=False,
                )

            st.markdown(answer)

            metadata = build_run_metadata(final_state)
            metadata.update(trace)
            st.session_state.last_run_metadata = metadata

            if show_technical_details:
                with st.expander("Detalles de ejecución"):
                    st.json(metadata)

            st.session_state.chat_messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "metadata": metadata,
                }
            )

        except Exception as exc:
            error_type = classify_runtime_error(exc)

            if error_type == "api_error":
                visible_error = (
                    "La cuota de Gemini está agotada o la API ha limitado "
                    "temporalmente las solicitudes. Inténtalo de nuevo más tarde."
                )
            elif error_type == "vectorstore_error":
                visible_error = (
                    "No se ha podido consultar el vectorstore local. "
                    "Comprueba la carpeta `data/vectorstore/chroma`."
                )
            else:
                visible_error = (
                    "Se ha producido un error al procesar la consulta."
                )

            st.error(visible_error)

            if show_technical_details:
                st.exception(exc)

            st.session_state.chat_messages.append(
                {
                    "role": "assistant",
                    "content": visible_error,
                    "metadata": {"error_type": error_type},
                }
            )
