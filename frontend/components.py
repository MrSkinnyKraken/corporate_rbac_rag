"""Reusable Streamlit UI components.

Every visual fragment that appears in more than one tab lives here so
the main app file (:mod:`frontend.streamlit_app`) stays a thin
orchestration layer. The legal AI disclaimer also lives here as
constants so any future copywriting tweak only needs to touch one
file.
"""

from __future__ import annotations

from typing import Any

import streamlit as st


# ─────────────────────────────────────────────────────────────────────────────
# Legal AI disclaimer — required by the project's compliance brief.
#
# Two variants:
#   * FULL  — for the persistent sidebar banner (one-time per session).
#   * SHORT — for inline placement under every AI-generated artefact
#             (LLM proposals, query answers) so it is impossible to read
#             AI output without seeing the disclaimer next to it.
# ─────────────────────────────────────────────────────────────────────────────

LEGAL_DISCLAIMER_FULL: str = (
    "⚠️ **Avís legal sobre l'ús d'intel·ligència artificial**\n\n"
    "Aquesta aplicació utilitza un model de llenguatge generatiu "
    "(`llama3.2`, executat localment via Ollama) per a dues tasques:\n\n"
    "1. **Classificar metadades** durant la ingesta de documents "
    "(home department, allowed departments, clearance level, resum semàntic).\n"
    "2. **Generar respostes** a partir del context recuperat durant "
    "les consultes.\n\n"
    "Les sortides poden contenir **imprecisions, errors o "
    "omissions** i no substitueixen al humà. Verifiqueu sempre "
    "la informació crítica abans de prendre decisions. Com a usuari final "
    "ets responsable de validar el contingut generat per IA."
)

LEGAL_DISCLAIMER_SHORT: str = (
    "⚠️ **Contingut generat per IA** — pot contenir imprecisions. "
    "Verifiqueu la informació crítica abans de prendre decisions."
)


def render_disclaimer_full() -> None:
    """Persistent legal banner — place once in the sidebar."""
    st.warning(LEGAL_DISCLAIMER_FULL)


def render_disclaimer_short() -> None:
    """Inline reminder placed next to every AI artefact."""
    st.caption(LEGAL_DISCLAIMER_SHORT)


# ─────────────────────────────────────────────────────────────────────────────
# System health
# ─────────────────────────────────────────────────────────────────────────────

def render_health_badge(health: dict[str, Any] | None) -> None:
    """Render a sidebar block with per-service status."""
    st.sidebar.markdown("### Estat del sistema")
    if health is None:
        st.sidebar.error("API no disponible")
        return
    chroma_emoji = "🟢" if health.get("chroma") == "up" else "🔴"
    ollama_emoji = "🟢" if health.get("ollama") == "up" else "🔴"
    embed_emoji = "🟢" if health.get("embedder_loaded") else "🔴"
    model_name = (health.get("embedding_model") or "").rsplit("/", 1)[-1]
    st.sidebar.markdown(
        f"- {chroma_emoji} **ChromaDB**  \n"
        f"- {ollama_emoji} **Ollama** (`{health.get('llm_model', '?')}`)  \n"
        f"- {embed_emoji} **Embedder** (`{model_name}`)"
    )
    if not health.get("embedder_loaded"):
        st.sidebar.caption(
            "_L'embedder es carregarà a la primera consulta (~2 s extra)._"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Identity picker
# ─────────────────────────────────────────────────────────────────────────────

_CLEARANCE_LABELS: dict[int, str] = {
    0: "PUBLIC",
    1: "INTERNAL",
    2: "CONFIDENTIAL",
    3: "STRICT",
}


def render_user_picker(users: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Render the demo user-picker dropdown.

    Returns the full user object (id, username, clearance, department,
    email) so the caller can both display the identity and attach
    ``X-User-Id`` to subsequent api calls.
    """
    if not users:
        st.sidebar.error("Cap usuari registrat a la fixture.")
        return None

    options = {
        f"{u['username']} — {u['department']} / "
        f"{_CLEARANCE_LABELS.get(u['clearance_level'], '?')}": u
        for u in users
    }
    selected_label = st.sidebar.selectbox(
        "Identitat (demo)",
        options=list(options.keys()),
        index=0,
        help=(
            "Demo-only: la identitat es resol des d'un fitxer JSON "
            "(no hi ha OAuth). La capçalera `X-User-Id` viatja a cada "
            "petició a l'API."
        ),
    )
    if not selected_label:
        return None
    u = options[selected_label]
    with st.sidebar.expander("Detalls de la identitat activa"):
        st.markdown(
            f"- **ID:** `{u['user_id']}`\n"
            f"- **Departament:** `{u['department']}`\n"
            f"- **Clearance:** `{u['clearance_level']}` "
            f"({_CLEARANCE_LABELS.get(u['clearance_level'], '?')})\n"
            f"- **Email:** `{u.get('email') or '—'}`"
        )
    return u


# ─────────────────────────────────────────────────────────────────────────────
# Query telemetry & citations
# ─────────────────────────────────────────────────────────────────────────────

def render_citations(citations: list[dict[str, Any]]) -> None:
    """Render the source documents that grounded an answer."""
    if not citations:
        st.caption("_Sense fonts adjuntes._")
        return
    st.markdown("**Fonts utilitzades:**")
    for c in citations:
        ratio = f"{c['n_accessible_chunks']}/{c['n_total_chunks']}"
        st.markdown(
            f"- `{c['source_file']}` "
            f"— {ratio} fragments accessibles "
            f"— doc_id `{c['parent_doc_id'][:12]}…`"
        )


def render_query_telemetry(response: dict[str, Any]) -> None:
    """Expandable block with routing + retrieval + LLM telemetry."""
    with st.expander("📊 Telemetria de la consulta (auditoria)"):
        c1, c2, c3 = st.columns(3)
        c1.metric("Intent", response.get("intent", "—"))
        c2.metric("α (vector weight)", f"{response.get('alpha', 0):.2f}")
        c3.metric(
            "Target departments",
            ", ".join(response.get("target_departments") or []) or "—",
        )

        c1, c2, c3 = st.columns(3)
        c1.metric(
            "Candidats pre/post RBAC",
            f"{response.get('candidates_pre_rbac', 0)} / "
            f"{response.get('candidates_post_rbac', 0)}",
        )
        c2.metric("Chunks usats", response.get("n_chunks_used", 0))
        c3.metric("Documents pare", response.get("n_parents_used", 0))

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Routing", f"{response.get('routing_latency_s', 0):.2f}s")
        c2.metric("Retrieval", f"{response.get('retrieval_latency_s', 0):.2f}s")
        c3.metric("LLM", f"{response.get('llm_latency_s', 0):.2f}s")
        c4.metric("Total", f"{response.get('total_latency_s', 0):.2f}s")

        if response.get("llm_model"):
            st.caption(f"Model: `{response['llm_model']}`")


# ─────────────────────────────────────────────────────────────────────────────
# Error rendering
# ─────────────────────────────────────────────────────────────────────────────

def render_api_error(response, default_msg: str = "Error en la crida a l'API.") -> None:
    """Render an error envelope coming from :mod:`api.errors`.

    Tolerant: if the body is not the structured envelope (e.g., FastAPI's
    own 422 from Pydantic), just dumps the raw JSON.
    """
    try:
        body = response.json()
    except Exception:  # noqa: BLE001
        st.error(f"HTTP {response.status_code}: {response.text[:200]}")
        return
    if isinstance(body, dict) and "error" in body and "detail" in body:
        st.error(
            f"**{body['error']}** (HTTP {response.status_code})\n\n"
            f"{body['detail']}"
        )
        if body.get("context"):
            with st.expander("Detalls"):
                st.json(body["context"])
    else:
        st.error(f"HTTP {response.status_code}: {body}")
