"""Streamlit frontend — entry point.

Two tabs: **Ingesta** and **Consulta**. The sidebar carries the
persistent legal AI disclaimer, the demo user-picker (replaces OAuth
for the TFG demo) and a live health badge.

The frontend never touches the storage layers directly; every action
becomes an HTTP call to the FastAPI backend through
:class:`frontend.api_client.ApiClient`. This keeps the demo's security
guarantees enforceable: even if the user opens the Streamlit code, they
cannot bypass RBAC, the HITL gate, or the identity resolver — they live
behind the api boundary.
"""

from __future__ import annotations

import os
from typing import Any

import streamlit as st

from frontend.api_client import ApiClient
from frontend.components import (
    LEGAL_DISCLAIMER_FULL,
    render_api_error,
    render_citations,
    render_disclaimer_full,
    render_disclaimer_short,
    render_health_badge,
    render_query_telemetry,
    render_user_picker,
)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

_API_URL: str = os.environ.get("API_URL", "http://localhost:8000")

_DEPARTMENTS: list[str] = [
    "hr", "finance", "engineering", "legal", "sales", "operations", "marketing",
]
_ALLOWED_DEPARTMENTS: list[str] = _DEPARTMENTS + ["all"]
_SUPPORTED_EXTENSIONS: list[str] = [
    "txt", "md", "csv", "pdf", "docx", "xlsx", "pptx", "html", "htm",
]


# ─────────────────────────────────────────────────────────────────────────────
# Resource caches (per-session)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def get_api_client() -> ApiClient:
    """One :class:`ApiClient` per Streamlit session (connection pool reuse)."""
    return ApiClient(_API_URL)


def _fetch_users(api: ApiClient) -> list[dict[str, Any]] | None:
    """Pull the user list, surfacing any backend error in the sidebar."""
    try:
        r = api.list_users()
    except Exception as exc:  # noqa: BLE001
        st.sidebar.error(f"API inaccessible a `{api.base_url}`: {exc}")
        return None
    if r.status_code != 200:
        st.sidebar.error(f"/users HTTP {r.status_code}: {r.text[:200]}")
        return None
    return r.json()


def _fetch_health(api: ApiClient) -> dict[str, Any] | None:
    """Health probe — never raises so the sidebar can show 'down' gracefully."""
    try:
        r = api.health()
        if r.status_code == 200:
            return r.json()
    except Exception:  # noqa: BLE001
        pass
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────

def ingestion_tab(api: ApiClient, user: dict[str, Any]) -> None:
    """Two-phase ingestion UI: upload → LLM propose → HITL review → commit."""
    st.header("📥 Ingesta de documents")
    st.markdown(
        f"Carregant com **{user['username']}** "
        f"(dept. `{user['department']}`, clearance `{user['clearance_level']}`)."
    )

    uploaded_file = st.file_uploader(
        "Selecciona un document per ingestar",
        type=_SUPPORTED_EXTENSIONS,
        accept_multiple_files=False,
        help=(
            "Formats suportats: " + ", ".join(_SUPPORTED_EXTENSIONS) + ". "
            "El document es classificarà amb el LLM i hauràs de revisar "
            "la metadata abans de la indexació definitiva."
        ),
    )

    if uploaded_file is None:
        st.info("Puja un fitxer per començar.")
        return

    if st.button("Analitzar amb IA", type="primary", key="propose_btn"):
        with st.spinner(
            "L'LLM està classificant el document (pot trigar uns segons "
            "depenent de la mida del fitxer i del hardware)..."
        ):
            try:
                r = api.propose_ingestion(
                    file_bytes=uploaded_file.getvalue(),
                    filename=uploaded_file.name,
                    user_id=user["user_id"],
                )
            except Exception as exc:  # noqa: BLE001
                st.error(f"Excepció en la crida a `/ingestion/propose`: {exc}")
                return
        if r.status_code != 200:
            render_api_error(r)
            return
        st.session_state["proposal"] = r.json()
        st.success(
            "✅ Proposta generada. **Revisa-la a sota** abans de confirmar."
        )

    proposal = st.session_state.get("proposal")
    if proposal:
        _render_hitl_review(api, user, proposal)


def _render_hitl_review(
    api: ApiClient,
    user: dict[str, Any],
    proposal: dict[str, Any],
) -> None:
    """Render the HITL review form + commit button."""
    st.markdown("---")
    st.subheader("Revisió HITL (Human-In-The-Loop)")
    render_disclaimer_short()

    pm = proposal["proposed_metadata"]
    st.markdown(
        f"**Document:** `{proposal['source_file']}`  \n"
        f"**Latència LLM (classifier):** {proposal['llm_latency_s']:.1f}s  \n"
        f"**Parent doc id:** `{proposal['parent_doc_id'][:12]}…`"
    )

    with st.expander("📄 Preview del text (primers ~2 KB)"):
        st.text(proposal["text_excerpt"])

    if pm.get("reasoning"):
        with st.expander("🤖 Raonament del LLM"):
            st.markdown(pm["reasoning"])

    st.markdown("**Edita la metadata si cal:**")
    col1, col2 = st.columns(2)
    with col1:
        home_idx = (
            _DEPARTMENTS.index(pm["home_department"])
            if pm["home_department"] in _DEPARTMENTS else 0
        )
        home = st.selectbox(
            "Home department (col·lecció física)",
            options=_DEPARTMENTS,
            index=home_idx,
            help="Determina a quina `chunks_<dept>` van els chunks.",
        )
        clearance = st.slider(
            "Clearance level",
            min_value=0,
            max_value=3,
            value=int(pm["clearance_level"]),
            help="0=PUBLIC · 1=INTERNAL · 2=CONFIDENTIAL · 3=STRICT (els chunks "
                 "amb PII s'escalen automàticament).",
        )
    with col2:
        # Sanitise LLM output: drop unknown department names defensively.
        safe_default = [
            d for d in pm["allowed_departments"] if d in _ALLOWED_DEPARTMENTS
        ] or [home]
        allowed = st.multiselect(
            "Allowed departments (RBAC scope baseline)",
            options=_ALLOWED_DEPARTMENTS,
            default=safe_default,
            help=(
                "Cal incloure el home_department. Utilitzeu `all` només per "
                "documents totalment públics; no es pot combinar amb altres."
            ),
        )
    ksp_text = st.text_area(
        "KSP text (1-3 frases descriptives — alimenta el router semàntic)",
        value=pm["ksp_text"],
        height=80,
    )

    # ── Client-side validation (mirrors ProposedMetadata.__post_init__) ────
    errors: list[str] = []
    if "all" in allowed and len(allowed) > 1:
        errors.append("`all` no pot combinar-se amb altres departaments.")
    if "all" not in allowed and home not in allowed:
        errors.append(
            f"El `home_department='{home}'` ha d'aparèixer a "
            f"`allowed_departments` (o substituir-ho per `all`)."
        )
    if not ksp_text.strip():
        errors.append("El `ksp_text` no pot estar buit.")
    for err in errors:
        st.error(err)

    col_commit, col_reset = st.columns([2, 1])
    with col_commit:
        commit_clicked = st.button(
            "Confirmar i indexar",
            type="primary",
            disabled=bool(errors),
            key="commit_btn",
        )
    with col_reset:
        if st.button("✖ Descartar proposta", key="discard_btn"):
            st.session_state.pop("proposal", None)
            st.rerun()

    if commit_clicked and not errors:
        final_metadata = {
            "home_department": home,
            "allowed_departments": allowed,
            "clearance_level": clearance,
            "ksp_text": ksp_text.strip(),
            "reasoning": pm.get("reasoning", ""),
        }
        with st.spinner(
            "Indexant a ChromaDB + BM25 + KSP "
            "(chunking, embedding, refit, persistència)..."
        ):
            try:
                r = api.commit_ingestion(
                    proposal=proposal,
                    final_metadata=final_metadata,
                    user_id=user["user_id"],
                )
            except Exception as exc:  # noqa: BLE001
                st.error(f"Excepció en la crida a `/ingestion/commit`: {exc}")
                return
        if r.status_code != 200:
            render_api_error(r)
            return
        report = r.json()
        st.success(
            f"✅ Indexats **{report['n_chunks_total']}** chunks a "
            f"`{report['home_collection']}` "
            f"(PII: {report['n_chunks_pii']}, "
            f"extended per regles: {report['n_chunks_extended']})"
        )
        with st.expander("📊 Detalls de l'indexat"):
            st.json(report)
        st.session_state.pop("proposal", None)


def query_tab(api: ApiClient, user: dict[str, Any]) -> None:
    """Query UI: question → /query → answer + citations + telemetry."""
    st.header("💬 Consulta")
    st.markdown(
        f"Consultant com **{user['username']}** "
        f"(dept. `{user['department']}`, clearance `{user['clearance_level']}`)."
    )

    query = st.text_area(
        "Pregunta",
        height=80,
        placeholder=(
            "p.ex. 'Quina és la durada de la garantia en el MSA?' / "
            "'What is the warranty period?' / "
            "'¿Cuál es el periodo de garantía?'"
        ),
        help=(
            "L'orquestrador detectarà l'idioma de la consulta i la "
            "respondrà en el mateix (EN/ES/CA)."
        ),
    )

    if not st.button("Preguntar", type="primary", key="query_btn"):
        return
    if not query.strip():
        st.warning("Escriu una pregunta primer.")
        return

    with st.spinner("Routing → retrieval → generació LLM..."):
        try:
            r = api.query(query_text=query, user_id=user["user_id"])
        except Exception as exc:  # noqa: BLE001
            st.error(f"Excepció en la crida a `/query`: {exc}")
            return

    if r.status_code != 200:
        render_api_error(r)
        return

    response = r.json()

    if response.get("refused"):
        st.warning(f"⛔ {response['answer']}")
        st.caption(
            f"*Motiu del rebuig:* `{response.get('refusal_reason', '?')}`"
        )
    else:
        st.markdown("### 🤖 Resposta")
        st.markdown(response["answer"])
        render_disclaimer_short()
        st.markdown("---")
        render_citations(response.get("citations") or [])

    render_query_telemetry(response)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="Zero-Trust RAG demo",
        page_icon="🔒",
        layout="wide",
    )

    api = get_api_client()

    # ─── Sidebar ───────────────────────────────────────────────────────────
    st.sidebar.title("🔒 Zero-Trust RAG")
    st.sidebar.caption(f"Backend: `{api.base_url}`")

    health = _fetch_health(api)
    render_health_badge(health)
    st.sidebar.markdown("---")

    users = _fetch_users(api)
    if users is None:
        st.title("Zero-Trust RAG demo")
        st.error(
            "No es pot connectar amb l'API. Comprova que la stack està "
            "aixecada (`docker compose up -d`)."
        )
        return

    current_user = render_user_picker(users)
    if current_user is None:
        st.title("Zero-Trust RAG demo")
        st.warning("Selecciona un usuari per continuar.")
        return

    st.sidebar.markdown("---")
    render_disclaimer_full()

    # ─── Main panel ────────────────────────────────────────────────────────
    st.title("Zero-Trust RAG demo")
    st.caption(
        "Sistema RAG on-premise amb RBAC 2D (clearance x departments), "
        "ingesta amb HITL gate i routing jeràrquic per KSP."
    )

    tab_ingest, tab_query = st.tabs(["📥 Ingesta", "💬 Consulta"])
    with tab_ingest:
        ingestion_tab(api, current_user)
    with tab_query:
        query_tab(api, current_user)


if __name__ == "__main__":
    main()
