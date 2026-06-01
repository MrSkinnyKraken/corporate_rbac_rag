"""Inspecció CLI de les col·leccions ChromaDB i els índexs BM25.

Útil per a:
  * Verificar quins documents s'han ingestat (`--list`).
  * Inspeccionar el contingut concret d'una col·lecció (`--collection NAME`).
  * Provar la qualitat del retrieval lèxic i vectorial (`--search "..."`).
  * Auditar el guard d'embedding model a cada col·lecció.

S'executa amb el mateix codi que el backend (`infrastructure.vector_db`,
`infrastructure.lexical_db`), garantint que el que veus aquí és exactament
el que el `AsymmetricEnsembleRetriever` veu en temps real.

Exemples
--------
  python -m scripts.inspect_stores
      Llista totes les col·leccions Chroma i tots els índexs BM25 amb un
      preview de 3 chunks per cada un.

  python -m scripts.inspect_stores --collection chunks_legal --limit 10
      Mostra els 10 primers chunks de la col·lecció `chunks_legal`.

  python -m scripts.inspect_stores --search "warranty clause"
      Cerca "warranty clause" a totes les col·leccions BM25 + a totes les
      col·leccions Chroma via kNN (vector dens).

  python -m scripts.inspect_stores --no-chroma --search "IBAN"
      Restringeix la cerca als índexs BM25.

  python -m scripts.inspect_stores --no-bm25 --collection ksp_router_index
      Mostra els KSPs de la col·lecció de routing.

Executar des del host (mode hybrid dev) o dins el container:
  docker exec zero_trust_api python -m scripts.inspect_stores [...]
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from core.exceptions import VectorDBConnectionError, LexicalIndexError
from infrastructure.embedder import get_embedder
from infrastructure.lexical_db import BM25Client
from infrastructure.vector_db import ChromaDBClient


_SEP_MAJOR = "=" * 78
_SEP_MINOR = "-" * 78


# ─────────────────────────────────────────────────────────────────────────────
# Helpers d'output
# ─────────────────────────────────────────────────────────────────────────────

def _truncate(s: str, n: int = 120) -> str:
    """Trunca una cadena a `n` caràcters amb el·lipsi."""
    s = (s or "").replace("\n", " ").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def _format_metadata(meta: dict[str, Any] | None) -> str:
    """Format compacte d'una metadata dict per al display."""
    if not meta:
        return "{}"
    interesting = (
        "parent_doc_id", "chunk_index", "source_file", "department",
        "allowed_departments", "clearance_level", "contains_PII",
        "sensitivity_types", "cross_dept_rules_fired",
        "home_department", "embedding_model",
    )
    parts: list[str] = []
    for key in interesting:
        if key in meta:
            val = meta[key]
            if isinstance(val, str) and len(val) > 40:
                val = val[:37] + "…"
            parts.append(f"{key}={val!r}")
    return "{ " + ", ".join(parts) + " }" if parts else json.dumps(meta)[:80]


def _print_header(title: str) -> None:
    print()
    print(_SEP_MAJOR)
    print(f"  {title}")
    print(_SEP_MAJOR)


# ─────────────────────────────────────────────────────────────────────────────
# Inspecció ChromaDB
# ─────────────────────────────────────────────────────────────────────────────

def list_chroma_collections(vdb: ChromaDBClient) -> list[str]:
    """Retorna els noms de totes les col·leccions; missatge clar si falla."""
    try:
        return vdb.list_collections()
    except VectorDBConnectionError as exc:
        print(f"  ❌ No es pot connectar amb ChromaDB: {exc}", file=sys.stderr)
        return []


def inspect_chroma_collection(
    vdb: ChromaDBClient,
    name: str,
    limit: int,
) -> None:
    """Imprimeix metadata + preview dels primers `limit` chunks d'una col·lecció."""
    try:
        coll = vdb._client.get_collection(name=name)  # raw client per veure metadata real del servidor
    except Exception as exc:  # noqa: BLE001
        print(f"  ❌ No es pot obrir la col·lecció {name!r}: {exc}", file=sys.stderr)
        return

    count = vdb.collection_count(coll)
    coll_meta = dict(coll.metadata or {})
    stamp = coll_meta.get("embedding_model", "(sense stamp)")

    print(f"\n[{name}] {count} vectors  ·  embedding_model: {stamp}")
    if count == 0:
        print("  (col·lecció buida)")
        return

    raw = vdb.get_all_documents(coll)
    ids = raw.get("ids") or []
    docs = raw.get("documents") or []
    metas = raw.get("metadatas") or []
    n_show = min(limit, len(ids))
    for i in range(n_show):
        print(_SEP_MINOR)
        print(f"  id        : {ids[i]}")
        print(f"  metadata  : {_format_metadata(metas[i] if i < len(metas) else {})}")
        print(f"  text      : {_truncate(docs[i] if i < len(docs) else '', 180)}")
    if count > n_show:
        print(f"  ... i {count - n_show} chunks més. Pugeu --limit per veure'n més.")


def search_chroma(
    vdb: ChromaDBClient,
    query: str,
    top_k: int,
) -> None:
    """Cerca el query a totes les col·leccions Chroma via kNN (embedder compartit)."""
    embedder = get_embedder()
    query_vec = embedder.embed(query)
    for name in list_chroma_collections(vdb):
        try:
            coll = vdb.get_or_create_collection(name)
            resp = vdb.query_collection(coll, query_embeddings=[query_vec], n_results=top_k)
        except Exception as exc:  # noqa: BLE001
            print(f"\n[{name}]  ❌ query fallida: {exc}")
            continue
        ids = (resp.get("ids") or [[]])[0]
        docs = (resp.get("documents") or [[]])[0]
        dists = (resp.get("distances") or [[]])[0]
        if not ids:
            continue
        print(f"\n[{name}]  query='{_truncate(query, 60)}'  → top-{len(ids)} vectorials")
        for i in range(len(ids)):
            d = dists[i] if i < len(dists) else float("nan")
            print(f"  · dist={d:.4f}  id={ids[i][:40]}  text={_truncate(docs[i], 100)}")


# ─────────────────────────────────────────────────────────────────────────────
# Inspecció BM25
# ─────────────────────────────────────────────────────────────────────────────

def list_bm25_indexes(ldb: BM25Client) -> list[str]:
    return ldb.list_indexes()


def inspect_bm25_index(ldb: BM25Client, name: str, limit: int) -> None:
    """Imprimeix metadata + preview dels primers chunks d'un índex BM25."""
    try:
        ldb.load_index(name)
    except LexicalIndexError as exc:
        print(f"  ❌ No es pot carregar l'índex {name!r}: {exc}", file=sys.stderr)
        return

    idx = ldb._cache[name]
    chunks = idx.chunks
    print(f"\n[{name}]  {len(chunks)} chunks")
    n_show = min(limit, len(chunks))
    for c in chunks[:n_show]:
        print(_SEP_MINOR)
        print(f"  chunk_id  : {c.chunk_id}")
        print(f"  metadata  : {_format_metadata(c.metadata)}")
        print(f"  text      : {_truncate(c.text, 180)}")
    if len(chunks) > n_show:
        print(f"  ... i {len(chunks) - n_show} chunks més. Pugeu --limit per veure'n més.")


def search_bm25(ldb: BM25Client, query: str, top_k: int) -> None:
    """Cerca `query` a tots els índexs BM25 disponibles."""
    for name in list_bm25_indexes(ldb):
        try:
            hits = ldb.search(name, query, top_k=top_k)
        except LexicalIndexError as exc:
            print(f"\n[{name}]  ❌ cerca fallida: {exc}")
            continue
        if not hits:
            continue
        print(f"\n[{name}]  query='{_truncate(query, 60)}'  → {len(hits)} hits lèxics")
        for chunk, score in hits:
            print(f"  · score={score:.4f}  id={chunk.chunk_id[:40]}  text={_truncate(chunk.text, 100)}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspecció CLI dels stores Chroma i BM25 del sistema RAG.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Exemples")[1] if __doc__ else "",
    )
    parser.add_argument(
        "--collection", "-c",
        default=None,
        help="Restringeix a una sola col·lecció / índex pel nom (per defecte: totes).",
    )
    parser.add_argument(
        "--limit", "-n",
        type=int,
        default=3,
        help="Quants chunks mostrar per col·lecció (default: 3).",
    )
    parser.add_argument(
        "--search", "-s",
        default=None,
        help="Si es passa, executa cerques BM25 + vectorial amb aquest text.",
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=5,
        help="Quants resultats retornar per cada cerca (default: 5).",
    )
    parser.add_argument("--no-chroma", action="store_true", help="No inspeccionar ChromaDB.")
    parser.add_argument("--no-bm25",   action="store_true", help="No inspeccionar BM25.")
    args = parser.parse_args()

    if args.no_chroma and args.no_bm25:
        print("Has desactivat les dues capes; res a fer.", file=sys.stderr)
        return 1

    # ── ChromaDB ──────────────────────────────────────────────────────────
    if not args.no_chroma:
        _print_header("ChromaDB")
        vdb = ChromaDBClient()
        all_collections = list_chroma_collections(vdb)
        if not all_collections:
            print("  Cap col·lecció trobada.")
        else:
            targets = [args.collection] if args.collection else all_collections
            for name in targets:
                if name not in all_collections:
                    print(f"  ⚠ col·lecció {name!r} no existeix; col·leccions disponibles: {all_collections}",
                          file=sys.stderr)
                    continue
                inspect_chroma_collection(vdb, name, args.limit)

            if args.search:
                _print_header(f"Cerca vectorial Chroma → '{args.search}'")
                search_chroma(vdb, args.search, args.top_k)

    # ── BM25 ──────────────────────────────────────────────────────────────
    if not args.no_bm25:
        _print_header("BM25 (índexs lèxics)")
        ldb = BM25Client()
        all_indexes = list_bm25_indexes(ldb)
        if not all_indexes:
            print("  Cap índex BM25 trobat a", ldb._dir)
        else:
            targets = [args.collection] if args.collection else all_indexes
            for name in targets:
                if name not in all_indexes:
                    print(f"  ⚠ índex BM25 {name!r} no existeix; disponibles: {all_indexes}",
                          file=sys.stderr)
                    continue
                inspect_bm25_index(ldb, name, args.limit)

            if args.search:
                _print_header(f"Cerca BM25 → '{args.search}'")
                search_bm25(ldb, args.search, args.top_k)

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
