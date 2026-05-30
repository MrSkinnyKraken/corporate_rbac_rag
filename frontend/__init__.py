"""Streamlit frontend for the Zero-Trust RAG demo.

The frontend is a thin layer over :mod:`api` — it never imports from
:mod:`application`, :mod:`domain` or :mod:`infrastructure`. Every state
mutation goes through the FastAPI surface so the demo's security
guarantees (RBAC, HITL gate, identity resolution) cannot be bypassed
by changing the UI alone.
"""
