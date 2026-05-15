"""Domain layer: business logic agnostic of network protocols and DB drivers.

The packages under :mod:`domain` contain the rules of the system —
chunking, RBAC validation, RRF math, intent classification — and never
import from :mod:`infrastructure` directly. This keeps the business
behaviour reproducible without spinning up Ollama, ChromaDB, or any other
external service.
"""
