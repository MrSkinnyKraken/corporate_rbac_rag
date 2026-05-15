# AI RAG based on Context Authorization System
Secure, on-premise Enterprise RAG. Features custom routing, RBAC-driven metadata filtering, and an Ensemble Retriever to prevent data leaks and eliminate LLM hallucinations.<br>

**PART 1:**<br>
Ingestion pipeline<br>
↓<br>
custom chunking<br>
↓<br>
metadata enrichment<br>
↓<br>
Router agent (store decision)<br>
↓<br>
Collection of the vector DB<br>

**PART 2:**<br>
User query to LLM <br>
↓<br>
Router agent (search)<br>
↓<br>
Ensemble retriever<br>
↓<br>
LLM<br>



Query + user{cl, dept}
↓
Router B (KSP, 1 sola colección para routing) → top-2 docs
↓
target_collections = { doc1.department, doc2.department }    # 1 o 2 colecciones físicas
↓
Para cada colección en target_collections:
Asymmetric Hybrid Fusion con where:
clearance_level ≤ user.cl
AND allowed_departments contains user.dept