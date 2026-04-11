# RAG Degree Plan Search

Two tools are available for querying ingested degree plan documents:

**rag_query**: Semantic and graph-based retrieval. Returns raw source chunks and
graph facts directly without LLM synthesis. Use for conceptual questions,
comparisons across years or programmes, prerequisite chain analysis, or anything
requiring reasoning across multiple documents.

**rag_grep**: Literal text search recursively across all markdown files in
/data/raw/ (all nested sub-directories are searched). Use when you need exact
course codes, credit numbers, or verbatim policy text as it appears in the
source document. Always prefer this for source attribution and exact quotes.

When answering a question, use rag_grep first for verifiable facts, then
rag_query for broader context. Both tools cover all degree programmes available
in the knowledge graph; the current corpus is a sample and will grow.
