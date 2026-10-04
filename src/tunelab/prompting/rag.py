from typing import List, Tuple

from tunelab.data.schema import DatabaseSchema
from tunelab.prompting.prompts import build_few_shot_prompt
from tunelab.retrieval.base import BaseRetriever, RetrievedExample


def build_rag_prompt(
    question: str,
    schema: DatabaseSchema,
    retriever: BaseRetriever,
    k: int = 3,
) -> Tuple[str, List[RetrievedExample]]:
    """Builds a dynamic RAG prompt by retrieving top-K training demonstrations.
    
    Reuses Phase 3's standardized few-shot prompt structure while populating
    demonstrations dynamically via semantic or lexical retrieval.
    
    Returns:
        (rag_prompt_string, list_of_retrieved_examples)
    """
    retrieved_items = retriever.retrieve(question, k=k)
    spider_demos = [item.to_spider_example(source_file="rag_retrieval") for item in retrieved_items]

    prompt = build_few_shot_prompt(
        question=question,
        schema=schema,
        demonstrations=spider_demos,
        k=k,
    )

    return prompt, retrieved_items
