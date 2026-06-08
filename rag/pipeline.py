"""
rag/pipeline.py
---------------
Responsibility: Orchestrate the full RAG pipeline — retrieve relevant
documents, format context, call Groq LLM, return structured verdict.
"""

import os
from dataclasses import dataclass
from typing import List, Optional

from dotenv import load_dotenv
from groq import Groq

from rag.retriever import KBRetriever, RetrievalResult

load_dotenv()


@dataclass
class VerdictResponse:
    """Structured response from the RAG pipeline."""
    verdict: str           # "Myth", "Fact", or "Partially True"
    explanation: str       # LLM-generated explanation in user's language
    confidence: float      # Top retrieval score (0.0 to 1.0)
    sources: List[str]     # Source citations from retrieved docs
    retrieved_myths: List[str]  # The matching myths found in KB
    language: str          # Detected language of the query


SYSTEM_PROMPT = """You are AgroInfoShield — an AI fact-checker specializing in GMO 
crop claims for African farmers, extension officers, and policymakers.

Your job:
1. Analyze the user's claim or question
2. Use the provided knowledge base evidence to determine if it is:
   - MYTH: The claim is false based on scientific evidence
   - FACT: The claim is true based on scientific evidence  
   - PARTIALLY TRUE: The claim has elements of truth but is incomplete or misleading

Rules:
- Base your verdict ONLY on the provided evidence — do not use outside knowledge
- If evidence is insufficient, say so honestly
- Respond in the SAME language the user wrote in
- For Swahili queries, respond fully in Swahili
- For Sheng queries, respond in Sheng mixed with English as appropriate
- For English queries, respond formally in English
- Always cite your sources
- Be clear, direct, and accessible — your audience includes farmers with no science background
- Keep explanations under 150 words

Response format (always follow this exactly):
VERDICT: [Myth/Fact/Partially True]
EXPLANATION: [Your explanation in the user's language]
SOURCES: [Source 1, Source 2, ...]"""


def _build_context(results: List[RetrievalResult]) -> str:
    """Format retrieved documents as context for the LLM."""
    if not results:
        return "No relevant evidence found in the knowledge base."

    context_parts = []
    for r in results:
        doc = r.document
        context_parts.append(
            f"Evidence {r.rank} (similarity: {r.score:.2f}):\n"
            f"Claim: {doc.myth}\n"
            f"Verdict: {doc.verdict}\n"
            f"Explanation: {doc.explanation}\n"
            f"Source: {doc.source}\n"
        )
    return "\n---\n".join(context_parts)


def _parse_response(response_text: str) -> dict:
    """Parse the LLM response into structured fields."""
    lines = response_text.strip().split("\n")
    verdict = "Unknown"
    explanation = ""
    sources = []

    for line in lines:
        if line.startswith("VERDICT:"):
            verdict = line.replace("VERDICT:", "").strip()
        elif line.startswith("EXPLANATION:"):
            explanation = line.replace("EXPLANATION:", "").strip()
        elif line.startswith("SOURCES:"):
            sources_raw = line.replace("SOURCES:", "").strip()
            sources = [s.strip() for s in sources_raw.split(",")]

    # Fallback — if parsing fails use full response as explanation
    if not explanation:
        explanation = response_text

    return {
        "verdict": verdict,
        "explanation": explanation,
        "sources": sources,
    }


def _detect_language(text: str) -> str:
    """Simple language detection — full detector built in language/detector.py."""
    sheng_markers = ["poa", "sawa", "niaje", "maze", "si", "ama", "buda",
                     "manze", "fiti", "uko", "hii", "hiyo", "kabisa"]
    swahili_markers = ["je", "ni", "ya", "wa", "na", "kwa", "mazao",
                       "salama", "hatari", "chakula", "mbegu"]

    text_lower = text.lower()
    words = text_lower.split()

    sheng_count = sum(1 for w in words if w in sheng_markers)
    swahili_count = sum(1 for w in words if w in swahili_markers)

    if sheng_count >= 1:
        return "sheng"
    elif swahili_count >= 2:
        return "swahili"
    return "english"


class RAGPipeline:
    """
    Full RAG pipeline for GMO myth verification.

    Usage:
        pipeline = RAGPipeline()
        response = pipeline.verify("Are GMO crops safe to eat?")
        print(response.verdict)
        print(response.explanation)
    """

    def __init__(self, top_k: int = 3, threshold: float = 0.3):
        self.top_k = top_k
        self.threshold = threshold
        self.retriever = KBRetriever(top_k=top_k)
        self._client = None

    @property
    def client(self) -> Groq:
        """Lazy-load Groq client on first use."""
        if self._client is None:
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError(
                    "GROQ_API_KEY not found. "
                    "Add it to your .env file."
                )
            self._client = Groq(api_key=api_key)
        return self._client

    def verify(self, query: str) -> VerdictResponse:
        """
        Verify a GMO claim or question.

        Args:
            query: User's claim or question in any language.

        Returns:
            VerdictResponse with verdict, explanation, sources, confidence.
        """
        # Step 1 — Detect language
        language = _detect_language(query)

        # Step 2 — Retrieve relevant documents
        results = self.retriever.search_with_threshold(
            query,
            threshold=self.threshold,
            top_k=self.top_k,
        )

        # Step 3 — Build context from retrieved docs
        context = _build_context(results)

        # Step 4 — Call Groq LLM
        user_message = (
            f"User query: {query}\n\n"
            f"Knowledge base evidence:\n{context}\n\n"
            f"Based ONLY on the evidence above, provide your verdict."
        )

        chat_response = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.1,
            max_tokens=400,
        )

        response_text = chat_response.choices[0].message.content

        # Step 5 — Parse and return structured response
        parsed = _parse_response(response_text)

        return VerdictResponse(
            verdict=parsed["verdict"],
            explanation=parsed["explanation"],
            sources=parsed["sources"] or [r.document.source for r in results],
            confidence=results[0].score if results else 0.0,
            retrieved_myths=[r.document.myth for r in results],
            language=language,
        )