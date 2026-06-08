"""
dashboard/chat.py
-----------------
Responsibility: Render the main chat interface tab.
"""

import streamlit as st
from rag.pipeline import RAGPipeline


VERDICT_COLORS = {
    "Myth": "#e74c3c",
    "Fact": "#27ae60",
    "Partially True": "#f39c12",
    "Unknown": "#95a5a6",
}

VERDICT_ICONS = {
    "Myth": "❌",
    "Fact": "✅",
    "Partially True": "⚠️",
    "Unknown": "❓",
}


@st.cache_resource
def get_pipeline():
    """Load pipeline once and cache it across reruns."""
    return RAGPipeline()


def render_verdict_card(verdict: str, explanation: str,
                         confidence: float, sources: list,
                         retrieved_myths: list):
    """Render a styled verdict card."""
    color = VERDICT_COLORS.get(verdict, "#95a5a6")
    icon = VERDICT_ICONS.get(verdict, "❓")

    st.markdown(f"""
        <div style="
            border-left: 6px solid {color};
            background-color: #1a1a2e;
            padding: 20px;
            border-radius: 8px;
            margin: 10px 0;
        ">
            <h2 style="color: {color}; margin: 0 0 10px 0;">
                {icon} {verdict}
            </h2>
            <p style="color: #ecf0f1; font-size: 16px; line-height: 1.6;">
                {explanation}
            </p>
        </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Confidence", f"{confidence:.0%}")
    with col2:
        st.metric("Evidence matched", f"{len(retrieved_myths)} entries")

    if sources:
        with st.expander("📚 Sources"):
            for s in sources:
                if s.strip():
                    st.markdown(f"- {s.strip()}")

    if retrieved_myths:
        with st.expander("🔍 Related claims in knowledge base"):
            for myth in retrieved_myths:
                st.markdown(f"- {myth}")


def render_chat():
    """Render the full chat interface."""

    st.markdown("""
        <div style="text-align: center; padding: 20px 0 10px 0;">
            <h1 style="color: #27ae60; font-size: 2.2em;">
                🌱 AgroInfoShield
            </h1>
            <p style="color: #bdc3c7; font-size: 1.1em;">
                AI-powered GMO myth-busting for African farmers and policymakers
            </p>
            <p style="color: #7f8c8d; font-size: 0.9em;">
                Ask in English · Swahili · Sheng
            </p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Example queries
    st.markdown("**Try an example:**")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🇬🇧 Are GMO crops safe?"):
            st.session_state.query_input = "Are GMO crops safe for human consumption?"
    with col2:
        if st.button("🇰🇪 Je, GMO ni salama?"):
            st.session_state.query_input = "Je, mazao ya GMO ni salama kula?"
    with col3:
        if st.button("Sheng: GMO ni poa?"):
            st.session_state.query_input = "Hiyo GMO ni poa ama ni poison?"

    st.markdown("---")

    # Query input
    query = st.text_area(
        "Enter your claim or question:",
        value=st.session_state.get("query_input", ""),
        height=100,
        placeholder=(
            "e.g. GMO crops will make soil infertile\n"
            "e.g. Mazao ya GMO husababisha utasa\n"
            "e.g. Hiyo GMO ni poison kabisa"
        ),
        key="query_input",
    )

    verify_clicked = st.button(
        "🔍 Verify Claim",
        type="primary",
        use_container_width=True,
    )

    if verify_clicked:
        if not query.strip():
            st.warning("Please enter a claim or question to verify.")
            return

        with st.spinner("Checking against knowledge base..."):
            try:
                pipeline = get_pipeline()
                response = pipeline.verify(query)

                st.markdown("### Result")
                render_verdict_card(
                    verdict=response.verdict,
                    explanation=response.explanation,
                    confidence=response.confidence,
                    sources=response.sources,
                    retrieved_myths=response.retrieved_myths,
                )

            except Exception as e:
                st.error(f"Something went wrong: {str(e)}")