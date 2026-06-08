"""
app.py
------
AgroInfoShield — Streamlit entry point.
"""

import streamlit as st

st.set_page_config(
    page_title="AgroInfoShield",
    page_icon="🌱",
    layout="centered",
    initial_sidebar_state="collapsed",
)

from dashboard.chat import render_chat

render_chat()