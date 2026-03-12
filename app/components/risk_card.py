"""Reusable risk level display component."""

import streamlit as st


RISK_COLORS = {
    "Low": "#28a745",
    "Medium": "#ffc107",
    "High": "#dc3545",
}


def render_risk_card(risk_level: str, confidence: dict):
    """Display a color-coded risk level card with confidence scores."""
    color = RISK_COLORS.get(risk_level, "#6c757d")

    st.markdown(f"""
    <div style="background-color: {color}; padding: 20px; border-radius: 10px;
                text-align: center; color: white; margin: 10px 0;">
        <h1 style="margin: 0; color: white;">{risk_level} Risk</h1>
        <p style="margin: 5px 0; font-size: 18px; color: white;">
            Low: {confidence.get('Low', 0):.1%} |
            Medium: {confidence.get('Medium', 0):.1%} |
            High: {confidence.get('High', 0):.1%}
        </p>
    </div>
    """, unsafe_allow_html=True)


def render_risk_badge(risk_level: str) -> str:
    """Return an HTML badge string for a risk level."""
    color = RISK_COLORS.get(risk_level, "#6c757d")
    return (
        f'<span style="background-color: {color}; color: white; '
        f'padding: 2px 8px; border-radius: 4px; font-weight: bold;">'
        f'{risk_level}</span>'
    )
