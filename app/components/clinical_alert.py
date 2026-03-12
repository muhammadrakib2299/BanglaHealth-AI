"""Clinical alert display components for Streamlit."""

import streamlit as st


def render_alerts(alerts: list[dict]):
    """Display a list of clinical alerts with appropriate severity styling."""
    if not alerts:
        st.info("No clinical alerts generated.")
        return

    for alert in alerts:
        severity = alert.get("severity", "normal")
        feature = alert.get("feature", "Unknown")
        message = alert.get("message", "")
        shap_impact = alert.get("shap_impact", 0)

        impact_text = f"(SHAP impact: {shap_impact:+.4f})" if shap_impact else ""

        if severity == "high":
            st.error(f"**{feature}**: {message} {impact_text}")
        else:
            st.success(f"**{feature}**: {message} {impact_text}")


def render_alert_summary(alerts: list[dict]):
    """Display a compact summary of alerts."""
    high_alerts = [a for a in alerts if a.get("severity") == "high"]
    normal_alerts = [a for a in alerts if a.get("severity") != "high"]

    if high_alerts:
        st.warning(f"**{len(high_alerts)} risk factor(s) detected**")
    else:
        st.success("All indicators within normal range")

    return high_alerts, normal_alerts
