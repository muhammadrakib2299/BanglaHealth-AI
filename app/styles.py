"""Minimal shared CSS — only what Streamlit can't do natively."""

ERP_CSS = """
<style>
    #MainMenu, footer {visibility: hidden;}
    .block-container { padding-top: 1rem !important; padding-bottom: 0 !important; }
    [data-testid="stSidebar"] { background: #1E293B; }
    [data-testid="stSidebar"] [data-testid="stSidebarNav"] a span { color: #94A3B8 !important; font-size: 13px !important; }
    [data-testid="stSidebar"] [data-testid="stSidebarNav"] a[aria-selected="true"] { background: #2563EB !important; }
    [data-testid="stSidebar"] [data-testid="stSidebarNav"] a[aria-selected="true"] span { color: #FFF !important; font-weight: 600; }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #CBD5E1 !important; }
    [data-testid="stSidebar"] hr { border-color: #334155; }
</style>
"""
