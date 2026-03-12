"""Shared ERP/Desktop CSS for all dashboard pages."""

ERP_CSS = """
<style>
    #MainMenu, footer {visibility: hidden;}
    .block-container { padding-top: 1rem !important; padding-bottom: 0 !important; }

    /* Sidebar */
    [data-testid="stSidebar"] { background: #1E293B; border-right: 1px solid #334155; }
    [data-testid="stSidebar"] [data-testid="stSidebarNav"] a { border-radius: 4px; margin: 1px 8px; padding: 6px 12px; }
    [data-testid="stSidebar"] [data-testid="stSidebarNav"] a span { color: #94A3B8 !important; font-size: 13px !important; }
    [data-testid="stSidebar"] [data-testid="stSidebarNav"] a[aria-selected="true"] { background: #2563EB !important; }
    [data-testid="stSidebar"] [data-testid="stSidebarNav"] a[aria-selected="true"] span { color: #FFF !important; font-weight: 600; }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #CBD5E1 !important; }
    [data-testid="stSidebar"] hr { border-color: #334155; }

    /* Toolbar */
    .toolbar {
        background: #1E293B; color: #F8FAFC; padding: 8px 16px; border-radius: 4px;
        display: flex; align-items: center; justify-content: space-between;
        margin-bottom: 10px; font-size: 13px; border: 1px solid #334155;
    }
    .toolbar .tb-title { font-weight: 700; font-size: 14px; }
    .toolbar .tb-right { font-size: 12px; color: #94A3B8; }

    /* Panel */
    .panel { background: #FFF; border: 1px solid #E2E8F0; border-radius: 4px; margin-bottom: 8px; }
    .panel-header {
        background: #F8FAFC; border-bottom: 1px solid #E2E8F0; padding: 7px 14px;
        font-size: 11px; font-weight: 600; color: #475569; text-transform: uppercase; letter-spacing: 0.5px;
    }
    .panel-body { padding: 10px 14px; }

    /* KPI */
    .kpi { text-align: center; padding: 8px 6px; }
    .kpi .kpi-value { font-size: 24px; font-weight: 700; line-height: 1.1; }
    .kpi .kpi-label { font-size: 10px; color: #64748B; text-transform: uppercase; letter-spacing: 0.5px; margin-top: 2px; }

    /* Risk result */
    .risk-result {
        border-radius: 4px; padding: 16px; text-align: center; color: white; margin-bottom: 8px;
        border: 1px solid rgba(0,0,0,0.1);
    }
    .risk-result h2 { margin: 0; font-size: 32px; color: white; font-weight: 800; }
    .risk-result .rl-label { font-size: 11px; text-transform: uppercase; letter-spacing: 1.5px; opacity: 0.85; margin-bottom: 4px; }
    .risk-result .rl-scores { font-size: 12px; opacity: 0.8; margin-top: 6px; }

    /* Data grid */
    .grid-row { display: flex; border-bottom: 1px solid #F1F5F9; padding: 5px 14px; font-size: 12px; }
    .grid-row:hover { background: #F8FAFC; }
    .grid-row .col-label { width: 160px; color: #64748B; font-weight: 500; flex-shrink: 0; }
    .grid-row .col-value { color: #1E293B; font-weight: 600; }

    /* Alert rows */
    .alert-row {
        display: flex; align-items: flex-start; gap: 8px;
        padding: 6px 12px; font-size: 12px; border-bottom: 1px solid #F1F5F9;
    }
    .alert-row:last-child { border-bottom: none; }
    .alert-row .a-icon { flex-shrink: 0; margin-top: 1px; }
    .alert-row .a-feat { font-weight: 600; color: #1E293B; }
    .alert-row .a-msg { color: #64748B; }
    .alert-high { background: #FEF2F2; }
    .alert-ok { background: #F0FDF4; }
</style>
"""
