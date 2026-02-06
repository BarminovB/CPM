from typing import Dict, Any, List

STATUS_OPTIONS = [
    "Not Started",
    "Planned",
    "In Progress",
    "Blocked",
    "Done",
]

RISK_OPTIONS = [
    "Low",
    "Medium",
    "High",
]


THEMES = {
    "Warm Clay": {
        "bg": "#f7f5f2",
        "surface": "#ffffff",
        "surface2": "#fbfaf7",
        "ink": "#1f2328",
        "muted": "#5f6c7b",
        "accent": "#ff6b4a",
        "accent2": "#2c7a7b",
        "border": "#e2ded7",
        "critical": "#ff6b4a",
        "critical_soft": "#ffd2c5",
        "noncritical": "#2c7a7b",
        "node_crit": "#ffc2b3",
        "node_noncrit": "#d7eef0",
        "edge_fs": "#2563eb",
        "edge_ss": "#2c7a7b",
        "edge_ff": "#f59e0b",
        "edge_sf": "#7c3aed",
        "graph_edge": "#c7bfb4",
        "highlight": "#ffe1d6",
        "sidebar_ink": "#f2f2f2",
        "status": {
            "Not Started": "#e9e6e1",
            "Planned": "#e0ecff",
            "In Progress": "#ffe6a7",
            "Blocked": "#ffd6d6",
            "Done": "#d9f5e8",
        },
        "risk": {
            "Low": "#d9f5e8",
            "Medium": "#fff3c4",
            "High": "#ffe0e0",
        },
    },
    "Nordic Blue": {
        "bg": "#f3f6fb",
        "surface": "#ffffff",
        "surface2": "#f2f7ff",
        "ink": "#1c2433",
        "muted": "#5b6b7f",
        "accent": "#3b82f6",
        "accent2": "#0f766e",
        "border": "#dbe3f2",
        "critical": "#f97316",
        "critical_soft": "#ffe2d1",
        "noncritical": "#0f766e",
        "node_crit": "#ffd6c7",
        "node_noncrit": "#d9f0ff",
        "edge_fs": "#3b82f6",
        "edge_ss": "#0f766e",
        "edge_ff": "#f59e0b",
        "edge_sf": "#8b5cf6",
        "graph_edge": "#c7d3e6",
        "highlight": "#ffe7d6",
        "sidebar_ink": "#f2f2f2",
        "status": {
            "Not Started": "#e8eef6",
            "Planned": "#dbeafe",
            "In Progress": "#fef3c7",
            "Blocked": "#fee2e2",
            "Done": "#dcfce7",
        },
        "risk": {
            "Low": "#dcfce7",
            "Medium": "#fef3c7",
            "High": "#fee2e2",
        },
    },
    "Studio Sage": {
        "bg": "#f4f7f3",
        "surface": "#ffffff",
        "surface2": "#f0f4ef",
        "ink": "#1f2a24",
        "muted": "#5f6f66",
        "accent": "#10b981",
        "accent2": "#2563eb",
        "border": "#dfe7de",
        "critical": "#f97316",
        "critical_soft": "#ffe3c2",
        "noncritical": "#10b981",
        "node_crit": "#ffd8b8",
        "node_noncrit": "#dff5ea",
        "edge_fs": "#2563eb",
        "edge_ss": "#10b981",
        "edge_ff": "#f59e0b",
        "edge_sf": "#7c3aed",
        "graph_edge": "#c5d0c6",
        "highlight": "#ffe6d0",
        "sidebar_ink": "#f2f2f2",
        "status": {
            "Not Started": "#e6ebe7",
            "Planned": "#d9f0ff",
            "In Progress": "#fff3c4",
            "Blocked": "#ffdede",
            "Done": "#d9f5e8",
        },
        "risk": {
            "Low": "#d9f5e8",
            "Medium": "#fff3c4",
            "High": "#ffe0e0",
        },
    },
}

def get_active_theme(theme_name: str) -> Dict[str, Any]:
    return THEMES.get(theme_name, THEMES["Warm Clay"])

APP_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap');
:root {
    --cpm-bg: __CPM_BG__;
    --cpm-surface: __CPM_SURFACE__;
    --cpm-ink: __CPM_INK__;
    --cpm-muted: __CPM_MUTED__;
    --cpm-accent: __CPM_ACCENT__;
    --cpm-accent-2: __CPM_ACCENT2__;
    --cpm-border: __CPM_BORDER__;
    --cpm-surface-2: __CPM_SURFACE2__;
    --cpm-sidebar-ink: __CPM_SIDEBAR_INK__;
    --cpm-radius: 16px;
    --cpm-radius-sm: 12px;
    --cpm-shadow: 0 16px 36px rgba(15, 23, 42, 0.10);
}
html, body, [data-testid="stAppViewContainer"], [data-testid="stSidebar"] {
    font-family: "Space Grotesk", sans-serif;
}
.stApp {
    background: radial-gradient(
        circle at 10% 0%,
        color-mix(in srgb, var(--cpm-accent) 12%, #ffffff 88%) 0%,
        var(--cpm-bg) 45%,
        var(--cpm-surface) 100%
    );
}
[data-testid="stAppViewContainer"] {
    color: var(--cpm-ink);
}
[data-testid="stAppViewContainer"] h1,
[data-testid="stAppViewContainer"] h2,
[data-testid="stAppViewContainer"] h3,
[data-testid="stAppViewContainer"] h4,
[data-testid="stAppViewContainer"] h5,
[data-testid="stAppViewContainer"] h6,
[data-testid="stAppViewContainer"] p,
[data-testid="stAppViewContainer"] label {
    color: var(--cpm-ink);
}
[data-testid="stAppViewContainer"] .stCaption,
[data-testid="stAppViewContainer"] small {
    color: var(--cpm-muted);
}
[data-testid="stAppViewContainer"] table,
[data-testid="stAppViewContainer"] thead th,
[data-testid="stAppViewContainer"] tbody td {
    color: var(--cpm-ink);
}
[data-testid="stDataFrame"] *,
[data-testid="stDataEditor"] * {
    color: var(--cpm-ink);
}
[data-testid="stMetricValue"] {
    color: var(--cpm-ink);
}
[data-testid="stMetricLabel"] {
    color: var(--cpm-muted);
}
[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        color-mix(in srgb, var(--cpm-accent) 6%, var(--cpm-surface) 94%),
        var(--cpm-bg)
    );
    border-right: 1px solid var(--cpm-border);
    box-shadow: inset -1px 0 0 color-mix(in srgb, var(--cpm-accent) 10%, transparent);
}
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h4 {
    padding: 6px 10px;
    border-radius: 10px;
    background: color-mix(in srgb, var(--cpm-accent) 6%, var(--cpm-surface) 94%);
    border: 1px solid color-mix(in srgb, var(--cpm-border) 65%, transparent);
}
[data-testid="stSidebar"] hr {
    border: none;
    height: 1px;
    background: color-mix(in srgb, var(--cpm-border) 60%, transparent);
    margin: 16px 0;
}
[data-testid="stSidebar"] * {
    color: var(--cpm-ink);
}
.cpm-hero {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 24px;
    padding: 20px 28px;
    background: var(--cpm-surface);
    border: 1px solid transparent;
    border-radius: var(--cpm-radius);
    box-shadow: var(--cpm-shadow);
    backdrop-filter: blur(12px);
    position: relative;
}
.cpm-hero::before,
.cpm-metric::before {
    content: "";
    position: absolute;
    inset: 0;
    padding: 1px;
    border-radius: inherit;
    background: linear-gradient(135deg, color-mix(in srgb, var(--cpm-accent) 35%, transparent), rgba(255,255,255,0.2));
    -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
    -webkit-mask-composite: xor;
    mask-composite: exclude;
    pointer-events: none;
}
.cpm-hero::after {
    content: "";
    position: absolute;
    inset: 0;
    border-radius: inherit;
    background: linear-gradient(135deg, rgba(255,255,255,0.55), rgba(255,255,255,0.1));
    pointer-events: none;
}
.cpm-hero > * {
    position: relative;
    z-index: 1;
}
.cpm-hero h1 {
    font-size: 28px;
    margin: 0 0 6px 0;
    color: var(--cpm-ink);
}
.cpm-hero p {
    margin: 0;
    color: var(--cpm-muted);
    font-size: 14px;
}
.cpm-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 12px;
    border-radius: 999px;
    border: 1px solid var(--cpm-border);
    background: var(--cpm-surface-2);
    color: var(--cpm-muted);
    font-size: 12px;
    margin-right: 8px;
    transition: 0.2s ease;
}
.cpm-chip:hover {
    color: var(--cpm-ink);
    border-color: color-mix(in srgb, var(--cpm-accent) 50%, var(--cpm-border) 50%);
    background: color-mix(in srgb, var(--cpm-accent) 10%, var(--cpm-surface-2) 90%);
}
.cpm-metric {
    padding: 16px;
    border-radius: var(--cpm-radius);
    border: 1px solid transparent;
    background: var(--cpm-surface);
    backdrop-filter: blur(10px);
    box-shadow: 0 12px 24px rgba(15, 23, 42, 0.08);
    position: relative;
}
.cpm-metric:hover,
.cpm-hero:hover {
    box-shadow: 0 20px 36px rgba(15, 23, 42, 0.16);
    transform: translateY(-1px);
    transition: 0.25s ease;
}
.cpm-status {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 999px;
    border: 1px solid var(--cpm-border);
    font-size: 12px;
    margin-right: 6px;
    color: var(--cpm-ink);
}
.stButton > button {
    border-radius: var(--cpm-radius-sm);
    font-weight: 600;
    padding: 10px 18px;
    border: 1px solid var(--cpm-border);
    box-shadow: 0 10px 20px rgba(15, 23, 42, 0.10);
    transition: 0.2s ease;
    background: var(--cpm-surface-2);
    color: var(--cpm-ink);
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 16px 28px rgba(15, 23, 42, 0.18);
}
.stButton > button[kind="primary"] {
    background: var(--cpm-accent);
    color: #ffffff;
    border: none;
}
.stButton > button:disabled {
    opacity: 0.6;
    color: var(--cpm-muted);
    background: var(--cpm-surface-2);
    border-color: var(--cpm-border);
}
.stButton > button[kind="primary"]:hover {
    background: color-mix(in srgb, var(--cpm-accent) 85%, #000 15%);
    color: #ffffff;
}
.stButton > button:focus-visible {
    outline: none;
    box-shadow: 0 0 0 4px color-mix(in srgb, var(--cpm-accent) 25%, transparent);
}
:focus-visible {
    outline: none;
    box-shadow: 0 0 0 3px color-mix(in srgb, var(--cpm-accent) 35%, transparent);
    border-radius: var(--cpm-radius-sm);
}
[data-testid="stDataFrame"] :focus-visible,
[data-testid="stDataEditor"] :focus-visible {
    box-shadow: 0 0 0 3px color-mix(in srgb, var(--cpm-accent) 30%, transparent);
}
.cpm-focus-trail:focus-within {
    position: relative;
}
.cpm-focus-trail:focus-within::after {
    content: "";
    position: absolute;
    inset: -6px;
    border-radius: calc(var(--cpm-radius) + 6px);
    border: 1px solid color-mix(in srgb, var(--cpm-accent) 35%, transparent);
    box-shadow: 0 0 0 6px color-mix(in srgb, var(--cpm-accent) 12%, transparent);
    animation: focusTrail 0.4s ease;
    pointer-events: none;
}
@keyframes focusTrail {
    from { opacity: 0; transform: scale(0.98); }
    to { opacity: 1; transform: scale(1); }
}
[data-testid="stTabs"] button {
    transition: 0.2s ease;
    border-radius: 12px;
}
[data-testid="stTabs"] button:hover {
    background: color-mix(in srgb, var(--cpm-accent) 8%, transparent);
}
[data-testid="stTabs"] [aria-selected="true"] {
    background: color-mix(in srgb, var(--cpm-accent) 18%, transparent);
    color: var(--cpm-ink);
    box-shadow: inset 0 -2px 0 var(--cpm-accent);
}
.cpm-shimmer {
    position: relative;
    overflow: hidden;
    background: color-mix(in srgb, var(--cpm-accent) 6%, var(--cpm-surface) 94%);
    border-radius: 12px;
    height: 12px;
    border: 1px solid var(--cpm-border);
}
.cpm-shimmer::after {
    content: "";
    position: absolute;
    inset: 0;
    transform: translateX(-100%);
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.6), transparent);
    animation: shimmer 1.6s infinite;
}
@keyframes shimmer {
    100% { transform: translateX(100%); }
}
.cpm-glass {
    background: color-mix(in srgb, var(--cpm-surface) 70%, transparent);
    border-radius: var(--cpm-radius);
    border: 1px solid color-mix(in srgb, var(--cpm-border) 60%, transparent);
    box-shadow: 0 18px 32px rgba(15, 23, 42, 0.10);
    backdrop-filter: blur(12px);
    padding: 12px;
}
.cpm-fade-in {
    animation: fadeInUp 0.4s ease;
}
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(6px); }
    to { opacity: 1; transform: translateY(0); }
}
.stTextInput input,
.stNumberInput input,
.stTextArea textarea,
.stSelectbox div[data-baseweb="select"] > div,
.stMultiSelect div[data-baseweb="select"] > div {
    border-radius: var(--cpm-radius-sm);
    background: var(--cpm-surface);
    border: 1px solid var(--cpm-border);
    box-shadow: 0 8px 18px rgba(15, 23, 42, 0.08);
    transition: 0.2s ease;
}
.stTextInput input:hover,
.stNumberInput input:hover,
.stTextArea textarea:hover,
.stSelectbox div[data-baseweb="select"] > div:hover,
.stMultiSelect div[data-baseweb="select"] > div:hover {
    border-color: color-mix(in srgb, var(--cpm-accent) 25%, var(--cpm-border) 75%);
    box-shadow: 0 12px 24px rgba(15, 23, 42, 0.12);
}
.stTextInput input:focus,
.stNumberInput input:focus,
.stTextArea textarea:focus,
.stSelectbox div[data-baseweb="select"] > div:focus,
.stMultiSelect div[data-baseweb="select"] > div:focus {
    outline: none;
    border-color: color-mix(in srgb, var(--cpm-accent) 55%, var(--cpm-border) 45%);
    box-shadow: 0 0 0 4px color-mix(in srgb, var(--cpm-accent) 20%, transparent);
}
[data-testid="stSidebar"] .stTextInput input,
[data-testid="stSidebar"] .stNumberInput input,
[data-testid="stSidebar"] .stTextArea textarea,
[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div {
    background: var(--cpm-surface);
    border-color: var(--cpm-border);
    color: var(--cpm-ink);
    box-shadow: 0 8px 18px rgba(15, 23, 42, 0.08);
    outline: none;
    appearance: none;
}
[data-testid="stSidebar"] div[data-baseweb="input"] > div,
[data-testid="stSidebar"] div[data-baseweb="input"] input,
[data-testid="stSidebar"] div[data-baseweb="textarea"] > div,
[data-testid="stSidebar"] div[data-baseweb="textarea"] textarea {
    border: 1px solid var(--cpm-border) !important;
    background: var(--cpm-surface) !important;
    box-shadow: 0 8px 18px rgba(15, 23, 42, 0.08) !important;
    outline: none !important;
}
[data-testid="stSidebar"] .stTextInput input:hover,
[data-testid="stSidebar"] .stNumberInput input:hover,
[data-testid="stSidebar"] .stTextArea textarea:hover,
[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div:hover {
    border-color: color-mix(in srgb, var(--cpm-accent) 25%, var(--cpm-border) 75%);
    box-shadow: 0 12px 24px rgba(15, 23, 42, 0.12);
}
[data-testid="stSidebar"] .stTextInput input:focus,
[data-testid="stSidebar"] .stNumberInput input:focus,
[data-testid="stSidebar"] .stTextArea textarea:focus,
[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div:focus {
    border-color: color-mix(in srgb, var(--cpm-accent) 55%, var(--cpm-border) 45%);
    box-shadow: 0 0 0 4px color-mix(in srgb, var(--cpm-accent) 18%, transparent);
}
[data-testid="stSidebar"] div[data-baseweb="input"]:focus-within > div,
[data-testid="stSidebar"] div[data-baseweb="input"]:focus-within input,
[data-testid="stSidebar"] div[data-baseweb="textarea"]:focus-within > div,
[data-testid="stSidebar"] div[data-baseweb="textarea"]:focus-within textarea {
    border-color: color-mix(in srgb, var(--cpm-accent) 55%, var(--cpm-border) 45%) !important;
    box-shadow: 0 0 0 4px color-mix(in srgb, var(--cpm-accent) 18%, transparent) !important;
}
[data-testid="stSidebar"] .stTextInput input,
[data-testid="stSidebar"] .stNumberInput input {
    box-shadow: 0 8px 18px rgba(15, 23, 42, 0.08);
}
[data-testid="stNumberInput"] button {
    background: var(--cpm-surface-2);
    border: 1px solid var(--cpm-border);
    border-radius: 10px;
    box-shadow: 0 6px 14px rgba(15, 23, 42, 0.08);
}
[data-testid="stNumberInput"] div[data-baseweb="button-group"] {
    background: var(--cpm-surface);
    border-radius: 10px;
    padding: 2px;
}
[data-testid="stNumberInput"] button svg {
    color: var(--cpm-accent);
    fill: var(--cpm-accent);
}
[data-testid="stNumberInput"] button:hover {
    background: color-mix(in srgb, var(--cpm-accent) 10%, var(--cpm-surface-2) 90%);
    border-color: color-mix(in srgb, var(--cpm-accent) 35%, var(--cpm-border) 65%);
}
[data-testid="stDataFrame"] th,
[data-testid="stDataFrame"] td,
[data-testid="stDataEditor"] th,
[data-testid="stDataEditor"] td {
    background: var(--cpm-surface);
    border-color: var(--cpm-border);
}
[data-testid="stDataFrame"] thead th,
[data-testid="stDataEditor"] thead th {
    background: var(--cpm-surface-2);
    color: var(--cpm-ink);
}
[data-testid="stDataFrame"] table,
[data-testid="stDataEditor"] table {
    border-radius: var(--cpm-radius-sm);
    overflow: hidden;
}
[data-testid="stDataFrame"],
[data-testid="stDataEditor"] {
    background: var(--cpm-surface);
    border-radius: calc(var(--cpm-radius) + 2px);
    border: 1px solid color-mix(in srgb, var(--cpm-border) 70%, transparent);
    box-shadow: 0 16px 28px rgba(15, 23, 42, 0.10);
    padding: 6px;
}
[data-testid="stDataFrame"] tbody tr:hover td,
[data-testid="stDataEditor"] tbody tr:hover td {
    background: color-mix(in srgb, var(--cpm-accent) 6%, var(--cpm-surface) 94%);
}
[data-testid="stAlert"] {
    background: color-mix(in srgb, var(--cpm-accent) 10%, var(--cpm-surface) 90%);
    border-color: color-mix(in srgb, var(--cpm-accent) 25%, var(--cpm-border) 75%);
}
[data-testid="stAlert"] code {
    background: color-mix(in srgb, var(--cpm-accent) 10%, var(--cpm-surface) 90%);
    color: var(--cpm-ink);
    border: 1px solid color-mix(in srgb, var(--cpm-accent) 25%, var(--cpm-border) 75%);
    border-radius: 8px;
    padding: 2px 6px;
}
[data-testid="stMarkdown"] code {
    background: color-mix(in srgb, var(--cpm-accent) 10%, var(--cpm-surface) 90%);
    color: var(--cpm-ink);
    border: 1px solid color-mix(in srgb, var(--cpm-accent) 25%, var(--cpm-border) 75%);
    border-radius: 8px;
    padding: 2px 6px;
}
[data-testid="stMarkdown"] pre {
    background: color-mix(in srgb, var(--cpm-accent) 6%, var(--cpm-surface) 94%);
    border-radius: 12px;
    border: 1px solid color-mix(in srgb, var(--cpm-border) 65%, transparent);
    padding: 12px;
}
</style>
"""
def get_theme_css(theme: Dict[str, Any]) -> str:
    """
    Returns the CSS for the application with tokens replaced by theme values.
    """
    css = APP_CSS
    replacements = {
        "__CPM_BG__": theme["bg"],
        "__CPM_SURFACE__": theme["surface"],
        "__CPM_INK__": theme["ink"],
        "__CPM_MUTED__": theme["muted"],
        "__CPM_ACCENT__": theme["accent"],
        "__CPM_ACCENT2__": theme["accent2"],
        "__CPM_BORDER__": theme["border"],
        "__CPM_SURFACE2__": theme["surface2"],
        "__CPM_SIDEBAR_INK__": theme["sidebar_ink"],
    }
    for token, value in replacements.items():
        css = css.replace(token, value)
    return css
