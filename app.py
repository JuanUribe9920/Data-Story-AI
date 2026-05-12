"""
DataStory AI — Tu Excel habla solo
Groq (Llama 3.3-70b) · Agente de análisis + visualización inteligente
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DataStory AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

MODEL = "llama-3.3-70b-versatile"

# ─────────────────────────────────────────────────────────────────────────────
# Design tokens
# ─────────────────────────────────────────────────────────────────────────────
ACCENT  = "#0a84ff"
SUCCESS = "#30d158"
DANGER  = "#ff453a"
WARN    = "#ff9f0a"
PURPLE  = "#bf5af2"
CYAN    = "#64d2ff"
GOLD    = "#ffd60a"

BG      = "#000000"
BG_CARD = "#111111"
BORDER  = "#2c2c2e"
TEXT    = "#f5f5f7"
MUTED   = "#8e8e93"

CHART_COLORS = [ACCENT, SUCCESS, WARN, DANGER, PURPLE, CYAN, GOLD, "#ff6b35"]

PLOTLY_BASE = dict(
    paper_bgcolor=BG_CARD,
    plot_bgcolor=BG_CARD,
    font=dict(color=TEXT, family="Inter, system-ui, sans-serif", size=12),
    xaxis=dict(gridcolor="#222", linecolor=BORDER, zerolinecolor=BORDER,
               tickfont=dict(color=MUTED, size=11)),
    yaxis=dict(gridcolor="#222", linecolor=BORDER, zerolinecolor=BORDER,
               tickfont=dict(color=MUTED, size=11)),
    legend=dict(bgcolor=BG_CARD, bordercolor=BORDER, font=dict(color=TEXT, size=11)),
    colorway=CHART_COLORS,
    margin=dict(l=16, r=16, t=48, b=24),
    title=dict(font=dict(size=13, color=TEXT, family="Inter, sans-serif"), x=0.02, xanchor="left"),
    hoverlabel=dict(bgcolor="#1c1c1e", bordercolor=BORDER, font=dict(color=TEXT, size=12)),
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', system-ui, sans-serif;
    background-color: {BG};
    color: {TEXT};
}}
#MainMenu, footer, header {{ visibility: hidden; }}
.block-container {{ padding: 2rem 3rem 5rem; max-width: 1440px; }}

/* ── Header ── */
.ds-header {{
    text-align: center;
    padding: 3rem 0 2.5rem;
    border-bottom: 1px solid {BORDER};
    margin-bottom: 2.5rem;
}}
.ds-header h1 {{
    font-size: 2.8rem;
    font-weight: 700;
    letter-spacing: -0.04em;
    color: {TEXT};
    margin: 0 0 0.5rem;
}}
.ds-header h1 span {{ color: {ACCENT}; }}
.ds-header p {{ font-size: 0.95rem; color: {MUTED}; margin: 0; }}

/* ── Upload zone ── */
[data-testid="stFileUploader"] {{
    background: {BG_CARD};
    border: 1.5px dashed {BORDER};
    border-radius: 16px;
    padding: 0.5rem 1rem;
    transition: border-color 0.2s;
}}
[data-testid="stFileUploader"]:hover {{ border-color: {ACCENT}; }}
[data-testid="stFileUploaderDropzone"] {{ background: transparent !important; }}

/* ── Welcome tips grid ── */
.tips-title {{
    font-size: 0.82rem;
    font-weight: 600;
    color: {MUTED};
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin: 2rem 0 1rem;
}}
.tips-grid {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.75rem;
    margin-bottom: 2rem;
}}
.tip-card {{
    background: {BG_CARD};
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 1rem 1.1rem;
    display: flex;
    gap: 0.7rem;
    align-items: flex-start;
}}
.tip-card .tip-icon {{
    font-size: 1.1rem;
    flex-shrink: 0;
    margin-top: 1px;
}}
.tip-card .tip-text {{
    font-size: 0.8rem;
    color: {MUTED};
    line-height: 1.45;
}}
.tip-card .tip-text b {{ color: {TEXT}; font-weight: 500; }}

/* ── Big loader ── */
.big-loader {{
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 4rem 2rem;
    gap: 1.5rem;
}}
.big-loader p {{
    font-size: 0.92rem;
    color: {MUTED};
    margin: 0;
    letter-spacing: 0.02em;
}}
.loader-ring {{
    width: 56px;
    height: 56px;
    border: 3px solid {BORDER};
    border-top-color: {ACCENT};
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
}}
@keyframes spin {{ to {{ transform: rotate(360deg); }} }}

/* ── Metrics ── */
.metrics-row {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin: 1.5rem 0;
}}
.metric-card {{
    background: {BG_CARD};
    border: 1px solid {BORDER};
    border-radius: 14px;
    padding: 1.3rem 1.5rem;
    transition: border-color 0.2s;
}}
.metric-card:hover {{ border-color: {ACCENT}44; }}
.metric-card .val {{
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: -0.03em;
    color: {TEXT};
    line-height: 1;
    margin-bottom: 0.4rem;
}}
.metric-card .lbl {{
    font-size: 0.72rem;
    color: {MUTED};
    text-transform: uppercase;
    letter-spacing: 0.07em;
    font-weight: 500;
}}
.metric-card.blue  .val {{ color: {ACCENT};  }}
.metric-card.green .val {{ color: {SUCCESS}; }}
.metric-card.red   .val {{ color: {DANGER};  }}
.metric-card.warn  .val {{ color: {WARN};    }}

/* ── Theme pill ── */
.theme-pill {{
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: {ACCENT}15;
    border: 1px solid {ACCENT}40;
    color: {ACCENT};
    font-size: 0.82rem;
    font-weight: 500;
    padding: 0.35rem 0.9rem;
    border-radius: 100px;
    margin-bottom: 1.5rem;
}}

/* ── Section title ── */
.section-title {{
    font-size: 0.72rem;
    font-weight: 600;
    color: {MUTED};
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin: 2.5rem 0 1rem;
    padding-bottom: 0.6rem;
    border-bottom: 1px solid {BORDER};
}}

/* ── Insight cards ── */
.insights-grid {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.9rem;
    margin-bottom: 1rem;
}}
.insight-card {{
    border-radius: 12px;
    padding: 1.1rem 1.3rem;
    border-left: 3px solid;
}}
.insight-card .ic-badge {{
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 0.45rem;
    opacity: 0.9;
}}
.insight-card .ic-title {{
    font-size: 0.9rem;
    font-weight: 600;
    color: {TEXT};
    margin-bottom: 0.3rem;
    line-height: 1.3;
}}
.insight-card .ic-detail {{
    font-size: 0.82rem;
    color: {MUTED};
    line-height: 1.5;
}}

/* ── Alert cards ── */
.alert-card {{
    display: flex;
    align-items: flex-start;
    gap: 0.75rem;
    border-radius: 10px;
    padding: 0.85rem 1.1rem;
    margin-bottom: 0.6rem;
    font-size: 0.85rem;
    border-left: 3px solid;
}}
.alert-card .al-icon {{ font-size: 1rem; flex-shrink: 0; margin-top: 1px; }}
.alert-card .al-text {{ color: {MUTED}; line-height: 1.4; }}

/* ── Chart insight line ── */
[data-testid="stPlotlyChart"] {{
    background: {BG_CARD};
    border: 1px solid {BORDER};
    border-radius: 14px;
    overflow: hidden;
    padding: 4px;
}}
.chart-insight {{
    font-size: 0.8rem;
    color: {MUTED};
    padding: 0.2rem 0.5rem 0.8rem;
    font-style: italic;
    line-height: 1.4;
}}

/* ── Department detail panel ── */
.dept-panel {{
    background: {BG_CARD};
    border: 1px solid {ACCENT}44;
    border-left: 3px solid {ACCENT};
    border-radius: 12px;
    padding: 1rem 1.3rem;
    margin: 0.5rem 0 1rem;
}}
.dept-panel .dept-title {{
    font-size: 0.85rem;
    font-weight: 600;
    color: {ACCENT};
    margin-bottom: 0.6rem;
}}
.dept-stats {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.6rem;
}}
.dept-stat {{
    background: #0a0a0a;
    border-radius: 8px;
    padding: 0.6rem 0.8rem;
}}
.dept-stat .ds-val {{
    font-size: 1.2rem;
    font-weight: 700;
    color: {TEXT};
    line-height: 1;
}}
.dept-stat .ds-lbl {{
    font-size: 0.68rem;
    color: {MUTED};
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 0.2rem;
}}

/* ── Buttons ── */
.stButton > button {{
    background: {ACCENT} !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    padding: 0.6rem 1.5rem !important;
    transition: opacity 0.2s !important;
}}
.stButton > button:hover {{ opacity: 0.85 !important; }}


/* ── Expander ── */
[data-testid="stExpander"] {{
    background: {BG_CARD} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 14px !important;
    margin-top: 1.5rem;
}}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {{
    border-radius: 12px;
    border: 1px solid {BORDER};
    overflow: hidden;
}}


/* ── st.status override ── */
[data-testid="stStatusWidget"] {{
    background: {BG_CARD} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 10px !important;
    margin: 0.5rem 0 !important;
}}

/* ── Footer ── */
.ds-footer {{
    text-align: center;
    padding: 2.5rem 0 1rem;
    margin-top: 4rem;
    border-top: 1px solid {BORDER};
}}
.ds-footer span {{
    font-size: 0.72rem;
    color: #3a3a3c;
    letter-spacing: 0.04em;
}}
.ds-footer span b {{ color: {MUTED}; font-weight: 500; }}
</style>
"""

# ─────────────────────────────────────────────────────────────────────────────
# Welcome tips
# ─────────────────────────────────────────────────────────────────────────────
TIPS = [
    ("📋", "<b>Encabezados en fila 1</b>", "La primera fila debe tener los nombres de columna, sin celdas vacías."),
    ("🚫", "<b>Sin celdas combinadas</b>", "Evita merge cells — dificultan el análisis automático."),
    ("🖼️", "<b>Sin imágenes ni gráficos</b>", "Solo datos en texto o números. Nada embebido en las celdas."),
    ("📑", "<b>Una sola pestaña</b>", "Coloca todos tus datos en una única hoja del libro Excel."),
    ("🔁", "<b>Sin duplicados</b>", "Elimina filas repetidas para no distorsionar el análisis."),
    ("⬜", "<b>Sin filas/columnas vacías</b>", "No dejes filas o columnas completamente en blanco entre datos."),
    ("🔢", "<b>Tipos consistentes</b>", "No mezcles números y texto en la misma columna."),
    ("📅", "<b>Fechas estándar</b>", "Usa formato DD/MM/YYYY o YYYY-MM-DD de forma uniforme."),
]

def render_welcome() -> None:
    st.markdown('<div class="tips-title">¿Cómo preparar tu dataset para mejores resultados?</div>',
                unsafe_allow_html=True)
    cards = "".join(
        f'<div class="tip-card"><span class="tip-icon">{icon}</span>'
        f'<div class="tip-text"><b>{title.replace("<b>","").replace("</b>","")}</b><br>{desc}</div></div>'
        for icon, title, desc in TIPS
    )
    st.markdown(f'<div class="tips-grid">{cards}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Data profiling
# ─────────────────────────────────────────────────────────────────────────────
def detect_col_types(df: pd.DataFrame) -> dict[str, list[str]]:
    numeric       = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical   = df.select_dtypes(include=["object", "category"]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime"]).columns.tolist()
    for col in categorical[:]:
        try:
            df[col] = pd.to_datetime(df[col], infer_datetime_format=True)
            datetime_cols.append(col)
            categorical.remove(col)
        except Exception:
            pass
    return {"numeric": numeric, "categorical": categorical, "datetime": datetime_cols}


def detect_outliers(df: pd.DataFrame, cols: list[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for col in cols:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        n = int(((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum())
        if n:
            result[col] = n
    return result


def profile_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    col_types = detect_col_types(df)
    numeric   = col_types["numeric"]
    cat       = col_types["categorical"]
    profile: dict[str, Any] = {
        "shape":      {"rows": len(df), "cols": len(df.columns)},
        "columns":    list(df.columns),
        "col_types":  col_types,
        "nulls":      df.isnull().sum().to_dict(),
        "null_pct":   (df.isnull().mean() * 100).round(2).to_dict(),
        "duplicates": int(df.duplicated().sum()),
    }
    if numeric:
        profile["numeric_stats"] = df[numeric].describe().round(3).to_dict()
        profile["outliers"]      = detect_outliers(df, numeric)
        if len(numeric) > 1:
            corr  = df[numeric].corr().round(3)
            pairs = [{"col1": c1, "col2": c2, "r": float(corr.loc[c1, c2])}
                     for i, c1 in enumerate(numeric) for c2 in numeric[i + 1:]]
            pairs.sort(key=lambda x: abs(x["r"]), reverse=True)
            profile["top_correlations"] = pairs[:5]
    if cat:
        profile["categorical_info"] = {
            col: {"unique": int(df[col].nunique()),
                  "top_values": df[col].value_counts().head(6).to_dict()}
            for col in cat[:6]
        }
    return profile


# ─────────────────────────────────────────────────────────────────────────────
# AI Agent — Phase 1: Analysis plan
# ─────────────────────────────────────────────────────────────────────────────
def run_analysis_agent(client: Groq, profile: dict, df: pd.DataFrame, filename: str) -> dict:
    numeric   = profile["col_types"]["numeric"]
    cat       = profile["col_types"]["categorical"]
    datetimes = profile["col_types"]["datetime"]

    stats = {col: {"min": round(float(df[col].dropna().min()), 2),
                   "max": round(float(df[col].dropna().max()), 2),
                   "mean": round(float(df[col].dropna().mean()), 2),
                   "std": round(float(df[col].dropna().std()), 2)}
             for col in numeric[:8]}

    cat_tops = {col: {str(k): int(v) for k, v in df[col].value_counts().head(6).items()}
                for col in cat[:5]}

    sample = df.head(4).fillna("").astype(str).to_dict(orient="records")

    prompt = f"""Eres un agente analista de datos de nivel senior. Analiza el siguiente dataset y devuelve UN JSON estructurado.

ARCHIVO: {filename}
FILAS: {profile['shape']['rows']} | COLUMNAS: {profile['shape']['cols']}
NUMÉRICAS: {numeric}
CATEGÓRICAS: {cat}
FECHAS: {datetimes}
NULOS TOTALES: {sum(profile['nulls'].values())}
DUPLICADOS: {profile['duplicates']}
ESTADÍSTICAS: {json.dumps(stats, ensure_ascii=False)}
TOP VALORES: {json.dumps(cat_tops, ensure_ascii=False, default=str)}
MUESTRA: {json.dumps(sample, ensure_ascii=False, default=str)}

Devuelve SOLO JSON válido:
{{
  "dataset_theme": "frase corta describiendo el dataset",
  "key_columns": {{
    "primary_metric": "columna numérica más importante (o null)",
    "time_column": "columna de fecha si existe (o null)",
    "main_category": "columna categórica más relevante (o null)"
  }},
  "insights": [
    {{"title": "Título corto (máx 6 palabras)", "detail": "Explicación con números reales", "type": "positive|negative|neutral"}}
  ],
  "charts": [
    {{
      "type": "bar|line|scatter|box|histogram|treemap",
      "title": "Título descriptivo",
      "x": "columna exacta que existe",
      "y": "columna exacta que existe (o null)",
      "color_by": "columna para colorear (o null)",
      "agg": "sum|mean|count|none",
      "chart_insight": "Una frase: qué historia cuenta este gráfico"
    }}
  ],
  "alerts": [
    {{"message": "Descripción del problema", "severity": "high|medium|low"}}
  ]
}}

REGLAS: insights 3-5 con números reales · charts 3-5 variados · alerts máx 3 solo si hay problemas reales · todas las columnas deben existir exactamente en el dataset."""

    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=2000,
        )
        raw = resp.choices[0].message.content.strip()
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            return json.loads(m.group())
    except Exception as e:
        st.warning(f"Agente: {e}")
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# AI Agent — Narrative
# ─────────────────────────────────────────────────────────────────────────────
def stream_narrative(client: Groq, profile: dict, plan: dict, filename: str):
    prompt = f"""Genera un reporte ejecutivo narrativo en español para el dataset "{filename}".

TEMA: {plan.get('dataset_theme', '')}
INSIGHTS: {json.dumps(plan.get('insights', []), ensure_ascii=False)}
ALERTAS: {json.dumps(plan.get('alerts', []), ensure_ascii=False)}
PERFIL: {json.dumps({k: v for k, v in profile.items() if k != 'col_types'}, ensure_ascii=False, default=str)}

## Resumen Ejecutivo
## Análisis Profundo
## Riesgos y Anomalías
## Recomendaciones Estratégicas

Tono ejecutivo, directo, con números concretos. No inventes datos."""

    stream = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=True, temperature=0.3, max_tokens=2000,
    )
    for chunk in stream:
        t = chunk.choices[0].delta.content
        if t:
            yield t


# ─────────────────────────────────────────────────────────────────────────────
# Smart charts
# ─────────────────────────────────────────────────────────────────────────────
def _theme(fig: go.Figure) -> go.Figure:
    fig.update_layout(**PLOTLY_BASE)
    return fig


def make_smart_charts(df: pd.DataFrame, plan: dict) -> list[tuple[str, go.Figure, dict]]:
    """Returns list of (insight, figure, metadata) tuples."""
    results: list[tuple[str, go.Figure, dict]] = []
    existing = set(df.columns)

    for spec in plan.get("charts", []):
        ctype     = spec.get("type", "bar")
        x_col     = spec.get("x")
        y_col     = spec.get("y")
        color_col = spec.get("color_by")
        title     = spec.get("title", "")
        agg       = spec.get("agg", "none")
        insight   = spec.get("chart_insight", "")

        if x_col and x_col not in existing:
            continue
        if y_col and y_col not in existing:
            y_col = None
        if color_col and color_col not in existing:
            color_col = None

        try:
            df_plot = df.copy()

            if agg in ("sum", "mean", "count") and x_col and y_col:
                grp = df_plot.groupby(x_col)[y_col]
                df_plot   = {"sum": grp.sum, "mean": grp.mean, "count": grp.count}[agg]().reset_index()
                color_col = None

            fig = None
            meta = {"type": ctype, "x_col": x_col, "y_col": y_col, "df": df}

            if ctype == "bar":
                if y_col:
                    fig = px.bar(df_plot, x=x_col, y=y_col,
                                 color=color_col if color_col else y_col,
                                 color_discrete_sequence=CHART_COLORS if color_col else None,
                                 color_continuous_scale=[[0, DANGER], [0.45, WARN], [1, SUCCESS]] if not color_col else None,
                                 title=title)
                    if not color_col:
                        fig.update_coloraxes(showscale=False)
                else:
                    vc = df_plot[x_col].value_counts().head(12).reset_index()
                    vc.columns = [x_col, "count"]
                    fig = px.bar(vc, x=x_col, y="count", color="count",
                                 color_continuous_scale=[[0, ACCENT + "55"], [1, ACCENT]], title=title)
                    fig.update_coloraxes(showscale=False)
                fig.update_traces(marker_cornerradius=4)

            elif ctype == "line":
                if x_col:
                    df_sorted = df_plot.sort_values(x_col)
                    fig = px.line(df_sorted, x=x_col, y=y_col, color=color_col,
                                  color_discrete_sequence=CHART_COLORS, title=title)
                    fig.update_traces(line_width=2.5)
                    if not color_col:
                        fig.update_traces(line_color=ACCENT, fill="tozeroy", fillcolor=ACCENT + "18")

            elif ctype == "scatter":
                fig = px.scatter(df_plot, x=x_col, y=y_col, color=color_col,
                                 color_discrete_sequence=CHART_COLORS, title=title,
                                 trendline="ols" if not color_col else None,
                                 trendline_color_override=SUCCESS)
                fig.update_traces(marker_size=7, marker_opacity=0.75)

            elif ctype == "box":
                target_y = y_col or x_col
                target_x = color_col or (x_col if y_col else None)
                fig = px.box(df_plot, x=target_x, y=target_y, color=target_x,
                             color_discrete_sequence=CHART_COLORS, title=title)
                fig.update_traces(marker_size=4, marker_opacity=0.6)

            elif ctype == "histogram":
                fig = px.histogram(df_plot, x=x_col, color=color_col,
                                   color_discrete_sequence=CHART_COLORS, nbins=30, title=title)
                if not color_col:
                    fig.update_traces(marker_color=ACCENT, marker_line_width=0, marker_opacity=0.85)

            elif ctype == "treemap" and y_col:
                path = [px.Constant("Total")]
                if color_col:
                    path.append(color_col)
                path.append(x_col)
                fig = px.treemap(df_plot, path=path, values=y_col, color=y_col,
                                 color_continuous_scale=[[0, "#1a1a2e"], [0.5, ACCENT], [1, CYAN]],
                                 title=title)

            if fig:
                results.append((insight, _theme(fig), meta))

        except Exception:
            continue

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Treemap department detail panel
# ─────────────────────────────────────────────────────────────────────────────
def render_dept_panel(df: pd.DataFrame, x_col: str, y_col: str | None, label: str) -> None:
    if not label or label in ("Total", "(?)"):
        return
    try:
        dept_df = df[df[x_col].astype(str) == str(label)]
        if dept_df.empty:
            return

        numeric_cols = dept_df.select_dtypes(include=[np.number]).columns.tolist()

        stats_html = ""
        if y_col and y_col in dept_df.columns:
            total  = dept_df[y_col].sum()
            avg    = dept_df[y_col].mean()
            mx     = dept_df[y_col].max()
            mn     = dept_df[y_col].min()
            stats_html += f"""
            <div class="dept-stat"><div class="ds-val">{total:,.1f}</div><div class="ds-lbl">Total {y_col}</div></div>
            <div class="dept-stat"><div class="ds-val">{avg:,.1f}</div><div class="ds-lbl">Promedio</div></div>
            <div class="dept-stat"><div class="ds-val">{mx:,.1f}</div><div class="ds-lbl">Máximo</div></div>
            <div class="dept-stat"><div class="ds-val">{mn:,.1f}</div><div class="ds-lbl">Mínimo</div></div>"""
        else:
            stats_html += f"""<div class="dept-stat"><div class="ds-val">{len(dept_df):,}</div><div class="ds-lbl">Registros</div></div>"""
            for col in numeric_cols[:3]:
                stats_html += f"""<div class="dept-stat"><div class="ds-val">{dept_df[col].mean():,.1f}</div><div class="ds-lbl">Prom. {col}</div></div>"""

        st.markdown(f"""
        <div class="dept-panel">
            <div class="dept-title">📌 {label} — {len(dept_df):,} registros</div>
            <div class="dept-stats">{stats_html}</div>
        </div>
        """, unsafe_allow_html=True)

    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# HTML helpers
# ─────────────────────────────────────────────────────────────────────────────
INSIGHT_STYLE = {
    "positive": (f"background:{SUCCESS}10;border-left-color:{SUCCESS}", f"color:{SUCCESS}", "↑ POSITIVO"),
    "negative": (f"background:{DANGER}10;border-left-color:{DANGER}",   f"color:{DANGER}",  "↓ ALERTA"),
    "neutral":  (f"background:{ACCENT}10;border-left-color:{ACCENT}",   f"color:{ACCENT}",  "→ INFO"),
}
ALERT_STYLE = {
    "high":   (f"background:{DANGER}10;border-left-color:{DANGER}", f"color:{DANGER}", "⚠"),
    "medium": (f"background:{WARN}10;border-left-color:{WARN}",     f"color:{WARN}",   "◆"),
    "low":    (f"background:{ACCENT}10;border-left-color:{ACCENT}", f"color:{ACCENT}", "●"),
}


def metric_card(val: str, lbl: str, cls: str = "") -> str:
    return f'<div class="metric-card {cls}"><div class="val">{val}</div><div class="lbl">{lbl}</div></div>'


def render_insights(insights: list[dict]) -> str:
    cards = ""
    for ins in insights:
        cs, bs, bt = INSIGHT_STYLE.get(ins.get("type", "neutral"), INSIGHT_STYLE["neutral"])
        cards += f'<div class="insight-card" style="{cs}"><div class="ic-badge" style="{bs}">{bt}</div><div class="ic-title">{ins.get("title","")}</div><div class="ic-detail">{ins.get("detail","")}</div></div>'
    return f'<div class="insights-grid">{cards}</div>'


def render_alerts(alerts: list[dict]) -> str:
    html = ""
    for al in alerts:
        cs, is_, ic = ALERT_STYLE.get(al.get("severity", "low"), ALERT_STYLE["low"])
        html += f'<div class="alert-card" style="{cs}"><span class="al-icon" style="{is_}">{ic}</span><span class="al-text">{al.get("message","")}</span></div>'
    return html




# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)

    # Session state init
    for key in ("profile", "plan", "filename", "df", "file_key"):
        if key not in st.session_state:
            st.session_state[key] = None

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("Configura GROQ_API_KEY en tu archivo .env")
        st.stop()
    client = Groq(api_key=api_key)

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="ds-header">
        <h1>DataStory <span>AI</span></h1>
        <p>Sube tu Excel · El agente analiza · La IA construye la historia</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Upload ────────────────────────────────────────────────────────────────
    uploaded = st.file_uploader("", type=["xlsx", "xls"], label_visibility="collapsed")

    if not uploaded:
        render_welcome()
        _render_footer()
        return

    # ── Detect if this is a new file ──────────────────────────────────────────
    file_key = f"{uploaded.name}_{uploaded.size}"
    is_new   = st.session_state.file_key != file_key

    if is_new:
        # ── BIG LOADING STATE ─────────────────────────────────────────────────
        st.markdown("""
        <div class="big-loader">
            <div class="loader-ring"></div>
            <p>Leyendo y validando archivo…</p>
        </div>
        """, unsafe_allow_html=True)

        try:
            df = pd.read_excel(uploaded, engine="openpyxl")
        except Exception as e:
            st.error(f"Error leyendo el archivo: {e}")
            return

        # Clear loading card by rerunning now that df is loaded
        st.session_state.df       = df
        st.session_state.file_key = file_key
        st.session_state.profile  = None
        st.session_state.plan     = None
        st.rerun()

    # ── Use cached data ───────────────────────────────────────────────────────
    df       = st.session_state.df
    filename = uploaded.name

    # ── Phase 1: Profile ──────────────────────────────────────────────────────
    if st.session_state.profile is None:
        with st.status("Perfilando dataset…", expanded=False) as s1:
            profile = profile_dataframe(df)
            st.session_state.profile = profile
            s1.update(label=f"Dataset perfilado · {profile['shape']['rows']:,} filas · {profile['shape']['cols']} columnas",
                      state="complete")
    else:
        profile = st.session_state.profile

    null_total = sum(profile["nulls"].values())
    null_pct   = max(profile["null_pct"].values(), default=0)
    dup_count  = profile["duplicates"]

    st.markdown(f"""
    <div class="metrics-row">
        {metric_card(f"{profile['shape']['rows']:,}", "Filas", "blue")}
        {metric_card(str(profile['shape']['cols']), "Columnas")}
        {metric_card(str(null_total), "Valores nulos", "red" if null_pct > 10 else ("warn" if null_pct > 0 else "green"))}
        {metric_card(str(dup_count), "Duplicados", "red" if dup_count > 0 else "green")}
    </div>
    """, unsafe_allow_html=True)

    with st.expander("Vista previa del dataset", expanded=False):
        st.dataframe(df.head(5), use_container_width=True)

    # ── Phase 2: Agent ────────────────────────────────────────────────────────
    if st.session_state.plan is None:
        with st.status("Agente IA analizando tu dataset…", expanded=False) as s2:
            plan = run_analysis_agent(client, profile, df, filename)
            st.session_state.plan     = plan
            st.session_state.filename = filename
            s2.update(label="Análisis del agente completado", state="complete")
    else:
        plan = st.session_state.plan

    if not plan:
        st.warning("El agente no pudo generar un plan. Intenta de nuevo.")
        return

    theme = plan.get("dataset_theme", "")
    if theme:
        st.markdown(f'<div class="theme-pill">📊 {theme}</div>', unsafe_allow_html=True)

    insights = plan.get("insights", [])
    if insights:
        st.markdown('<div class="section-title">Hallazgos del Agente</div>', unsafe_allow_html=True)
        st.markdown(render_insights(insights), unsafe_allow_html=True)

    alerts = plan.get("alerts", [])
    if alerts:
        st.markdown('<div class="section-title">Alertas de Calidad</div>', unsafe_allow_html=True)
        st.markdown(render_alerts(alerts), unsafe_allow_html=True)

    # ── Phase 3: Charts ───────────────────────────────────────────────────────
    with st.status("Generando visualizaciones inteligentes…", expanded=False) as s3:
        charts = make_smart_charts(df, plan)
        s3.update(label=f"{len(charts)} visualizaciones generadas", state="complete")

    if charts:
        st.markdown('<div class="section-title">Visualizaciones</div>', unsafe_allow_html=True)
        for i in range(0, len(charts), 2):
            c1, c2 = st.columns(2, gap="medium")

            with c1:
                ins_txt, fig, meta = charts[i]
                if meta["type"] == "treemap":
                    state = st.plotly_chart(fig, use_container_width=True,
                                            on_select="rerun", key=f"chart_{i}_L")
                    if ins_txt:
                        st.markdown(f'<div class="chart-insight">↳ {ins_txt}</div>', unsafe_allow_html=True)
                    pts = getattr(getattr(state, "selection", None), "points", [])
                    if pts:
                        render_dept_panel(df, meta["x_col"], meta["y_col"],
                                         pts[0].get("label", ""))
                else:
                    st.plotly_chart(fig, use_container_width=True, key=f"chart_{i}_L")
                    if ins_txt:
                        st.markdown(f'<div class="chart-insight">↳ {ins_txt}</div>', unsafe_allow_html=True)

            if i + 1 < len(charts):
                with c2:
                    ins_txt, fig, meta = charts[i + 1]
                    if meta["type"] == "treemap":
                        state = st.plotly_chart(fig, use_container_width=True,
                                                on_select="rerun", key=f"chart_{i}_R")
                        if ins_txt:
                            st.markdown(f'<div class="chart-insight">↳ {ins_txt}</div>', unsafe_allow_html=True)
                        pts = getattr(getattr(state, "selection", None), "points", [])
                        if pts:
                            render_dept_panel(df, meta["x_col"], meta["y_col"],
                                              pts[0].get("label", ""))
                    else:
                        st.plotly_chart(fig, use_container_width=True, key=f"chart_{i}_R")
                        if ins_txt:
                            st.markdown(f'<div class="chart-insight">↳ {ins_txt}</div>', unsafe_allow_html=True)

    # ── Narrative ─────────────────────────────────────────────────────────────
    with st.expander("✦ Reporte Narrativo Ejecutivo", expanded=False):
        if st.button("Generar reporte completo", type="primary"):
            st.write_stream(stream_narrative(client, profile, plan, filename))

    _render_footer()


def _render_footer() -> None:
    st.markdown("""
    <div class="ds-footer">
        <span>Desarrollado por <b>Juan Uribe</b> · Asistido con IA</span>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
