from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ============================================================
# CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Olist Analytics — Soutenance",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Palette sobre et lisible pour une soutenance
NAVY = "#0F172A"
BLUE = "#2563EB"
CYAN = "#06B6D4"
GREEN = "#10B981"
AMBER = "#F59E0B"
RED = "#EF4444"
PURPLE = "#8B5CF6"
MUTED = "#64748B"
GRID = "rgba(148,163,184,.18)"
PLOT_BG = "rgba(0,0,0,0)"

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.2rem; padding-bottom: 3rem; max-width: 1600px;}
      section[data-testid="stSidebar"] {border-right: 1px solid rgba(148,163,184,.22);}
      div[data-testid="stMetric"] {
        background: linear-gradient(145deg, rgba(37,99,235,.08), rgba(6,182,212,.03));
        border: 1px solid rgba(148,163,184,.22);
        border-radius: 16px;
        padding: 16px 16px 12px 16px;
      }
      div[data-testid="stMetric"] label {font-weight: 650;}
      div[data-testid="stMetricValue"] {font-size: 1.55rem;}
      .olist-hero {
        padding: 18px 22px;
        border-radius: 18px;
        background: linear-gradient(110deg, rgba(37,99,235,.14), rgba(6,182,212,.08), rgba(139,92,246,.07));
        border: 1px solid rgba(148,163,184,.22);
        margin-bottom: 14px;
      }
      .olist-hero h1 {margin:0; font-size: 2rem;}
      .olist-hero p {margin:.35rem 0 0 0; color:#64748B; font-size:1rem;}
      .insight {
        border-left: 4px solid #2563EB;
        background: rgba(37,99,235,.06);
        border-radius: 10px;
        padding: 11px 14px;
        margin: 4px 0 10px 0;
      }
      .filter-pill {
        display:inline-block; padding:4px 9px; margin:2px 4px 2px 0;
        border-radius:999px; background:rgba(37,99,235,.10); color:#1D4ED8;
        font-size:.82rem; font-weight:650;
      }
      .small-muted {color:#64748B; font-size:.88rem;}
      .stTabs [data-baseweb="tab-list"] {gap: 8px;}
      .stTabs [data-baseweb="tab"] {border-radius: 10px; padding: 8px 14px;}
    </style>
    """,
    unsafe_allow_html=True,
)

DATA_DIR = Path("data")

# ============================================================
# OUTILS
# ============================================================
def money(v):
    if pd.isna(v):
        return "—"
    return f"{v:,.0f} R$".replace(",", " ")


def pct(v):
    if pd.isna(v):
        return "—"
    return f"{v:.1f} %"


def nfmt(v):
    if pd.isna(v):
        return "—"
    return f"{int(v):,}".replace(",", " ")


def chart_layout(fig, height=430, legend=True):
    fig.update_layout(
        height=height,
        margin=dict(l=12, r=12, t=55, b=20),
        paper_bgcolor=PLOT_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(family="Arial, sans-serif", size=12),
        hoverlabel=dict(font_size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1) if legend else None,
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False)
    return fig


@st.cache_data(show_spinner=False)
def load_and_prepare(data_dir: str):
    p = Path(data_dir)
    files = {
        "customers": "olist_customers_dataset.csv",
        "geolocation": "olist_geolocation_dataset.csv",
        "items": "olist_order_items_dataset.csv",
        "payments": "olist_order_payments_dataset.csv",
        "reviews": "olist_order_reviews_dataset.csv",
        "orders": "olist_orders_dataset.csv",
        "products": "olist_products_dataset.csv",
        "sellers": "olist_sellers_dataset.csv",
        "translation": "product_category_name_translation.csv",
    }
    missing = [f for f in files.values() if not (p / f).exists()]
    if missing:
        raise FileNotFoundError("Fichiers manquants : " + ", ".join(missing))

    customers = pd.read_csv(p / files["customers"])
    geo = pd.read_csv(p / files["geolocation"])
    items = pd.read_csv(p / files["items"])
    payments = pd.read_csv(p / files["payments"])
    reviews = pd.read_csv(p / files["reviews"])
    orders = pd.read_csv(p / files["orders"])
    products = pd.read_csv(p / files["products"])
    sellers = pd.read_csv(p / files["sellers"])
    translation = pd.read_csv(p / files["translation"])

    # ---------- Dates ----------
    date_cols = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]
    for c in date_cols:
        orders[c] = pd.to_datetime(orders[c], errors="coerce")

    orders["delivery_days"] = (
        orders["order_delivered_customer_date"] - orders["order_purchase_timestamp"]
    ).dt.total_seconds() / 86400
    orders["delay_days"] = (
        orders["order_delivered_customer_date"] - orders["order_estimated_delivery_date"]
    ).dt.total_seconds() / 86400
    orders["is_late"] = np.where(
        orders["delay_days"].notna(), orders["delay_days"] > 0, np.nan
    )
    orders["purchase_month"] = orders["order_purchase_timestamp"].dt.to_period("M").dt.to_timestamp()

    # ---------- Paiements : 1 ligne / commande ----------
    pay_order = (
        payments.groupby("order_id", as_index=False)
        .agg(
            payment_value=("payment_value", "sum"),
            payment_transactions=("payment_sequential", "count"),
            max_installments=("payment_installments", "max"),
        )
    )
    main_pay = (
        payments.groupby(["order_id", "payment_type"], as_index=False)["payment_value"]
        .sum()
        .sort_values(["order_id", "payment_value"], ascending=[True, False])
        .drop_duplicates("order_id")
        .rename(columns={"payment_type": "main_payment_type"})[["order_id", "main_payment_type"]]
    )
    pay_order = pay_order.merge(main_pay, on="order_id", how="left", validate="one_to_one")

    # ---------- Articles : 1 ligne / commande ----------
    item_order = (
        items.groupby("order_id", as_index=False)
        .agg(
            items_count=("order_item_id", "count"),
            unique_products=("product_id", "nunique"),
            unique_sellers=("seller_id", "nunique"),
            products_value=("price", "sum"),
            freight_value=("freight_value", "sum"),
        )
    )

    # ---------- Avis : 1 ligne / commande ----------
    review_order = (
        reviews.groupby("order_id", as_index=False)
        .agg(review_score=("review_score", "mean"), review_count=("review_id", "nunique"))
    )

    # ---------- Catégorie principale par commande ----------
    item_products = (
        items.merge(products, on="product_id", how="left", validate="many_to_one")
        .merge(translation, on="product_category_name", how="left", validate="many_to_one")
    )
    item_products["category"] = (
        item_products["product_category_name_english"]
        .fillna(item_products["product_category_name"])
        .fillna("unknown")
    )
    cat_value = (
        item_products.groupby(["order_id", "category"], as_index=False)["price"].sum()
        .sort_values(["order_id", "price"], ascending=[True, False])
        .drop_duplicates("order_id")
        .rename(columns={"category": "main_category"})[["order_id", "main_category"]]
    )

    # ---------- Table analytique finale ----------
    df = (
        orders.merge(customers, on="customer_id", how="left", validate="one_to_one")
        .merge(pay_order, on="order_id", how="left", validate="one_to_one")
        .merge(item_order, on="order_id", how="left", validate="one_to_one")
        .merge(review_order, on="order_id", how="left", validate="one_to_one")
        .merge(cat_value, on="order_id", how="left", validate="one_to_one")
    )
    df["delivery_status"] = np.select(
        [df["delay_days"].isna(), df["delay_days"] > 0],
        ["Non disponible", "En retard"],
        default="À l'heure / en avance",
    )

    # ---------- Coordonnées locales, sans GeoJSON / API ----------
    # On retire quelques coordonnées impossibles puis on calcule un centre robuste par préfixe postal.
    geo_clean = geo[
        geo["geolocation_lat"].between(-35, 6)
        & geo["geolocation_lng"].between(-75, -32)
    ].copy()
    zip_geo = (
        geo_clean.groupby("geolocation_zip_code_prefix", as_index=False)
        .agg(lat=("geolocation_lat", "median"), lon=("geolocation_lng", "median"))
    )
    cust_geo = customers.merge(
        zip_geo,
        left_on="customer_zip_code_prefix",
        right_on="geolocation_zip_code_prefix",
        how="left",
        validate="many_to_one",
    )
    state_coords = (
        cust_geo.dropna(subset=["lat", "lon"])
        .groupby("customer_state", as_index=False)
        .agg(lat=("lat", "median"), lon=("lon", "median"))
    )

    # Nuage de points géographiques Olist : sert de fond de carte local/offline.
    if len(geo_clean) > 7000:
        geo_background = geo_clean.sample(7000, random_state=42)[["geolocation_lat", "geolocation_lng"]]
    else:
        geo_background = geo_clean[["geolocation_lat", "geolocation_lng"]]

    return {
        "df": df,
        "orders": orders,
        "customers": customers,
        "payments": payments,
        "reviews": reviews,
        "items": items,
        "products": products,
        "sellers": sellers,
        "item_products": item_products,
        "state_coords": state_coords,
        "geo_background": geo_background,
    }


try:
    D = load_and_prepare(str(DATA_DIR))
except Exception as e:
    st.error("Le dashboard ne peut pas démarrer car les fichiers Olist n'ont pas été trouvés ou ne correspondent pas au schéma attendu.")
    st.code(str(e))
    st.markdown(
        "Place les **9 CSV Kaggle Olist** dans un dossier `data/` situé au même niveau que ce fichier `app_olist_premium.py`."
    )
    st.stop()

DF = D["df"]

# ============================================================
# FILTRES — simples, explicites et réellement appliqués
# ============================================================
all_states = sorted(DF["customer_state"].dropna().astype(str).unique())
all_status = sorted(DF["order_status"].dropna().astype(str).unique())
all_payments = sorted(DF["main_payment_type"].dropna().astype(str).unique())
all_delivery = ["Tous", "À l'heure / en avance", "En retard", "Non disponible"]
all_categories = sorted(DF["main_category"].dropna().astype(str).unique())

valid_dates = DF["order_purchase_timestamp"].dropna()
min_date = valid_dates.min().date()
max_date = valid_dates.max().date()

filter_keys = ["f_dates", "f_state", "f_status", "f_payment", "f_delivery", "f_category"]

with st.sidebar:
    st.markdown("## 🎛️ Filtres")
    st.caption("Tu peux sélectionner plusieurs États. Modifie les valeurs puis clique sur **Appliquer**.")

    with st.form("filters_form", clear_on_submit=False):
        dates = st.slider(
            "Période",
            min_value=min_date,
            max_value=max_date,
            value=(min_date, max_date),
            key="f_dates",
        )
        state_choices = st.multiselect(
            "État client — sélection multiple",
            options=all_states,
            default=[],
            key="f_state",
            help="Laisse vide pour conserver tous les États."
        )
        status_choice = st.selectbox("Statut commande", ["Tous"] + all_status, key="f_status")
        payment_choice = st.selectbox("Paiement principal", ["Tous"] + all_payments, key="f_payment")
        delivery_choice = st.selectbox("Livraison", all_delivery, key="f_delivery")
        category_choice = st.selectbox("Catégorie principale", ["Toutes"] + all_categories, key="f_category")
        apply_filters = st.form_submit_button("✅ Appliquer les filtres", type="primary", use_container_width=True)

    def _reset_filters():
        for k in filter_keys:
            st.session_state.pop(k, None)

    st.button("↺ Réinitialiser", use_container_width=True, on_click=_reset_filters)

# Les valeurs des widgets sont disponibles même sans clic ; le formulaire garantit une mise à jour groupée.
start_date, end_date = dates
start_ts = pd.Timestamp(start_date)
end_ts = pd.Timestamp(end_date) + pd.Timedelta(days=1)

filtered = DF[
    (DF["order_purchase_timestamp"] >= start_ts)
    & (DF["order_purchase_timestamp"] < end_ts)
].copy()

if state_choices:
    filtered = filtered[filtered["customer_state"].isin(state_choices)]
if status_choice != "Tous":
    filtered = filtered[filtered["order_status"] == status_choice]
if payment_choice != "Tous":
    filtered = filtered[filtered["main_payment_type"] == payment_choice]
if delivery_choice != "Tous":
    filtered = filtered[filtered["delivery_status"] == delivery_choice]
if category_choice != "Toutes":
    filtered = filtered[filtered["main_category"] == category_choice]

# Comparaison avec période précédente en conservant les filtres catégoriels
period_days = max((pd.Timestamp(end_date) - pd.Timestamp(start_date)).days + 1, 1)
prev_end = pd.Timestamp(start_date)
prev_start = prev_end - pd.Timedelta(days=period_days)
previous = DF[
    (DF["order_purchase_timestamp"] >= prev_start)
    & (DF["order_purchase_timestamp"] < prev_end)
].copy()
if state_choices: previous = previous[previous["customer_state"].isin(state_choices)]
if status_choice != "Tous": previous = previous[previous["order_status"] == status_choice]
if payment_choice != "Tous": previous = previous[previous["main_payment_type"] == payment_choice]
if delivery_choice != "Tous": previous = previous[previous["delivery_status"] == delivery_choice]
if category_choice != "Toutes": previous = previous[previous["main_category"] == category_choice]

with st.sidebar:
    st.divider()
    st.metric("Commandes affichées", nfmt(filtered["order_id"].nunique()))
    st.caption(f"{len(filtered):,} lignes analytiques — grain : 1 ligne = 1 commande".replace(",", " "))
    if filtered.empty:
        st.error("Aucune donnée pour ces filtres.")

if filtered.empty:
    st.warning("Aucune commande ne correspond aux filtres sélectionnés. Réinitialise les filtres ou élargis la période.")
    st.stop()

order_ids = set(filtered["order_id"])

# ============================================================
# EN-TÊTE + FILTRES ACTIFS
# ============================================================
st.markdown(
    """
    <div class="olist-hero">
      <h1>Olist Analytics — Dashboard de soutenance</h1>
      <p>Ventes • expérience client • logistique • géographie • corrélations</p>
    </div>
    """,
    unsafe_allow_html=True,
)

active = [f"📅 {start_date.strftime('%d/%m/%Y')} → {end_date.strftime('%d/%m/%Y')}"]
if state_choices:
    if len(state_choices) <= 4:
        active.append("📍 " + ", ".join(state_choices))
    else:
        active.append(f"📍 {len(state_choices)} États sélectionnés")
if status_choice != "Tous": active.append(f"📦 {status_choice}")
if payment_choice != "Tous": active.append(f"💳 {payment_choice}")
if delivery_choice != "Tous": active.append(f"🚚 {delivery_choice}")
if category_choice != "Toutes": active.append(f"🏷️ {category_choice}")
st.markdown(" ".join([f'<span class="filter-pill">{x}</span>' for x in active]), unsafe_allow_html=True)

# ============================================================
# KPI + deltas période précédente
# ============================================================
def kpis(d):
    delivered = d[d["is_late"].notna()]
    return {
        "orders": d["order_id"].nunique(),
        "customers": d["customer_unique_id"].nunique(),
        "paid": d["payment_value"].sum(),
        "basket": d["payment_value"].mean(),
        "review": d["review_score"].mean(),
        "late": pd.to_numeric(delivered["is_late"], errors="coerce").mean() * 100 if len(delivered) else np.nan,
    }

K = kpis(filtered)
P = kpis(previous)

def delta_pct(cur, prev):
    if pd.isna(cur) or pd.isna(prev) or prev == 0:
        return None
    return f"{((cur / prev) - 1) * 100:+.1f}%"

cols = st.columns(6)
cols[0].metric("Commandes", nfmt(K["orders"]), delta_pct(K["orders"], P["orders"]))
cols[1].metric("Clients uniques", nfmt(K["customers"]), delta_pct(K["customers"], P["customers"]))
cols[2].metric("Montant payé", money(K["paid"]), delta_pct(K["paid"], P["paid"]))
cols[3].metric("Panier moyen", money(K["basket"]), delta_pct(K["basket"], P["basket"]))
cols[4].metric("Note moyenne", "—" if pd.isna(K["review"]) else f"{K['review']:.2f} / 5", None)
cols[5].metric("Taux de retard", pct(K["late"]), None)

# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Cockpit",
    "💰 Ventes",
    "❤️ Expérience client",
    "📍 Géographie",
    "🧠 Corrélations & qualité",
])

# ============================================================
# 1 — COCKPIT
# ============================================================
with tab1:
    st.subheader("Vue exécutive")

    monthly = (
        filtered.dropna(subset=["purchase_month"])
        .groupby("purchase_month", as_index=False)
        .agg(montant=("payment_value", "sum"), commandes=("order_id", "nunique"))
        .sort_values("purchase_month")
    )
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(x=monthly["purchase_month"], y=monthly["montant"], name="Montant payé", marker_color=BLUE, opacity=.72),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=monthly["purchase_month"], y=monthly["commandes"], name="Commandes", mode="lines+markers", line=dict(color=AMBER, width=3)),
        secondary_y=True,
    )
    fig.update_layout(title="Activité mensuelle — valeur + volume", hovermode="x unified")
    fig.update_yaxes(title_text="Montant payé (R$)", secondary_y=False)
    fig.update_yaxes(title_text="Nombre de commandes", secondary_y=True, showgrid=False)
    chart_layout(fig, 450)
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

    c1, c2 = st.columns([1, 1])
    with c1:
        paymix = (
            filtered["main_payment_type"].fillna("inconnu").value_counts().rename_axis("Paiement").reset_index(name="Commandes")
        )
        fig_pay = px.pie(
            paymix, names="Paiement", values="Commandes", hole=.62,
            title="Mix des paiements", color_discrete_sequence=[BLUE, CYAN, PURPLE, AMBER, GREEN, RED]
        )
        fig_pay.update_traces(textposition="inside", textinfo="percent+label")
        chart_layout(fig_pay, 410, legend=False)
        st.plotly_chart(fig_pay, use_container_width=True, config={"displaylogo": False})

    with c2:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=0 if pd.isna(K["late"]) else K["late"],
            number={"suffix": "%", "font": {"size": 42}},
            title={"text": "Taux de retard parmi les commandes livrées"},
            gauge={
                "axis": {"range": [0, max(25, (0 if pd.isna(K['late']) else K['late']) * 1.8)]},
                "bar": {"color": RED},
                "steps": [
                    {"range": [0, 5], "color": "rgba(16,185,129,.18)"},
                    {"range": [5, 12], "color": "rgba(245,158,11,.18)"},
                    {"range": [12, 100], "color": "rgba(239,68,68,.14)"},
                ],
            },
        ))
        chart_layout(gauge, 410, legend=False)
        st.plotly_chart(gauge, use_container_width=True, config={"displaylogo": False})

    # Insight dynamique
    top_state = filtered.groupby("customer_state")["order_id"].nunique().sort_values(ascending=False)
    top_pay = filtered["main_payment_type"].value_counts()
    if len(top_state) and len(top_pay):
        st.markdown(
            f'<div class="insight"><b>Lecture rapide :</b> {top_state.index[0]} concentre le plus de commandes dans la sélection '
            f'({nfmt(top_state.iloc[0])}). Le paiement principal dominant est <b>{top_pay.index[0]}</b>. '
            f'La note moyenne est de <b>{K["review"]:.2f}/5</b>.</div>',
            unsafe_allow_html=True,
        )

# ============================================================
# 2 — VENTES
# ============================================================
with tab2:
    st.subheader("Structure des ventes")
    ip = D["item_products"]
    ipf = ip[ip["order_id"].isin(order_ids)].copy()

    cat = (
        ipf.groupby("category", as_index=False)
        .agg(valeur=("price", "sum"), articles=("order_item_id", "count"), commandes=("order_id", "nunique"))
        .sort_values("valeur", ascending=False)
    )
    cat_top = cat.head(18)

    c1, c2 = st.columns([1.2, .8])
    with c1:
        fig_tree = px.treemap(
            cat_top,
            path=["category"], values="valeur", color="commandes",
            color_continuous_scale="Blues",
            hover_data={"articles": True, "commandes": True, "valeur": ":,.0f"},
            title="Catégories — poids financier et volume de commandes",
        )
        chart_layout(fig_tree, 510, legend=False)
        st.plotly_chart(fig_tree, use_container_width=True, config={"displaylogo": False})

    with c2:
        status = filtered["order_status"].value_counts().rename_axis("Statut").reset_index(name="Commandes")
        status["Part"] = status["Commandes"] / status["Commandes"].sum() * 100
        fig_status = px.bar(
            status.sort_values("Commandes"), x="Commandes", y="Statut", orientation="h",
            text="Part", title="Statuts de commande", color="Commandes", color_continuous_scale="Blues"
        )
        fig_status.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        chart_layout(fig_status, 510, legend=False)
        st.plotly_chart(fig_status, use_container_width=True, config={"displaylogo": False})

    vals = filtered["payment_value"].dropna()
    if len(vals):
        q99 = vals.quantile(.99)
        zoom = vals[vals <= q99]
        fig_hist = px.histogram(
            x=zoom, nbins=55, marginal="box",
            labels={"x": "Montant payé (R$)", "y": "Commandes"},
            title=f"Distribution des montants — affichage jusqu'au 99e percentile ({q99:.0f} R$)",
            color_discrete_sequence=[CYAN],
        )
        fig_hist.add_vline(x=zoom.median(), line_dash="dash", line_color=AMBER, annotation_text=f"Médiane {zoom.median():.0f} R$")
        chart_layout(fig_hist, 480, legend=False)
        st.plotly_chart(fig_hist, use_container_width=True, config={"displaylogo": False})

# ============================================================
# 3 — EXPERIENCE CLIENT
# ============================================================
with tab3:
    st.subheader("Livraison ↔ satisfaction")

    sat = filtered[
        filtered["delivery_status"].isin(["En retard", "À l'heure / en avance"])
        & filtered["review_score"].notna()
    ].copy()
    sat["note"] = sat["review_score"].round().clip(1, 5).astype(int)

    c1, c2 = st.columns([1.05, .95])

    with c1:
        if len(sat):
            dist = pd.crosstab(sat["delivery_status"], sat["note"], normalize="index") * 100
            dist = dist.reindex(["En retard", "À l'heure / en avance"]).fillna(0)
            for note in range(1, 6):
                if note not in dist.columns: dist[note] = 0
            fig_dist = go.Figure()
            score_colors = {1: RED, 2: "#FB7185", 3: AMBER, 4: CYAN, 5: GREEN}
            for note in range(1, 6):
                vals_note = dist[note]
                fig_dist.add_bar(
                    y=dist.index, x=vals_note, name=f"{note} ★", orientation="h",
                    marker_color=score_colors[note],
                    text=[f"{v:.1f}%" if v >= 4 else "" for v in vals_note], textposition="inside",
                )
            fig_dist.update_layout(barmode="stack", title="Distribution complète des notes", xaxis_title="Part des avis (%)", xaxis_range=[0, 100])
            chart_layout(fig_dist, 440)
            st.plotly_chart(fig_dist, use_container_width=True, config={"displaylogo": False})

    with c2:
        # Barres empilées : effectifs par niveau de satisfaction
        if len(sat):
            sat["groupe_note"] = pd.cut(
                sat["note"],
                bins=[0, 2, 3, 5],
                labels=["1–2 ★", "3 ★", "4–5 ★"]
            )

            stack_counts = pd.crosstab(
                sat["delivery_status"],
                sat["groupe_note"]
            ).reindex(
                index=["En retard", "À l'heure / en avance"],
                columns=["1–2 ★", "3 ★", "4–5 ★"],
                fill_value=0
            )

            fig_stack = go.Figure()
            satisfaction_colors = {
                "1–2 ★": RED,
                "3 ★": AMBER,
                "4–5 ★": GREEN,
            }

            for group in ["1–2 ★", "3 ★", "4–5 ★"]:
                values = stack_counts[group]
                fig_stack.add_bar(
                    x=stack_counts.index,
                    y=values,
                    name=group,
                    marker_color=satisfaction_colors[group],
                    text=[
                        f"{int(v):,}".replace(",", " ") if v > 0 else ""
                        for v in values
                    ],
                    textposition="inside",
                    hovertemplate=(
                        "<b>%{x}</b><br>"
                        + group
                        + " : %{y:,.0f} avis<extra></extra>"
                    ),
                )

            fig_stack.update_layout(
                barmode="stack",
                title="Satisfaction par ponctualité — effectifs empilés",
                xaxis_title="Respect du délai",
                yaxis_title="Nombre d'avis",
                legend_title="Niveau de satisfaction",
            )
            chart_layout(fig_stack, 440)
            st.plotly_chart(
                fig_stack,
                use_container_width=True,
                config={"displaylogo": False}
            )

    delay = filtered.dropna(subset=["delay_days", "review_score"]).copy()
    delay["note"] = delay["review_score"].round().clip(1, 5).astype(int)
    if len(delay):
        delay["classe_delai"] = pd.cut(
            delay["delay_days"],
            bins=[-np.inf, -7, 0, 3, 7, 30, np.inf],
            labels=["Avance > 7 j", "À l'heure / avance ≤ 7 j", "Retard 0–3 j", "Retard 3–7 j", "Retard 7–30 j", "Retard > 30 j"],
        )
        heat = pd.crosstab(delay["classe_delai"], delay["note"], normalize="index") * 100
        for note in range(1, 6):
            if note not in heat.columns: heat[note] = 0
        heat = heat[[1,2,3,4,5]]
        fig_heat = px.imshow(
            heat, text_auto=".1f", aspect="auto", color_continuous_scale="RdYlGn",
            labels={"x": "Note client", "y": "Écart à la date estimée", "color": "Part (%)"},
            title="Comment la note évolue avec l'intensité du retard",
        )
        chart_layout(fig_heat, 500, legend=False)
        st.plotly_chart(fig_heat, use_container_width=True, config={"displaylogo": False})

        rho = delay["delay_days"].corr(delay["review_score"], method="spearman")
        st.markdown(
            f'<div class="insight"><b>Corrélation de Spearman retard ↔ note :</b> ρ = <b>{rho:.3f}</b>. '
            f'Une valeur négative signifie qu’un retard plus important est associé à une note plus faible. '
            f'Il s’agit d’une association, pas d’une preuve de causalité.</div>',
            unsafe_allow_html=True,
        )

# ============================================================
# 4 — GEOGRAPHIE (100 % LOCALE, sans API ni GeoJSON)
# ============================================================
with tab4:
    st.subheader("Carte analytique Olist — sans dépendance externe")
    st.caption("Le fond est construit à partir des coordonnées du fichier Olist `geolocation`; aucune API cartographique ni GeoJSON externe n'est nécessaire.")

    sm = (
        filtered.groupby("customer_state", as_index=False)
        .agg(
            commandes=("order_id", "nunique"),
            clients=("customer_unique_id", "nunique"),
            montant=("payment_value", "sum"),
            panier=("payment_value", "mean"),
            note=("review_score", "mean"),
            retard=("is_late", lambda s: pd.to_numeric(s, errors="coerce").mean() * 100),
        )
        .merge(D["state_coords"], on="customer_state", how="left")
        .dropna(subset=["lat", "lon"])
    )

    bg = D["geo_background"]

    # Carte Plotly "geo" : pas de tuiles externes, mais un vrai fond de carte visible
    fig_map = go.Figure()

    # Nuage de fond : empreinte brute des coordonnées Olist
    fig_map.add_trace(go.Scattergeo(
        lon=bg["geolocation_lng"],
        lat=bg["geolocation_lat"],
        mode="markers",
        marker=dict(size=2, color="rgba(100,116,139,0.18)"),
        hoverinfo="skip",
        name="Coordonnées Olist",
        showlegend=False,
    ))

    if len(sm):
        max_orders = max(sm["commandes"].max(), 1)
        sizes = 14 + 42 * np.sqrt(sm["commandes"] / max_orders)
        custom = np.column_stack([
            sm["commandes"], sm["clients"], sm["montant"], sm["panier"], sm["note"], sm["retard"]
        ])

        fig_map.add_trace(go.Scattergeo(
            lon=sm["lon"],
            lat=sm["lat"],
            mode="markers+text",
            text=sm["customer_state"],
            textposition="middle center",
            textfont=dict(color="white", size=10),
            marker=dict(
                size=sizes,
                color=sm["retard"],
                colorscale="Turbo",
                showscale=True,
                colorbar=dict(title="Retard %"),
                line=dict(color="white", width=1.2),
                opacity=0.92,
            ),
            customdata=custom,
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Commandes : %{customdata[0]:,.0f}<br>"
                "Clients : %{customdata[1]:,.0f}<br>"
                "Montant : %{customdata[2]:,.0f} R$<br>"
                "Panier : %{customdata[3]:.1f} R$<br>"
                "Note : %{customdata[4]:.2f}/5<br>"
                "Retard : %{customdata[5]:.1f}%<extra></extra>"
            ),
            name="États",
            showlegend=False,
        ))

    fig_map.update_geos(
        scope="south america",
        projection_type="equirectangular",
        center=dict(lat=-14, lon=-52),
        lonaxis_range=[-76, -31],
        lataxis_range=[-35, 6],
        showland=True,
        landcolor="#E5E7EB",
        showcountries=True,
        countrycolor="white",
        showocean=True,
        oceancolor="#DBEAFE",
        showlakes=True,
        lakecolor="#DBEAFE",
        showrivers=False,
        showcoastlines=True,
        coastlinecolor="#94A3B8",
        showframe=False,
        bgcolor="#F8FAFC",
    )

    fig_map.update_layout(
        title="Empreinte géographique des commandes — taille = volume, couleur = taux de retard",
        paper_bgcolor=PLOT_BG,
        plot_bgcolor="#F8FAFC",
        margin=dict(l=10, r=10, t=60, b=10),
        height=650,
    )

    st.plotly_chart(fig_map, use_container_width=True, config={"displaylogo": False})

    c1, c2 = st.columns([1.05, .95])
    with c1:
        sm2 = sm.dropna(subset=["note", "retard", "montant"]).copy()
        fig_bubble = px.scatter(
            sm2, x="commandes", y="note", size="montant", color="retard", text="customer_state",
            log_x=True, size_max=58, color_continuous_scale="RdYlGn_r",
            hover_data={"panier": ":.1f", "clients": ":,", "montant": ":,.0f", "retard": ":.1f"},
            labels={"commandes": "Commandes (échelle log)", "note": "Note moyenne", "retard": "Retard (%)"},
            title="Performance commerciale & expérience par État",
        )
        fig_bubble.update_traces(textposition="top center")
        chart_layout(fig_bubble, 520, legend=False)
        st.plotly_chart(fig_bubble, use_container_width=True, config={"displaylogo": False})
    with c2:
        top = sm.sort_values("commandes", ascending=False).head(12).sort_values("commandes")
        fig_rank = px.bar(
            top, x="commandes", y="customer_state", orientation="h", color="retard",
            color_continuous_scale="RdYlGn_r", text="commandes",
            title="Top États — volume et taux de retard",
            labels={"customer_state": "État", "commandes": "Commandes", "retard": "Retard (%)"},
        )
        fig_rank.update_traces(textposition="outside")
        chart_layout(fig_rank, 520, legend=False)
        st.plotly_chart(fig_rank, use_container_width=True, config={"displaylogo": False})

# ============================================================
# 5 — CORRELATIONS & QUALITE
# ============================================================
with tab5:
    st.subheader("Relations entre variables et contrôles de qualité")
    corr_cols = [
        "payment_value", "products_value", "freight_value", "items_count",
        "unique_products", "unique_sellers", "payment_transactions",
        "max_installments", "delivery_days", "delay_days", "review_score",
    ]
    corr_cols = [c for c in corr_cols if c in filtered.columns]
    corr = filtered[corr_cols].corr(method="spearman")

    c1, c2 = st.columns([1.25, .75])
    with c1:
        fig_corr = px.imshow(
            corr, text_auto=".2f", zmin=-1, zmax=1, color_continuous_scale="RdBu_r",
            labels={"color": "ρ"}, title="Matrice de corrélation de Spearman", aspect="auto"
        )
        chart_layout(fig_corr, 620, legend=False)
        st.plotly_chart(fig_corr, use_container_width=True, config={"displaylogo": False})
    with c2:
        mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
        pairs = corr.where(mask).stack().reset_index()
        pairs.columns = ["Variable 1", "Variable 2", "rho"]
        pairs["force"] = pairs["rho"].abs()
        pairs = pairs.sort_values("force", ascending=False).head(10)
        pairs["relation"] = pairs["Variable 1"] + " ↔ " + pairs["Variable 2"]
        fig_top = px.bar(
            pairs.sort_values("rho"), x="rho", y="relation", orientation="h", color="rho",
            color_continuous_scale="RdBu_r", range_color=[-1,1], text="rho",
            title="Top 10 des associations"
        )
        fig_top.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        chart_layout(fig_top, 620, legend=False)
        st.plotly_chart(fig_top, use_container_width=True, config={"displaylogo": False})

    st.markdown("### Qualité et intégrité")
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Doublons order_id", nfmt(DF["order_id"].duplicated().sum()))
    q2.metric("Review manquante", nfmt(DF["review_score"].isna().sum()))
    q3.metric("Délai manquant", nfmt(DF["delivery_days"].isna().sum()))
    q4.metric("Commandes = lignes", "Oui" if len(DF) == DF["order_id"].nunique() else "Non")

    st.markdown(
        '<div class="insight"><b>Pourquoi Spearman ?</b> Plusieurs variables sont asymétriques et contiennent des valeurs extrêmes ; '
        'la note client est en plus ordinale. Spearman mesure une association monotone plus robuste. '
        '<b>Corrélation ≠ causalité.</b></div>',
        unsafe_allow_html=True,
    )

# ============================================================
# METHODOLOGIE
# ============================================================
st.divider()
with st.expander("🔎 Méthodologie — ce qu'il faut expliquer au professeur"):
    st.markdown(
        """
        **Grain :** une ligne de la table analytique = une commande.

        **Jointures :** `payments`, `order_items` et `reviews` sont d'abord agrégés par `order_id` avant jointure. Cela empêche la multiplication des lignes liée aux relations One-to-Many.

        **Clients :** `customer_unique_id` permet de reconnaître un même acheteur sur plusieurs commandes.

        **Valeurs manquantes :** elles ne sont pas supprimées automatiquement ; elles sont interprétées selon le contexte métier.

        **Outliers :** ils restent dans les calculs. Certains graphiques sont seulement zoomés au 99e percentile pour rester lisibles.

        **Carte :** elle est construite uniquement à partir de `olist_geolocation_dataset.csv`. Elle ne dépend ni d'un GeoJSON externe, ni d'une API cartographique : elle continue donc de fonctionner hors connexion.
        """
    )
