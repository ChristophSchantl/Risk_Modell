import os, json
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
from scipy.stats import norm

STATE_FILE = "portfolio_inputs.json"

DEFAULT = pd.DataFrame([
    {"ticker":"REI","shares":17400,"entry":0.72,"entry_ccy":"USD"},
    {"ticker":"NVO","shares":400,"entry":36.50,"entry_ccy":"USD"},
    {"ticker":"PYPL","shares":280,"entry":62.10,"entry_ccy":"USD"},
    {"ticker":"SRPT","shares":950,"entry":109.30,"entry_ccy":"USD"},
    {"ticker":"LULU","shares":80,"entry":176.00,"entry_ccy":"USD"},
    {"ticker":"CMCSA","shares":630,"entry":42.50,"entry_ccy":"USD"},
    {"ticker":"CAG","shares":970,"entry":29.60,"entry_ccy":"USD"},
    {"ticker":"VIXL.L","shares":3000,"entry":2.31,"entry_ccy":"EUR"},
])

DEFAULT_CASH = 40805.85
DEFAULT_BENCH = "^GSPC"
DEFAULT_FX = "EURUSD=X"

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            d = json.load(f)
        return pd.DataFrame(d["portfolio"]), float(d["cash"]), d["bench"], d["fx"]
    return DEFAULT.copy(), DEFAULT_CASH, DEFAULT_BENCH, DEFAULT_FX

def save_state(df, cash, bench, fx):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump({"portfolio": df.to_dict("records"), "cash": cash, "bench": bench, "fx": fx}, f, indent=2)

@st.cache_data(ttl=1800)
def dl_close(tickers):
    px = yf.download(tickers=tickers, auto_adjust=True, progress=False)["Close"]
    if isinstance(px, pd.Series):
        px = px.to_frame()
    return px.dropna(how="all")

def infer_usd_like(cols):
    return [c for c in cols if (("." not in c) and (not c.endswith("=X")) and (not c.startswith("^")))]

def var_hist(pnl, lvl):
    x = pnl.dropna().to_numpy()
    if x.size < 50: return np.nan
    return max(0.0, -np.quantile(x, 1-lvl))

def es_hist(pnl, lvl):
    x = pnl.dropna().to_numpy()
    if x.size < 50: return np.nan
    q = np.quantile(x, 1-lvl)
    tail = x[x <= q]
    return max(0.0, -tail.mean()) if tail.size else np.nan

def var_param(pnl, lvl):
    x = pnl.dropna().to_numpy()
    if x.size < 50: return np.nan
    mu, sig = float(np.mean(x)), float(np.std(x, ddof=1))
    return max(0.0, norm.ppf(lvl)*sig - mu)

def es_param(pnl, lvl):
    x = pnl.dropna().to_numpy()
    if x.size < 50: return np.nan
    mu, sig = float(np.mean(x)), float(np.std(x, ddof=1))
    z = float(norm.ppf(lvl)); phi = float(norm.pdf(z))
    return max(0.0, (phi/(1-lvl))*sig - mu)

def compute_all(df, cash, bench, fx_ticker, lookback):
    tickers = df["ticker"].tolist()
    all_t = list(dict.fromkeys(tickers + [bench, fx_ticker]))
    px_all = dl_close(all_t)

    if fx_ticker not in px_all.columns:
        raise ValueError(f"FX {fx_ticker} nicht ladbar")

    eurusd = px_all[fx_ticker].dropna()
    px = px_all.drop(columns=[fx_ticker]).copy()

    usd_cols = infer_usd_like(px.columns.tolist())
    fx = eurusd.reindex(px.index).ffill()
    for c in usd_cols:
        if c in px.columns:
            px[c] = px[c] / fx

    px = px.dropna()
    px = px.tail(lookback)

    # usable tickers
    have = set(px.columns)
    tickers = [t for t in tickers if t in have]
    if bench not in have:
        raise ValueError(f"Benchmark {bench} nicht in Daten")

    last = px.iloc[-1]
    df2 = df[df["ticker"].isin(tickers)].copy()
    df2["price_eur"] = df2["ticker"].map(last)
    df2["value_eur"] = df2["price_eur"] * df2["shares"]

    total = float(df2["value_eur"].sum() + cash)
    df2["weight"] = df2["value_eur"] / total

    rets = px.pct_change().dropna()
    port_rets = (rets[tickers] * df2.set_index("ticker")["weight"]).sum(axis=1)
    port_pnl = port_rets * total

    # entry conversion for unrealized pnl
    eurusd_last = float(eurusd.reindex(px.index).ffill().iloc[-1])
    entry_eur = []
    for _, r in df2.iterrows():
        ep = r["entry"]/eurusd_last if r["entry_ccy"] == "USD" else r["entry"]
        entry_eur.append(ep)
    df2["entry_eur"] = entry_eur
    df2["unreal_pnl_eur"] = (df2["price_eur"] - df2["entry_eur"]) * df2["shares"]
    df2["unreal_pnl_pct"] = df2["price_eur"] / df2["entry_eur"] - 1.0

    corr = rets[tickers].corr()

    return px, rets, df2, total, port_rets, port_pnl, corr

def risk_table(port_pnl, levels, use_param=False):
    rows = []
    for lvl in levels:
        rows.append({
            "level": lvl,
            "VaR_hist": var_hist(port_pnl, lvl),
            "ES_hist": es_hist(port_pnl, lvl),
            "VaR_param": var_param(port_pnl, lvl) if use_param else np.nan,
            "ES_param": es_param(port_pnl, lvl) if use_param else np.nan,
        })
    return pd.DataFrame(rows)

def what_if_trim(df2, cash, bench, fx, lookback, levels, use_param, trim_map):
    # trim_map: ticker -> trim_eur
    df_new = df2[["ticker","shares","entry","entry_ccy"]].copy()
    # convert trim_eur to shares using current price_eur
    price = df2.set_index("ticker")["price_eur"].to_dict()
    shares_new = []
    cash_add = 0.0
    for _, r in df_new.iterrows():
        t = r["ticker"]
        sh = float(r["shares"])
        tr = float(trim_map.get(t, 0.0))
        if tr > 0 and price.get(t, np.nan) > 0:
            sh_sell = min(sh, tr / price[t])
            sh = sh - sh_sell
            cash_add += sh_sell * price[t]
        shares_new.append(sh)
    df_new["shares"] = shares_new
    cash_new = cash + cash_add

    px, rets, df_new2, total, port_rets, port_pnl, corr = compute_all(df_new, cash_new, bench, fx, lookback)
    rt = risk_table(port_pnl, levels, use_param=use_param)
    return df_new2, cash_new, total, port_pnl, rt

# ------------------- UI -------------------
st.set_page_config(layout="wide")
st.title("Portfolio Risiko-Analyse — VaR/ES, Korrelation, P&L, Trim-Optimizer (stabil)")

df, cash, bench, fx = load_state()

with st.sidebar:
    st.header("Settings")
    cash = st.number_input("Cash (EUR)", value=float(cash), step=1000.0)
    bench = st.text_input("Benchmark", value=bench)
    fx = st.text_input("FX (USD/EUR)", value=fx)
    lookback = st.slider("Lookback (Tage)", 252, 1500, 756)
    levels = st.multiselect("Levels", [0.90,0.95,0.975,0.99], default=[0.95,0.99])
    use_param = st.checkbox("Parametric (Normal) zusätzlich anzeigen", value=False)
    if st.button("💾 Save"):
        save_state(df, cash, bench, fx)
        st.success("Saved")

st.subheader("Portfolio Inputs (editierbar)")
df = st.data_editor(
    df,
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "ticker": st.column_config.TextColumn("ticker"),
        "shares": st.column_config.NumberColumn("shares", step=1.0),
        "entry": st.column_config.NumberColumn("entry", step=0.01),
        "entry_ccy": st.column_config.SelectboxColumn("entry_ccy", options=["EUR","USD"]),
    }
).dropna()

df["ticker"] = df["ticker"].astype(str).str.strip()
df = df[df["ticker"]!=""].copy()

if df.duplicated("ticker").any():
    st.error("Duplicate tickers – bitte bereinigen.")
    st.stop()

colA, colB = st.columns([1,1])
with colA:
    run = st.button("▶️ Compute")
with colB:
    st.caption("Trim basiert auf *echtem Recompute* (nach Trim neue Gewichte → neues Portfolio-PnL → neues VaR/ES).")

if not run:
    st.stop()

px, rets, df2, total, port_rets, port_pnl, corr = compute_all(df, cash, bench, fx, lookback)

st.subheader("Key")
k1,k2,k3,k4 = st.columns(4)
k1.metric("Total Value (EUR)", f"{total:,.0f}")
k2.metric("Unrealized P&L (EUR)", f"{df2['unreal_pnl_eur'].sum():,.0f}")
equity = (1+port_rets).cumprod()
dd = equity/equity.cummax() - 1
k3.metric("Max Drawdown", f"{dd.min()*100:.2f}%")
k4.metric("Ann Vol", f"{(port_rets.std()*np.sqrt(252))*100:.2f}%")

st.subheader("Portfolio Risk (EUR PnL)")
rt = risk_table(port_pnl, levels, use_param=use_param)
st.dataframe(rt.style.format({"VaR_hist":"{:,.0f}","ES_hist":"{:,.0f}","VaR_param":"{:,.0f}","ES_param":"{:,.0f}"}), use_container_width=True)

st.subheader("Positions (EUR)")
st.dataframe(df2[["ticker","shares","price_eur","value_eur","weight","unreal_pnl_eur","unreal_pnl_pct"]]
             .style.format({"price_eur":"{:,.4f}","value_eur":"{:,.0f}","weight":"{:.2%}","unreal_pnl_eur":"{:,.0f}","unreal_pnl_pct":"{:.2%}"}),
             use_container_width=True)

st.subheader("Correlation")
fig = plt.figure()
plt.imshow(corr.values, aspect="auto")
plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
plt.yticks(range(len(corr.index)), corr.index)
plt.colorbar()
plt.tight_layout()
st.pyplot(fig)

st.subheader("Equity / Drawdown")
c1,c2 = st.columns(2)
with c1:
    fig1 = plt.figure()
    plt.plot(equity.index, equity.values)
    plt.title("Equity (norm)")
    plt.tight_layout()
    st.pyplot(fig1)
with c2:
    fig2 = plt.figure()
    plt.plot(dd.index, dd.values)
    plt.title("Drawdown")
    plt.tight_layout()
    st.pyplot(fig2)

# ---------------- Trim Screener (true recompute) ----------------
st.subheader("✂️ Trim Screener (echtes Recompute)")
trim_notional = st.number_input("Trim pro Position (EUR)", 1000, 50000, 10000, 1000)
target_lvl = float(max(levels)) if len(levels) else 0.95

base_es = float(rt.loc[rt["level"]==target_lvl, "ES_hist"].values[0]) if (rt["level"]==target_lvl).any() else np.nan
base_var = float(rt.loc[rt["level"]==target_lvl, "VaR_hist"].values[0]) if (rt["level"]==target_lvl).any() else np.nan

rows = []
for t in df2["ticker"].tolist():
    if float(df2.loc[df2["ticker"]==t, "value_eur"].values[0]) <= trim_notional:
        continue
    df_new2, cash_new, total_new, port_pnl_new, rt_new = what_if_trim(
        df2, cash, bench, fx, lookback, levels, use_param,
        trim_map={t: float(trim_notional)}
    )
    es_new = float(rt_new.loc[rt_new["level"]==target_lvl, "ES_hist"].values[0])
    var_new = float(rt_new.loc[rt_new["level"]==target_lvl, "VaR_hist"].values[0])
    rows.append({
        "ticker": t,
        "trim_eur": trim_notional,
        "ES_before": base_es,
        "ES_after": es_new,
        "dES": base_es - es_new,
        "dES_per_1k": (base_es - es_new)/(trim_notional/1000),
        "VaR_before": base_var,
        "VaR_after": var_new,
        "dVaR": base_var - var_new
    })

trim_df = pd.DataFrame(rows).sort_values("dES_per_1k", ascending=False) if rows else pd.DataFrame()
st.dataframe(trim_df.style.format({
    "trim_eur":"{:,.0f}","ES_before":"{:,.0f}","ES_after":"{:,.0f}","dES":"{:,.0f}","dES_per_1k":"{:,.2f}",
    "VaR_before":"{:,.0f}","VaR_after":"{:,.0f}","dVaR":"{:,.0f}"
}), use_container_width=True)

st.subheader("🤖 Auto-Trim (greedy, echtes Recompute)")
target_des = st.number_input("Ziel ΔES (EUR)", 1000, int(total*0.2), 20000, 1000)

if len(trim_df):
    remaining = float(target_des)
    plan = []
    used = {}

    for _, r in trim_df.iterrows():
        if remaining <= 0:
            break
        t = r["ticker"]
        used[t] = used.get(t, 0.0) + float(trim_notional)
        plan.append({"ticker": t, "trim_eur": float(trim_notional), "dES_est": float(r["dES"])})
        remaining -= float(r["dES"])

    plan_df = pd.DataFrame(plan)
    st.dataframe(plan_df, use_container_width=True)

    # What-if for full plan
    df_after, cash_after, total_after, port_pnl_after, rt_after = what_if_trim(
        df2, cash, bench, fx, lookback, levels, use_param,
        trim_map=used
    )
    es_after = float(rt_after.loc[rt_after["level"]==target_lvl, "ES_hist"].values[0])
    c1,c2,c3 = st.columns(3)
    c1.metric("ES vorher", f"{base_es:,.0f}")
    c2.metric("ES nach Plan", f"{es_after:,.0f}")
    c3.metric("ΔES", f"{(base_es-es_after):,.0f}")
else:
    st.info("Kein Trim möglich (Positionen zu klein oder Daten fehlen).")
