# app.py
# MasturBoard — Streamlit + Neon(Postgres) dashboard
#
# ✅ Streamlit Cloud setup:
# 1) Deploy this repo
# 2) In Streamlit Cloud → App → Settings → Secrets, add:
#    DATABASE_URL="postgresql://USER:PASSWORD@HOST:PORT/DB?sslmode=require"
#    TABLE_NAME="vasya_database_log"   # or your table name
#    SCHEMA_NAME="public"             # optional
#
# ✅ requirements.txt (put in repo root):
# streamlit
# pandas
# numpy
# plotly
# sqlalchemy
# psycopg2-binary

import os
from dataclasses import dataclass
from datetime import datetime, date, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sqlalchemy import create_engine, text


# ----------------------------
# Page config + basic styling
# ----------------------------
st.set_page_config(
    page_title="MasturBoard",
    page_icon="📈",
    layout="wide",
)

st.markdown(
    """
<style>
/* Make it look like a product */
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
h1, h2, h3 { letter-spacing: -0.02em; }
div[data-testid="metric-container"]{
  background: rgba(255,255,255,0.04);
  border: 1px solid rgba(255,255,255,0.08);
  padding: 12px 14px;
  border-radius: 16px;
}
small { opacity: 0.8; }
hr { opacity: 0.25; }
</style>
""",
    unsafe_allow_html=True,
)


# ----------------------------
# Config
# ----------------------------
@dataclass(frozen=True)
class AppConfig:
    db_url: str
    schema: str
    table: str


def get_config() -> AppConfig:
    db_url = os.getenv("DATABASE_URL", "").strip()
    if not db_url:
        st.error(
            "Не бачу DATABASE_URL. Додай його в Streamlit Secrets.\n\n"
            "Приклад:\n"
            'DATABASE_URL="postgresql://USER:PASSWORD@HOST:PORT/DB?sslmode=require"'
        )
        st.stop()

    schema = os.getenv("SCHEMA_NAME", "public").strip() or "public"
    table = os.getenv("TABLE_NAME", "vasya_database_log").strip() or "vasya_database_log"
    return AppConfig(db_url=db_url, schema=schema, table=table)


CFG = get_config()


@st.cache_resource
def get_engine():
    # Neon typically needs sslmode=require; include it in DATABASE_URL
    return create_engine(CFG.db_url, pool_pre_ping=True)


ENGINE = get_engine()


# ----------------------------
# Data access (cached)
# ----------------------------
@st.cache_data(ttl=60 * 10)
def fetch_usernames() -> list[str]:
    q = text(f'SELECT DISTINCT username FROM "{CFG.schema}"."{CFG.table}" WHERE username IS NOT NULL ORDER BY username')
    with ENGINE.connect() as c:
        rows = c.execute(q).fetchall()
    return [r[0] for r in rows]


@st.cache_data(ttl=60 * 5)
def fetch_logs(user: str | None, start: date, end: date, event_types: list[str] | None) -> pd.DataFrame:
    # We filter in SQL for speed and stability
    # Expecting columns like:
    # message_id, user_id, username, date, time, event_type, original_message
    params = {
        "start": start,
        "end": end + timedelta(days=1),  # inclusive end in UI
    }

    where = ['("date" >= :start AND "date" < :end)']

    if user and user != "ALL":
        where.append('"username" = :user')
        params["user"] = user

    if event_types:
        where.append('"event_type" = ANY(:event_types)')
        params["event_types"] = event_types

    where_sql = " AND ".join(where)

    q = text(
        f"""
        SELECT
            "message_id",
            "user_id",
            "username",
            "date"::date AS d,
            "time"::time AS t,
            "event_type",
            "original_message",
            ("date"::timestamp + "time"::time) AS ts
        FROM "{CFG.schema}"."{CFG.table}"
        WHERE {where_sql}
        ORDER BY ts
        """
    )

    with ENGINE.connect() as c:
        df = pd.read_sql(q, c, params=params)

    if df.empty:
        return df

    # Normalize
    df["ts"] = pd.to_datetime(df["ts"], utc=False, errors="coerce")
    df["d"] = pd.to_datetime(df["d"], errors="coerce").dt.date
    df["hour"] = df["ts"].dt.hour.astype("Int64")
    df["weekday"] = df["ts"].dt.day_name()  # English names, we'll remap below
    df["month"] = df["ts"].dt.to_period("M").astype(str)
    return df


@st.cache_data(ttl=60 * 10)
def fetch_event_types() -> list[str]:
    q = text(f'SELECT DISTINCT event_type FROM "{CFG.schema}"."{CFG.table}" WHERE event_type IS NOT NULL ORDER BY event_type')
    with ENGINE.connect() as c:
        rows = c.execute(q).fetchall()
    return [r[0] for r in rows]


# ----------------------------
# Helper metrics
# ----------------------------
WEEKDAY_UA = {
    "Monday": "Пн",
    "Tuesday": "Вт",
    "Wednesday": "Ср",
    "Thursday": "Чт",
    "Friday": "Пт",
    "Saturday": "Сб",
    "Sunday": "Нд",
}
WEEKDAY_ORDER_UA = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Нд"]


def compute_streaks(active_days: pd.Series) -> tuple[int, int]:
    """
    active_days: unique dates (python date) sorted asc
    Returns: (current_streak, longest_streak)
    """
    if active_days.empty:
        return 0, 0

    days = pd.to_datetime(active_days).dt.date.sort_values().unique()
    # Build streaks by scanning consecutive dates
    longest = 1
    cur = 1

    for i in range(1, len(days)):
        if days[i] == days[i - 1] + timedelta(days=1):
            cur += 1
            longest = max(longest, cur)
        else:
            cur = 1

    # current streak depends on "today-ish": last day in selection
    # We define current streak as consecutive days ending at the latest active day in selection.
    # That’s useful for the dashboard “right now”.
    current = 1
    for i in range(len(days) - 1, 0, -1):
        if days[i] == days[i - 1] + timedelta(days=1):
            current += 1
        else:
            break

    return int(current), int(longest)


def safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def human_timedelta(dt: timedelta) -> str:
    sec = int(dt.total_seconds())
    if sec < 60:
        return f"{sec}s"
    m = sec // 60
    if m < 60:
        return f"{m}m"
    h = m // 60
    if h < 48:
        return f"{h}h"
    d = h // 24
    return f"{d}d"


# ----------------------------
# Sidebar controls
# ----------------------------
st.title("MasturBoard")

with st.sidebar:
    st.header("Фільтри")

    usernames = ["ALL"] + fetch_usernames()
    user = st.selectbox("username", usernames, index=0)

    # Default range: last 90 days (or since earliest if smaller)
    today = date.today()
    default_start = today - timedelta(days=90)
    start, end = st.date_input(
        "Діапазон дат",
        value=(default_start, today),
        min_value=date(2000, 1, 1),
        max_value=today,
    )
    if isinstance(start, tuple) or isinstance(start, list):
        # Streamlit sometimes returns a tuple; normalize
        start, end = start[0], start[1]

    event_types_all = fetch_event_types()
    event_types = st.multiselect("event_type (опціонально)", event_types_all, default=[])

    st.divider()
    st.caption("Порада: якщо щось лагає — звузь діапазон дат.")


# ----------------------------
# Load data
# ----------------------------
df = fetch_logs(user=user, start=start, end=end, event_types=event_types or None)

if df.empty:
    st.warning("Немає даних за вибраний період/фільтри.")
    st.stop()

# ----------------------------
# Core aggregations
# ----------------------------
total = len(df)
active_days = pd.Series(df["d"].unique())
active_days_count = active_days.nunique()

days_in_range = (end - start).days + 1
avg_per_day = safe_div(total, active_days_count)

last_ts = df["ts"].max()
first_ts = df["ts"].min()
since_last = datetime.now() - last_ts.to_pydatetime()

# Most active hour
by_hour = df.groupby("hour", dropna=True).size().reset_index(name="count").sort_values("hour")
peak_hour_row = by_hour.sort_values("count", ascending=False).head(1)
peak_hour = int(peak_hour_row["hour"].iloc[0]) if not peak_hour_row.empty else None
peak_hour_count = int(peak_hour_row["count"].iloc[0]) if not peak_hour_row.empty else 0

# Most active weekday
df["weekday_ua"] = df["weekday"].map(WEEKDAY_UA).fillna(df["weekday"])
by_wd = df.groupby("weekday_ua").size().reindex(WEEKDAY_ORDER_UA).fillna(0).reset_index(name="count")
peak_wd_row = by_wd.sort_values("count", ascending=False).head(1)
peak_wd = str(peak_wd_row["weekday_ua"].iloc[0])
peak_wd_count = int(peak_wd_row["count"].iloc[0])

# Daily series
daily = (
    df.groupby("d")
    .size()
    .rename("count")
    .reset_index()
    .sort_values("d")
)
daily["d"] = pd.to_datetime(daily["d"])
daily["rolling_7"] = daily["count"].rolling(7, min_periods=1).mean()

peak_day_row = daily.sort_values("count", ascending=False).head(1)
peak_day = peak_day_row["d"].dt.date.iloc[0]
peak_day_count = int(peak_day_row["count"].iloc[0])

# Streaks
current_streak, longest_streak = compute_streaks(pd.Series(daily["d"]))

# Time between events
df_sorted = df.sort_values("ts")
dt = df_sorted["ts"].diff().dropna()
dt_minutes = dt.dt.total_seconds() / 60.0
interval_median = float(np.nanmedian(dt_minutes)) if len(dt_minutes) else np.nan
interval_mean = float(np.nanmean(dt_minutes)) if len(dt_minutes) else np.nan
interval_p95 = float(np.nanpercentile(dt_minutes, 95)) if len(dt_minutes) else np.nan

# Monthly
monthly = (
    df.groupby("month")
    .size()
    .rename("count")
    .reset_index()
)
# Keep chronological order
monthly["month_dt"] = pd.to_datetime(monthly["month"] + "-01")
monthly = monthly.sort_values("month_dt")

# Weekday x hour heatmap
heat = (
    df.dropna(subset=["hour"])
    .groupby(["weekday_ua", "hour"])
    .size()
    .rename("count")
    .reset_index()
)
heat["weekday_ua"] = pd.Categorical(heat["weekday_ua"], categories=WEEKDAY_ORDER_UA, ordered=True)
heat = heat.sort_values(["weekday_ua", "hour"])
heat_pivot = heat.pivot_table(index="weekday_ua", columns="hour", values="count", fill_value=0).reindex(WEEKDAY_ORDER_UA)

# Cumulative
cum = daily.copy()
cum["cumulative"] = cum["count"].cumsum()

# Activity ratio
activity_ratio = safe_div(active_days_count, days_in_range)

# ----------------------------
# Top KPIs
# ----------------------------
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns([1.1, 1.1, 1.2, 1.2, 1.6])

kpi1.metric("Total", f"{total:,}".replace(",", " "))
kpi2.metric("Average per active day", f"{avg_per_day:.2f}")
kpi3.metric("Most active hour", f"{peak_hour:02d}:00", f"{peak_hour_count} events")
kpi4.metric("Most active weekday", f"{peak_wd}", f"{peak_wd_count} events")
kpi5.metric("Last activity", last_ts.strftime("%b %d, %Y, %H:%M:%S"), f"{human_timedelta(since_last)} ago")

kpi6, kpi7, kpi8, kpi9 = st.columns(4)
kpi6.metric("Active days", f"{active_days_count}/{days_in_range}", f"{activity_ratio*100:.1f}% coverage")
kpi7.metric("Peak day", str(peak_day), f"{peak_day_count} events")
kpi8.metric("Streak now", f"{current_streak} days")
kpi9.metric("Longest streak", f"{longest_streak} days")

st.divider()

# ----------------------------
# Charts row 1
# ----------------------------
c1, c2 = st.columns([1.35, 1.0])

with c1:
    fig = px.line(
        daily,
        x="d",
        y="count",
        title="Count by Days",
        markers=False,
    )
    fig.add_scatter(x=daily["d"], y=daily["rolling_7"], mode="lines", name="7d avg")
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)

with c2:
    fig = px.area(
        by_hour,
        x="hour",
        y="count",
        title="Count by Hours",
    )
    fig.update_xaxes(dtick=1)
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# Charts row 2
# ----------------------------
c3, c4 = st.columns([1.0, 1.35])

with c3:
    fig = px.bar(
        by_wd,
        x="weekday_ua",
        y="count",
        title="Count by Weekdays",
    )
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)

with c4:
    fig = px.imshow(
        heat_pivot,
        title="Heatmap: Weekday × Hour",
        aspect="auto",
        labels=dict(x="Hour", y="Weekday", color="Count"),
    )
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# Charts row 3
# ----------------------------
c5, c6, c7 = st.columns([1.0, 1.0, 1.0])

with c5:
    fig = px.bar(
        monthly,
        x="month",
        y="count",
        title="Count by Months",
    )
    fig.update_layout(height=330, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)

with c6:
    fig = px.line(
        cum,
        x="d",
        y="cumulative",
        title="Cumulative count",
    )
    fig.update_layout(height=330, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)

with c7:
    # Intervals histogram (minutes)
    if len(dt_minutes) >= 5:
        tmp = pd.DataFrame({"minutes": dt_minutes})
        # cap extreme outliers for readability
        cap = float(np.nanpercentile(tmp["minutes"], 99))
        tmp["minutes_capped"] = np.minimum(tmp["minutes"], cap)
        fig = px.histogram(
            tmp,
            x="minutes_capped",
            nbins=40,
            title="Time between events (minutes)",
        )
        fig.update_layout(height=330, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Мало даних, щоб показати розподіл інтервалів між подіями.")

# ----------------------------
# Extra stats (compact)
# ----------------------------
st.divider()
s1, s2, s3, s4 = st.columns(4)

with s1:
    st.subheader("Період")
    st.write(f"Початок: **{first_ts.strftime('%Y-%m-%d %H:%M:%S')}**")
    st.write(f"Кінець: **{last_ts.strftime('%Y-%m-%d %H:%M:%S')}**")

with s2:
    st.subheader("Інтервали")
    if len(dt_minutes):
        st.write(f"Median: **{interval_median:.1f} min**")
        st.write(f"Mean: **{interval_mean:.1f} min**")
        st.write(f"P95: **{interval_p95:.1f} min**")
    else:
        st.write("Немає інтервалів (замало подій).")

with s3:
    st.subheader("Концентрація")
    # "top hour share" — how much activity is in the peak hour
    top_hour_share = safe_div(peak_hour_count, total) * 100
    st.write(f"Пік-година: **{top_hour_share:.1f}%** від усіх подій")
    top_day_share = safe_div(peak_day_count, total) * 100
    st.write(f"Пік-день: **{top_day_share:.1f}%** від усіх подій")

with s4:
    st.subheader("Дані")
    st.write(f"Rows: **{total:,}**".replace(",", " "))
    st.write(f"Active days: **{active_days_count}**")
    st.write(f"Filter user: **{user}**")

# ----------------------------
# Raw table (optional)
# ----------------------------
with st.expander("Показати сирі події"):
    show_cols = ["ts", "username", "event_type", "original_message", "message_id", "user_id"]
    existing = [c for c in show_cols if c in df.columns]
    st.dataframe(df[existing].sort_values("ts", ascending=False), use_container_width=True)
