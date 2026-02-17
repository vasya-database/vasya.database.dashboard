import os
from dataclasses import dataclass
from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sqlalchemy import create_engine, text

st.set_page_config(page_title="MasturBoard", page_icon="📈", layout="wide")

st.markdown(
    """
<style>
.block-container { padding-top: 1.1rem; padding-bottom: 2rem; }
h1, h2, h3 { letter-spacing: -0.02em; }
div[data-testid="metric-container"]{
  background: rgba(255,255,255,0.04);
  border: 1px solid rgba(255,255,255,0.08);
  padding: 12px 14px;
  border-radius: 16px;
}
hr { opacity: 0.25; }
</style>
""",
    unsafe_allow_html=True,
)

KYIV_TZ = ZoneInfo("Europe/Kyiv")

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


@dataclass(frozen=True)
class AppConfig:
    db_url: str
    schema: str
    table: str


def get_config() -> AppConfig:
    db_url = os.getenv("DATABASE_URL", "").strip()
    if not db_url:
        st.error(
            "Не бачу DATABASE_URL у Secrets.\n\n"
            "Streamlit Cloud → Manage app → Settings → Secrets\n"
            'DATABASE_URL="postgresql://USER:PASSWORD@HOST:PORT/DB?sslmode=require"'
        )
        st.stop()

    schema = os.getenv("SCHEMA_NAME", "public").strip() or "public"
    table = os.getenv("TABLE_NAME", "events_kyiv").strip() or "events_kyiv"
    return AppConfig(db_url, schema, table)


CFG = get_config()


@st.cache_resource
def get_engine():
    return create_engine(CFG.db_url, pool_pre_ping=True)


ENGINE = get_engine()


def safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def human_timedelta(td: timedelta) -> str:
    sec = int(td.total_seconds())
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


def compute_streaks(days: pd.Series) -> tuple[int, int]:
    if days.empty:
        return 0, 0
    arr = pd.to_datetime(days).dt.date.sort_values().unique()
    longest = 1
    cur = 1
    for i in range(1, len(arr)):
        if arr[i] == arr[i - 1] + timedelta(days=1):
            cur += 1
            longest = max(longest, cur)
        else:
            cur = 1
    current = 1
    for i in range(len(arr) - 1, 0, -1):
        if arr[i] == arr[i - 1] + timedelta(days=1):
            current += 1
        else:
            break
    return int(current), int(longest)


@st.cache_data(ttl=60 * 10)
def fetch_usernames() -> list[str]:
    q = text(
        f'SELECT DISTINCT username FROM "{CFG.schema}"."{CFG.table}" '
        "WHERE username IS NOT NULL ORDER BY username"
    )
    with ENGINE.connect() as c:
        rows = c.execute(q).fetchall()
    return [r[0] for r in rows]


HARD_MIN_DATE = date(2023, 5, 8)


@st.cache_data(ttl=60 * 30)
def get_data_bounds_kyiv() -> tuple[date, date]:
    """
    Returns (min_date, max_date) in KYIV time, based on event_ts (timestamptz).
    Applies HARD_MIN_DATE as a floor so UI never goes earlier than 2023-05-08.
    """
    q = text(
        f"""
        SELECT
          MIN("event_ts") AS min_ts,
          MAX("event_ts") AS max_ts
        FROM "{CFG.schema}"."{CFG.table}"
        """
    )
    with ENGINE.connect() as c:
        row = c.execute(q).first()

    min_ts, max_ts = row[0], row[1]
    if not min_ts or not max_ts:
        return HARD_MIN_DATE, date.today()

    # Parse as tz-aware, convert to Kyiv, take date()
    min_kyiv = pd.to_datetime(min_ts, utc=True).tz_convert(KYIV_TZ).date()
    max_kyiv = pd.to_datetime(max_ts, utc=True).tz_convert(KYIV_TZ).date()

    if min_kyiv < HARD_MIN_DATE:
        min_kyiv = HARD_MIN_DATE

    return min_kyiv, max_kyiv


@st.cache_data(ttl=60 * 5)
def fetch_logs(user: str | None, start_date: date, end_date: date) -> pd.DataFrame:
    """
    Filters by KYIV date range, but does it correctly in SQL using event_ts (timestamptz).

    We build Kyiv midnight bounds and convert them to UTC for filtering.
    """
    # Kyiv bounds (inclusive start, exclusive end)
    start_kyiv = datetime.combine(start_date, datetime.min.time()).replace(tzinfo=KYIV_TZ)
    end_kyiv_excl = datetime.combine(end_date + timedelta(days=1), datetime.min.time()).replace(tzinfo=KYIV_TZ)

    # Convert to UTC for DB filter
    start_utc = start_kyiv.astimezone(ZoneInfo("UTC"))
    end_utc = end_kyiv_excl.astimezone(ZoneInfo("UTC"))

    params = {"start": start_utc, "end": end_utc}

    where = ['("event_ts" >= :start AND "event_ts" < :end)']
    if user and user != "ALL":
        where.append('"username" = :user')
        params["user"] = user

    q = text(
        f"""
        SELECT
            "message_id",
            "user_id",
            "username",
            "event_ts",
            "event_type",
            "original_message"
        FROM "{CFG.schema}"."{CFG.table}"
        WHERE {" AND ".join(where)}
        ORDER BY "event_ts"
        """
    )

    with ENGINE.connect() as c:
        df = pd.read_sql(q, c, params=params)

    if df.empty:
        return df

    # event_ts -> UTC aware, then convert to Kyiv for everything shown/aggregated
    df["ts_utc"] = pd.to_datetime(df["event_ts"], utc=True)
    df["ts"] = df["ts_utc"].dt.tz_convert(KYIV_TZ)

    df["d"] = df["ts"].dt.date
    df["hour"] = df["ts"].dt.hour.astype(int)
    df["weekday"] = df["ts"].dt.day_name()
    df["weekday_ua"] = df["weekday"].map(WEEKDAY_UA).fillna(df["weekday"])
    df["month"] = df["ts"].dt.to_period("M").astype(str)

    return df


# ---------------- UI ----------------
st.title("MasturBoard")

MIN_DATE, MAX_DATE_DB = get_data_bounds_kyiv()
TODAY_KYIV = datetime.now(KYIV_TZ).date()
MAX_DATE_UI = min(MAX_DATE_DB, TODAY_KYIV)

with st.sidebar:
    st.header("Фільтри")
    usernames = ["ALL"] + fetch_usernames()
    user = st.selectbox("username", usernames, index=0)

    start_date, end_date = st.date_input(
        "Діапазон дат",
        value=(MIN_DATE, MAX_DATE_UI),
        min_value=MIN_DATE,
        max_value=MAX_DATE_UI,
    )
    if isinstance(start_date, (tuple, list)):
        start_date, end_date = start_date[0], start_date[1]

df = fetch_logs(user=user, start_date=start_date, end_date=end_date)

if df.empty:
    st.warning("Немає даних за вибраний період/фільтри.")
    st.stop()

# ---------------- Metrics ----------------
total = len(df)
days_in_range = (end_date - start_date).days + 1

active_days = pd.Series(df["d"].unique())
active_days_count = active_days.nunique()
avg_per_active_day = safe_div(total, active_days_count)
coverage = safe_div(active_days_count, days_in_range) * 100

last_ts = df["ts"].max()          # Kyiv tz-aware
first_ts = df["ts"].min()         # Kyiv tz-aware
now_kyiv = datetime.now(KYIV_TZ)
since_last = now_kyiv - last_ts.to_pydatetime()  # ✅ no negatives, Kyiv-based

by_hour = df.groupby("hour").size().reset_index(name="count").sort_values("hour")
peak_hour_row = by_hour.sort_values("count", ascending=False).head(1)
peak_hour = int(peak_hour_row["hour"].iloc[0])
peak_hour_count = int(peak_hour_row["count"].iloc[0])

by_wd = (
    df.groupby("weekday_ua")
    .size()
    .reindex(WEEKDAY_ORDER_UA)
    .fillna(0)
    .reset_index(name="count")
)
peak_wd_row = by_wd.sort_values("count", ascending=False).head(1)
peak_wd = str(peak_wd_row["weekday_ua"].iloc[0])
peak_wd_count = int(peak_wd_row["count"].iloc[0])

daily = df.groupby("d").size().rename("count").reset_index().sort_values("d")
daily["d"] = pd.to_datetime(daily["d"])
daily["rolling_7"] = daily["count"].rolling(7, min_periods=1).mean()

peak_day_row = daily.sort_values("count", ascending=False).head(1)
peak_day = peak_day_row["d"].dt.date.iloc[0]
peak_day_count = int(peak_day_row["count"].iloc[0])

current_streak, longest_streak = compute_streaks(pd.Series(daily["d"]))

df_sorted = df.sort_values("ts")
diff = df_sorted["ts"].diff().dropna()
diff_min = diff.dt.total_seconds() / 60.0
interval_median = float(np.nanmedian(diff_min)) if len(diff_min) else np.nan
interval_mean = float(np.nanmean(diff_min)) if len(diff_min) else np.nan

monthly = df.groupby("month").size().rename("count").reset_index()
monthly["month_dt"] = pd.to_datetime(monthly["month"] + "-01")
monthly = monthly.sort_values("month_dt")

heat = df.groupby(["weekday_ua", "hour"]).size().rename("count").reset_index()
heat["weekday_ua"] = pd.Categorical(heat["weekday_ua"], categories=WEEKDAY_ORDER_UA, ordered=True)
heat = heat.sort_values(["weekday_ua", "hour"])
heat_pivot = heat.pivot_table(index="weekday_ua", columns="hour", values="count", fill_value=0).reindex(
    WEEKDAY_ORDER_UA
)

cum = daily.copy()
cum["cumulative"] = cum["count"].cumsum()

# ---------------- KPI cards ----------------
r1 = st.columns([1.1, 1.1, 1.2, 1.2, 1.6])
r1[0].metric("Total", f"{total:,}".replace(",", " "))
r1[1].metric("Avg / active day", f"{avg_per_active_day:.2f}")
r1[2].metric("Most active hour", f"{peak_hour:02d}:00", f"{peak_hour_count} events")
r1[3].metric("Most active weekday", peak_wd, f"{peak_wd_count} events")
r1[4].metric("Last activity", last_ts.strftime("%b %d, %Y, %H:%M:%S"), f"{human_timedelta(since_last)} ago")

r2 = st.columns(4)
r2[0].metric("Active days", f"{active_days_count}/{days_in_range}", f"{coverage:.1f}% coverage")
r2[1].metric("Peak day", str(peak_day), f"{peak_day_count} events")
r2[2].metric("Streak now", f"{current_streak} days")
r2[3].metric("Longest streak", f"{longest_streak} days")

st.divider()

# ---------------- Charts ----------------
c1, c2 = st.columns([1.35, 1.0])
with c1:
    fig = px.line(daily, x="d", y="count", title="Count by Days")
    fig.add_scatter(x=daily["d"], y=daily["rolling_7"], mode="lines", name="7d avg")
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)

with c2:
    fig = px.area(by_hour, x="hour", y="count", title="Count by Hours")
    fig.update_xaxes(dtick=1)
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)

c3, c4 = st.columns([1.0, 1.35])
with c3:
    fig = px.bar(by_wd, x="weekday_ua", y="count", title="Count by Weekdays")
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

c5, c6, c7 = st.columns([1.0, 1.0, 1.0])
with c5:
    fig = px.bar(monthly, x="month", y="count", title="Count by Months")
    fig.update_layout(height=330, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)

with c6:
    fig = px.line(cum, x="d", y="cumulative", title="Cumulative count")
    fig.update_layout(height=330, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)

with c7:
    if len(diff_min) >= 5:
        tmp = pd.DataFrame({"minutes": diff_min})
        cap = float(np.nanpercentile(tmp["minutes"], 99))
        tmp["minutes_capped"] = np.minimum(tmp["minutes"], cap)
        fig = px.histogram(tmp, x="minutes_capped", nbins=40, title="Time between events (minutes)")
        fig.update_layout(height=330, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Мало даних, щоб показати розподіл інтервалів.")

st.divider()

# ---------------- Goals + Achievements (side-by-side) ----------------
st.divider()
st.header("🏁 Goals & 🏆 Achievements")

# ✅ ТУТ ТИ ПРОСТО РЕДАГУЄШ СПИСКИ

GOALS = [
    {
        "goal": "Тримати streak 50 днів",
        "target": "50 days",
        "status": "In progress",
        "note": "Без пропусків"
    },
    {
        "goal": "Зменшити до 1 раз/день",
        "target": "≤ 1 / day",
        "status": "In progress",
        "note": "Контроль частоти"
    },
    # Додавай свої
    # {"goal": "...", "target": "...", "status": "Done/In progress/Planned", "note": "..."},
]

ACHIEVEMENTS = [
    {"date": "2025-07-03", "time": "16:48:52", "title": "Подрочив в горах"},
    # {"date": "2026-02-18", "time": "00:17:00", "title": "Нічний рейд"},
]

goals_df = pd.DataFrame(GOALS)
ach_df = pd.DataFrame(ACHIEVEMENTS)

# (опціонально) підсортуємо досягнення по даті+часу
if not ach_df.empty:
    ach_df["_dt"] = pd.to_datetime(ach_df["date"] + " " + ach_df["time"], errors="coerce")
    ach_df = ach_df.sort_values("_dt", ascending=False).drop(columns=["_dt"])

# Дві колонки поруч
left, right = st.columns([1.05, 1.0])

with left:
    st.subheader("🏁 Goals")
    if goals_df.empty:
        st.info("Поки що немає цілей.")
    else:
        st.dataframe(goals_df, use_container_width=True, hide_index=True)

with right:
    st.subheader("🏆 Achievements")
    if ach_df.empty:
        st.info("Поки що немає досягнень.")
    else:
        st.dataframe(ach_df, use_container_width=True, hide_index=True)
