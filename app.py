"""MasturBoard — дашборд активності (Streamlit + Postgres + Plotly).

Основні відмінності від першої версії:
  * дані тягнуться один раз на всю історію (3 колонки) і фільтруються в pandas —
    зміна діапазону дат більше не б'є по БД;
  * усі часові ряди реіндексуються на повний календар (немає фальшивих ліній
    через прогалини, rolling(7) = 7 календарних днів, а не 7 активних);
  * "поточний стрік" рахується від сьогодні по всій історії, а не від кінця
    вибраного діапазону;
  * інтервали між подіями рахуються в межах одного користувача;
  * цілі обчислюються з даних, а не захардкоджені;
  * додано календарний хітмап, punchcard, добовий "годинник", YoY, ECDF,
    розбивку за event_type і порівняння користувачів.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

# --------------------------------------------------------------------------- #
# Константи
# --------------------------------------------------------------------------- #

KYIV_TZ = ZoneInfo("Europe/Kyiv")
UTC = ZoneInfo("UTC")
HARD_MIN_DATE = date(2023, 5, 8)

WEEKDAY_UA = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Нд"]
MONTH_UA = ["Січ", "Лют", "Бер", "Кві", "Тра", "Чер", "Лип", "Сер", "Вер", "Жов", "Лис", "Гру"]

IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

st.set_page_config(page_title="MasturBoard", page_icon="📈", layout="wide")

# --------------------------------------------------------------------------- #
# Тема і палітра (validated categorical palette, фіксований порядок слотів)
# --------------------------------------------------------------------------- #


def _theme_is_dark() -> bool:
    try:  # Streamlit >= 1.46
        t = st.context.theme
        if t and getattr(t, "type", None):
            return t.type == "dark"
    except Exception:
        pass
    try:
        return (st.get_option("theme.base") or "dark") == "dark"
    except Exception:
        return True


DARK = _theme_is_dark()

# слоти призначаються у фіксованому порядку і ніколи не циклються
CAT = (
    ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]
    if DARK
    else ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
)
# послідовна шкала: одна барва, від "майже поверхня" до насиченої
SEQ = (
    ["#0d366b", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb"]
    if DARK
    else ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}

SURFACE = "#1a1a19" if DARK else "#fcfcfb"
INK = "#f2f2ef" if DARK else "#0b0b0b"
INK_MUTED = "#a3a29a" if DARK else "#6b6a66"
GRID = "rgba(255,255,255,0.07)" if DARK else "rgba(0,0,0,0.07)"

pio.templates["mb"] = go.layout.Template(
    layout=dict(
        colorway=CAT,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=INK, size=13),
        title=dict(font=dict(size=15, color=INK), x=0, xanchor="left", pad=dict(b=8)),
        margin=dict(l=8, r=8, t=48, b=8),
        hoverlabel=dict(bgcolor=SURFACE, bordercolor=GRID, font=dict(color=INK, size=12)),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
            font=dict(color=INK_MUTED, size=12), bgcolor="rgba(0,0,0,0)",
        ),
        xaxis=dict(gridcolor=GRID, zeroline=False, linecolor=GRID,
                   tickfont=dict(color=INK_MUTED), title=dict(font=dict(color=INK_MUTED))),
        yaxis=dict(gridcolor=GRID, zeroline=False, linecolor=GRID,
                   tickfont=dict(color=INK_MUTED), title=dict(font=dict(color=INK_MUTED))),
        colorscale=dict(sequential=[[i / (len(SEQ) - 1), c] for i, c in enumerate(SEQ)]),
    )
)
pio.templates.default = "mb"
px.defaults.template = "mb"
px.defaults.color_discrete_sequence = CAT

PLOTLY_CFG = {"displayModeBar": False, "displaylogo": False, "scrollZoom": False}

st.markdown(
    f"""
<style>
.block-container {{ padding-top: 1.1rem; padding-bottom: 3rem; max-width: 1500px; }}
h1, h2, h3 {{ letter-spacing: -0.02em; }}
div[data-testid="stMetric"], div[data-testid="metric-container"] {{
  background: {"rgba(255,255,255,0.035)" if DARK else "rgba(0,0,0,0.025)"};
  border: 1px solid {GRID};
  padding: 12px 14px;
  border-radius: 14px;
}}
div[data-testid="stMetricValue"] {{ font-size: 1.45rem; }}
hr {{ opacity: 0.2; }}
.goal-row {{ display:flex; justify-content:space-between; font-size:0.9rem; margin-bottom:2px; }}
.goal-note {{ color:{INK_MUTED}; font-size:0.8rem; margin:-6px 0 12px 0; }}
</style>
""",
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- #
# Сумісність із різними версіями Streamlit
# --------------------------------------------------------------------------- #

_SV = tuple(int(x) for x in (re.findall(r"\d+", st.__version__) + ["0", "0"])[:2])
_NEW_WIDTH_API = _SV >= (1, 49)


def show(fig: go.Figure, height: int = 340) -> None:
    """Єдина точка рендеру графіка: висота, конфіг, ширина контейнера."""
    fig.update_layout(height=height)
    if _NEW_WIDTH_API:
        st.plotly_chart(fig, width="stretch", config=PLOTLY_CFG)
    else:
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CFG)


def show_df(data: pd.DataFrame, **kw) -> None:
    if _NEW_WIDTH_API:
        st.dataframe(data, width="stretch", **kw)
    else:
        st.dataframe(data, use_container_width=True, **kw)


def rounded_bars(fig: go.Figure) -> go.Figure:
    """4px заокруглені кінці стовпчиків (plotly >= 5.19); тихо ігнорується на старих."""
    try:
        fig.update_layout(barcornerradius=4)
    except Exception:
        pass
    return fig


# --------------------------------------------------------------------------- #
# Конфіг і доступ до БД
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class AppConfig:
    db_url: str
    schema: str
    table: str

    @property
    def qualified(self) -> str:
        return f'"{self.schema}"."{self.table}"'


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
    for name, val in (("SCHEMA_NAME", schema), ("TABLE_NAME", table)):
        if not IDENT_RE.match(val):
            st.error(f"Недопустиме значення {name}={val!r}: очікую [A-Za-z_][A-Za-z0-9_]*")
            st.stop()
    return AppConfig(db_url, schema, table)


CFG = get_config()


@st.cache_resource
def get_engine():
    return create_engine(CFG.db_url, pool_pre_ping=True)


@st.cache_data(ttl=600, show_spinner=False)
def fetch_usernames() -> list[str]:
    q = text(f"SELECT DISTINCT username FROM {CFG.qualified} WHERE username IS NOT NULL ORDER BY username")
    with get_engine().connect() as c:
        return [r[0] for r in c.execute(q).fetchall()]


@st.cache_data(ttl=300, show_spinner="Читаю дані…")
def fetch_events(user: str) -> pd.DataFrame:
    """Уся історія користувача, лише потрібні колонки. Фільтрація дат — у pandas."""
    params: dict = {}
    where = ['"event_ts" IS NOT NULL']
    if user and user != "ALL":
        where.append('"username" = :user')
        params["user"] = user

    q = text(
        f"""
        SELECT "username", "event_ts", "event_type"
        FROM {CFG.qualified}
        WHERE {" AND ".join(where)}
        ORDER BY "event_ts"
        """
    )
    with get_engine().connect() as c:
        df = pd.read_sql(q, c, params=params)

    if df.empty:
        return df

    ts = pd.to_datetime(df["event_ts"], utc=True).dt.tz_convert(KYIV_TZ)
    df = df.drop(columns=["event_ts"])
    df["ts"] = ts
    df["d"] = ts.dt.date
    df["hour"] = ts.dt.hour.astype(int)
    df["hour_f"] = ts.dt.hour + ts.dt.minute / 60.0
    df["wd"] = ts.dt.weekday.astype(int)
    df["weekday_ua"] = df["wd"].map(dict(enumerate(WEEKDAY_UA)))
    df["year"] = ts.dt.year.astype(int)
    df["doy"] = ts.dt.dayofyear.astype(int)
    df = df[df["d"] >= HARD_MIN_DATE]
    return df


def guarded(fn, *args, **kwargs):
    """Будь-яка помилка БД → людське повідомлення замість traceback."""
    try:
        return fn(*args, **kwargs)
    except SQLAlchemyError as exc:
        st.error("Не вдалося прочитати дані з бази.")
        with st.expander("Технічні деталі"):
            st.code(str(exc.__cause__ or exc)[:2000])
        st.stop()


# --------------------------------------------------------------------------- #
# Утиліти
# --------------------------------------------------------------------------- #


def safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def human_td(td: timedelta) -> str:
    sec = int(td.total_seconds())
    if sec < 60:
        return f"{sec} с"
    m, h = sec // 60, sec // 3600
    if m < 60:
        return f"{m} хв"
    if h < 48:
        return f"{h} год"
    return f"{h // 24} дн"


def human_minutes(minutes: float) -> str:
    if not np.isfinite(minutes):
        return "—"
    return human_td(timedelta(minutes=float(minutes)))


def compute_streaks(active: list[date], today: date) -> tuple[int, int]:
    """(поточний стрік, найдовший). Поточний обривається, якщо останній
    активний день раніше за вчора — інакше метрика "бреше" на старих даних."""
    if not active:
        return 0, 0
    arr = sorted(set(active))
    longest = cur = 1
    for i in range(1, len(arr)):
        if arr[i] == arr[i - 1] + timedelta(days=1):
            cur += 1
            longest = max(longest, cur)
        else:
            cur = 1
    if arr[-1] < today - timedelta(days=1):
        return 0, longest
    current = 1
    for i in range(len(arr) - 1, 0, -1):
        if arr[i] == arr[i - 1] + timedelta(days=1):
            current += 1
        else:
            break
    return current, longest


def daily_series(events: pd.DataFrame, d_from: date, d_to: date) -> pd.DataFrame:
    """Щоденні лічильники, реіндексовані на ПОВНИЙ календар (нулі присутні)."""
    idx = pd.date_range(d_from, d_to, freq="D")
    counts = events.groupby("d").size() if not events.empty else pd.Series(dtype=int)
    counts.index = pd.to_datetime(counts.index) if len(counts) else counts.index
    out = counts.reindex(idx, fill_value=0).rename("count").rename_axis("d").reset_index()
    out["count"] = out["count"].astype(int)
    return out


def zero_runs(daily: pd.DataFrame) -> pd.DataFrame:
    """Періоди простою (послідовні дні з 0 подій)."""
    z = (daily["count"].values == 0).astype(int)
    if z.sum() == 0:
        return pd.DataFrame(columns=["start", "end", "days"])
    edges = np.diff(np.concatenate(([0], z, [0])))
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0] - 1
    dates = daily["d"].dt.date.values
    return pd.DataFrame(
        {"start": dates[starts], "end": dates[ends], "days": (ends - starts + 1).astype(int)}
    )


def gaps_minutes(events: pd.DataFrame) -> pd.Series:
    """Інтервали між подіями — В МЕЖАХ одного користувача."""
    if events.empty:
        return pd.Series(dtype=float)
    s = events.sort_values(["username", "ts"])
    d = s.groupby("username", dropna=False)["ts"].diff().dropna()
    return d.dt.total_seconds() / 60.0


# --------------------------------------------------------------------------- #
# Сайдбар
# --------------------------------------------------------------------------- #

TODAY = datetime.now(KYIV_TZ).date()

with st.sidebar:
    st.header("Фільтри")
    usernames = ["ALL"] + guarded(fetch_usernames)
    user = st.selectbox("username", usernames, index=0)

ALL_EVENTS = guarded(fetch_events, user)

if ALL_EVENTS.empty:
    st.title("MasturBoard")
    st.warning("Для цього користувача немає жодної події.")
    st.stop()

DATA_MIN = ALL_EVENTS["d"].min()
DATA_MAX = min(ALL_EVENTS["d"].max(), TODAY)

PRESETS = {
    "7 днів": 7,
    "30 днів": 30,
    "90 днів": 90,
    "365 днів": 365,
    "Цей рік": None,
    "Весь час": None,
    "Свій діапазон": None,
}

with st.sidebar:
    preset = st.radio("Період", list(PRESETS), index=5)

    if preset == "Весь час":
        start_date, end_date = DATA_MIN, DATA_MAX
    elif preset == "Цей рік":
        start_date, end_date = max(DATA_MIN, date(TODAY.year, 1, 1)), DATA_MAX
    elif preset == "Свій діапазон":
        picked = st.date_input(
            "Діапазон дат",
            value=(DATA_MIN, DATA_MAX),
            min_value=DATA_MIN,
            max_value=DATA_MAX,
            format="YYYY-MM-DD",
        )
        # Streamlit віддає кортеж довжини 1, поки обрана лише перша дата.
        picked = picked if isinstance(picked, (tuple, list)) else (picked,)
        if len(picked) < 2:
            st.info("Оберіть кінцеву дату діапазону.")
            st.stop()
        start_date, end_date = picked[0], picked[1]
        if start_date > end_date:
            start_date, end_date = end_date, start_date
    else:
        end_date = DATA_MAX
        start_date = max(DATA_MIN, end_date - timedelta(days=PRESETS[preset] - 1))

    st.caption(f"{start_date:%d.%m.%Y} → {end_date:%d.%m.%Y}")
    st.divider()
    smooth = st.slider("Вікно згладжування, днів", 3, 30, 7, step=1)
    compare_prev = st.toggle("Порівнювати з попереднім періодом", value=True)

# --------------------------------------------------------------------------- #
# Похідні набори
# --------------------------------------------------------------------------- #

mask = (ALL_EVENTS["d"] >= start_date) & (ALL_EVENTS["d"] <= end_date)
df = ALL_EVENTS.loc[mask].copy()

if df.empty:
    st.title("MasturBoard")
    st.warning("Немає даних за вибраний період/фільтри.")
    st.stop()

days_in_range = (end_date - start_date).days + 1
daily = daily_series(df, start_date, end_date)
daily[f"roll_{smooth}"] = daily["count"].rolling(smooth, min_periods=1).mean()

daily_all = daily_series(ALL_EVENTS, DATA_MIN, max(DATA_MAX, TODAY))
weekly = (
    daily.set_index("d")["count"].resample("W-MON", label="left", closed="left").sum().rename("count").reset_index()
)
monthly = daily.set_index("d")["count"].resample("MS").sum().rename("count").reset_index()

active_days = int((daily["count"] > 0).sum())
total = int(daily["count"].sum())
avg_active = safe_div(total, active_days)
avg_calendar = safe_div(total, days_in_range)
coverage = safe_div(active_days, days_in_range) * 100

last_ts = df["ts"].max()
since_last = datetime.now(KYIV_TZ) - last_ts.to_pydatetime()

by_hour = (
    df.groupby("hour").size().reindex(range(24), fill_value=0).rename("count").rename_axis("hour").reset_index()
)
peak_hour = int(by_hour.loc[by_hour["count"].idxmax(), "hour"])
peak_hour_count = int(by_hour["count"].max())

by_wd = (
    df.groupby("weekday_ua").size().reindex(WEEKDAY_UA, fill_value=0).rename("count").rename_axis("weekday_ua").reset_index()
)
peak_wd = str(by_wd.loc[by_wd["count"].idxmax(), "weekday_ua"])
peak_wd_count = int(by_wd["count"].max())

peak_day_i = int(daily["count"].idxmax())
peak_day, peak_day_count = daily.loc[peak_day_i, "d"].date(), int(daily.loc[peak_day_i, "count"])
peak_week_i = int(weekly["count"].idxmax())
peak_week, peak_week_count = weekly.loc[peak_week_i, "d"].date(), int(weekly.loc[peak_week_i, "count"])
peak_month_i = int(monthly["count"].idxmax())
peak_month, peak_month_count = monthly.loc[peak_month_i, "d"], int(monthly.loc[peak_month_i, "count"])

# стріки — по ВСІЙ історії відносно сьогодні, інакше число не має сенсу
all_active_dates = sorted(ALL_EVENTS["d"].unique())
current_streak, longest_streak = compute_streaks(list(all_active_dates), TODAY)

gap_min = gaps_minutes(df)
interval_median = float(np.nanmedian(gap_min)) if len(gap_min) else np.nan

# попередній період такої ж довжини
prev_end = start_date - timedelta(days=1)
prev_start = prev_end - timedelta(days=days_in_range - 1)
prev_mask = (ALL_EVENTS["d"] >= prev_start) & (ALL_EVENTS["d"] <= prev_end)
prev_total = int(prev_mask.sum())
prev_avg = safe_div(prev_total, days_in_range)


def delta_of(cur: float, prev: float, suffix: str = "") -> str | None:
    if not compare_prev or prev_total == 0:
        return None
    if prev == 0:
        return "новий період"
    return f"{(cur - prev) / prev * 100:+.1f}%{suffix}"


# --------------------------------------------------------------------------- #
# Шапка + KPI
# --------------------------------------------------------------------------- #

st.title("MasturBoard")
st.caption(
    f"{start_date:%d.%m.%Y} — {end_date:%d.%m.%Y} · {days_in_range} дн · "
    f"користувач: {user} · час: Europe/Kyiv"
)

r1 = st.columns(5)
r1[0].metric("Всього", f"{total:,}".replace(",", " "), delta_of(total, prev_total))
r1[1].metric("Сер. / календарний день", f"{avg_calendar:.2f}", delta_of(avg_calendar, prev_avg))
r1[2].metric("Сер. / активний день", f"{avg_active:.2f}")
r1[3].metric("Медіанний інтервал", human_minutes(interval_median))
r1[4].metric("Остання подія", last_ts.strftime("%d.%m.%Y, %H:%M"), f"{human_td(since_last)} тому", delta_color="off")

r2 = st.columns(6)
r2[0].metric("Активних днів", f"{active_days}/{days_in_range}", f"{coverage:.1f}% покриття", delta_color="off")
r2[1].metric("Пікова година", f"{peak_hour:02d}:00", f"{peak_hour_count} подій", delta_color="off")
r2[2].metric("Піковий день тижня", peak_wd, f"{peak_wd_count} подій", delta_color="off")
r2[3].metric("Рекорд за добу", f"{peak_day_count}", f"{peak_day:%d.%m.%Y}", delta_color="off")
r2[4].metric("Стрік зараз", f"{current_streak} дн", "по всій історії", delta_color="off")
r2[5].metric("Найдовший стрік", f"{longest_streak} дн", "по всій історії", delta_color="off")

st.divider()

tab_overview, tab_patterns, tab_dist, tab_goals = st.tabs(
    ["📈 Огляд", "🧭 Патерни", "📊 Розподіли", "🏆 Цілі"]
)

# --------------------------------------------------------------------------- #
# ТАБ 1 — Огляд
# --------------------------------------------------------------------------- #

with tab_overview:
    fig = go.Figure()
    fig.add_bar(
        x=daily["d"], y=daily["count"], name="За день",
        marker_color=CAT[0], marker_line_width=0, opacity=0.55,
        hovertemplate="%{x|%d.%m.%Y}<br>%{y} подій<extra></extra>",
    )
    fig.add_scatter(
        x=daily["d"], y=daily[f"roll_{smooth}"], name=f"Середнє за {smooth} дн",
        mode="lines", line=dict(color=CAT[1], width=2),
        hovertemplate="%{x|%d.%m.%Y}<br>%{y:.2f} / день<extra></extra>",
    )
    fig.update_layout(
        title="Активність по днях", hovermode="x unified",
        yaxis_title="подій", xaxis_title=None, bargap=0.15,
    )
    rounded_bars(fig)
    show(fig, 380)

    c1, c2 = st.columns([1.25, 1.0])

    with c1:
        cum = daily.copy()
        cum["cumulative"] = cum["count"].cumsum()
        fig = go.Figure()
        fig.add_scatter(
            x=cum["d"], y=cum["cumulative"], name="Факт", mode="lines",
            line=dict(color=CAT[0], width=2), fill="tozeroy",
            fillcolor="rgba(57,135,229,0.13)" if DARK else "rgba(42,120,214,0.10)",
            hovertemplate="%{x|%d.%m.%Y}<br>разом %{y}<extra></extra>",
        )
        # прогноз до кінця місяця темпом останніх 30 днів (лише якщо період — до сьогодні)
        note = ""
        if end_date == TODAY:
            pace = float(daily["count"].tail(min(30, len(daily))).mean())
            eom = (date(TODAY.year + (TODAY.month == 12), (TODAY.month % 12) + 1, 1) - timedelta(days=1))
            left = (eom - TODAY).days
            if left > 0 and pace > 0:
                fx = pd.date_range(TODAY, eom, freq="D")
                fy = cum["cumulative"].iloc[-1] + np.arange(len(fx)) * pace
                fig.add_scatter(
                    x=fx, y=fy, name="Прогноз (темп 30 дн)", mode="lines",
                    line=dict(color=CAT[3], width=2, dash="dash"),
                    hovertemplate="%{x|%d.%m.%Y}<br>≈ %{y:.0f}<extra></extra>",
                )
                note = f"Прогноз на {eom:%d.%m}: ≈ {fy[-1]:.0f} (темп {pace:.2f}/день)"
        fig.update_layout(title="Накопичувальний підсумок", yaxis_title="подій", showlegend=note != "")
        show(fig, 340)
        if note:
            st.caption(note)

    with c2:
        m = monthly.copy()
        m["label"] = m["d"].dt.strftime("%Y-%m")
        fig = px.bar(m, x="label", y="count", title="По місяцях")
        fig.update_traces(
            marker_color=CAT[0], marker_line_width=0,
            hovertemplate="%{x}<br>%{y} подій<extra></extra>",
        )
        fig.update_layout(showlegend=False, xaxis_title=None, yaxis_title="подій", bargap=0.2)
        rounded_bars(fig)
        show(fig, 340)

    years = sorted(df["year"].unique())
    if len(years) >= 2:
        yoy = []
        for y in years:
            sub = daily[daily["d"].dt.year == y].copy()
            sub["doy"] = sub["d"].dt.dayofyear
            sub["cum"] = sub["count"].cumsum()
            sub["year"] = str(y)
            yoy.append(sub[["doy", "cum", "year", "d"]])
        yoy = pd.concat(yoy)
        fig = px.line(
            yoy, x="doy", y="cum", color="year",
            title="Рік до року: накопичувально від 1 січня",
            labels={"doy": "день року", "cum": "подій разом", "year": "рік"},
        )
        fig.update_traces(line=dict(width=2), hovertemplate="день %{x}<br>%{y} разом<extra>%{fullData.name}</extra>")
        show(fig, 340)
        st.caption("Порівняння в межах вибраного періоду. Для чесного YoY беріть пресет «Весь час».")

    with st.expander("Дані та експорт"):
        export = df[["ts", "username", "event_type", "weekday_ua", "hour"]].copy()
        export["ts"] = export["ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
        show_df(export.tail(500), hide_index=True)
        st.download_button(
            "⬇️ Завантажити CSV за період",
            export.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"masturboard_{user}_{start_date}_{end_date}.csv",
            mime="text/csv",
        )

# --------------------------------------------------------------------------- #
# ТАБ 2 — Патерни
# --------------------------------------------------------------------------- #


def calendar_heatmap(daily_df: pd.DataFrame, year: int) -> go.Figure:
    """Календар у стилі GitHub: колонки — тижні, рядки — дні тижня."""
    jan1 = date(year, 1, 1)
    dec31 = date(year, 12, 31)
    idx = pd.date_range(jan1, dec31, freq="D")
    s = daily_df.set_index("d")["count"].reindex(idx)

    offset = jan1.weekday()
    ncols = ((dec31 - jan1).days + offset) // 7 + 1
    z = np.full((7, ncols), np.nan)
    cd = np.empty((7, ncols), dtype=object)
    cd[:] = ""

    for ts, v in s.items():
        dd = ts.date()
        col = ((dd - jan1).days + offset) // 7
        z[dd.weekday(), col] = v
        cd[dd.weekday(), col] = f"{dd:%d.%m.%Y} · {WEEKDAY_UA[dd.weekday()]}"

    tickvals, ticktext = [], []
    for mth in range(1, 13):
        first = date(year, mth, 1)
        tickvals.append(((first - jan1).days + offset) // 7)
        ticktext.append(MONTH_UA[mth - 1])

    fig = go.Figure(
        go.Heatmap(
            z=z, customdata=cd, xgap=2, ygap=2,
            colorscale=[[i / (len(SEQ) - 1), c] for i, c in enumerate(SEQ)],
            zmin=0, zmax=max(1, np.nanmax(z) if np.isfinite(np.nanmax(z)) else 1),
            hovertemplate="%{customdata}<br>%{z:.0f} подій<extra></extra>",
            colorbar=dict(title="подій", thickness=10, len=0.8, outlinewidth=0),
        )
    )
    fig.update_layout(
        title=f"Календар {year}",
        xaxis=dict(tickmode="array", tickvals=tickvals, ticktext=ticktext, showgrid=False, zeroline=False),
        yaxis=dict(
            tickmode="array", tickvals=list(range(7)), ticktext=WEEKDAY_UA,
            autorange="reversed", showgrid=False, zeroline=False,
        ),
    )
    return fig


with tab_patterns:
    for y in sorted(daily["d"].dt.year.unique()):
        show(calendar_heatmap(daily, int(y)), 230)

    st.caption("Порожні клітинки — дні поза вибраним діапазоном; найтемніші — дні без подій.")

    punch = df.groupby(["d", "hour"]).size().rename("count").reset_index()
    punch["d"] = pd.to_datetime(punch["d"])
    fig = px.scatter(
        punch, x="d", y="hour", size="count", color="count", custom_data=["count"],
        color_continuous_scale=[[i / (len(SEQ) - 1), c] for i, c in enumerate(SEQ)],
        size_max=16, title="Punchcard: коли саме (дата × година доби)",
        labels={"d": "дата", "hour": "година", "count": "подій"},
    )
    fig.update_traces(
        marker=dict(line=dict(width=1, color=SURFACE)),
        hovertemplate="%{x|%d.%m.%Y} о %{y}:00<br>%{customdata[0]} подій<extra></extra>",
    )
    fig.update_yaxes(dtick=3, range=[-0.6, 23.6])
    fig.update_xaxes(title=None)
    fig.update_layout(coloraxis_showscale=False)
    show(fig, 380)
    st.caption("Показує дрейф режиму в часі — те, що губиться, коли години й дати агрегуються окремо.")

    c1, c2 = st.columns(2)

    with c1:
        fig = go.Figure(
            go.Barpolar(
                r=by_hour["count"], theta=by_hour["hour"] * 15, width=[14] * 24,
                marker_color=by_hour["count"],
                marker_colorscale=[[i / (len(SEQ) - 1), c] for i, c in enumerate(SEQ)],
                marker_line_color=SURFACE, marker_line_width=1,
                hovertemplate="%{customdata}:00 — %{r} подій<extra></extra>",
                customdata=by_hour["hour"],
            )
        )
        fig.update_layout(
            title="Добовий годинник",
            showlegend=False,
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                hole=0.12,
                radialaxis=dict(showticklabels=False, ticks="", gridcolor=GRID, linewidth=0),
                angularaxis=dict(
                    direction="clockwise", rotation=90,
                    tickmode="array", tickvals=[h * 15 for h in range(0, 24, 2)],
                    ticktext=[f"{h:02d}" for h in range(0, 24, 2)],
                    gridcolor=GRID, tickfont=dict(color=INK_MUTED),
                ),
            ),
        )
        show(fig, 380)

    with c2:
        fig = px.bar(by_wd, x="weekday_ua", y="count", title="По днях тижня")
        fig.update_traces(
            marker_color=CAT[0], marker_line_width=0,
            hovertemplate="%{x}<br>%{y} подій<extra></extra>",
        )
        fig.update_layout(showlegend=False, xaxis_title=None, yaxis_title="подій", bargap=0.25)
        rounded_bars(fig)
        show(fig, 380)

    heat = (
        df.groupby(["wd", "hour"]).size().rename("count").reset_index()
        .pivot(index="wd", columns="hour", values="count")
        .reindex(index=range(7), columns=range(24))
        .fillna(0)
    )
    fig = go.Figure(
        go.Heatmap(
            z=heat.values, x=[f"{h:02d}" for h in range(24)], y=WEEKDAY_UA, xgap=2, ygap=2,
            colorscale=[[i / (len(SEQ) - 1), c] for i, c in enumerate(SEQ)], zmin=0,
            hovertemplate="%{y}, %{x}:00<br>%{z:.0f} подій<extra></extra>",
            colorbar=dict(title="подій", thickness=10, len=0.85, outlinewidth=0),
        )
    )
    fig.update_layout(
        title="День тижня × година",
        xaxis=dict(showgrid=False, title="година"),
        yaxis=dict(showgrid=False, autorange="reversed"),
    )
    show(fig, 330)

    fig = px.box(
        df, x="weekday_ua", y="hour_f", category_orders={"weekday_ua": WEEKDAY_UA},
        title="Розподіл годин по днях тижня", labels={"weekday_ua": "день тижня", "hour_f": "година доби"},
        points=False,
    )
    fig.update_traces(marker_color=CAT[0], line_color=CAT[0], fillcolor="rgba(57,135,229,0.20)")
    fig.update_yaxes(dtick=3, range=[0, 24])
    fig.update_xaxes(title=None)
    fig.update_layout(showlegend=False)
    show(fig, 330)
    st.caption("Чи відрізняється розклад вихідних від буднів — хітмап показує це грубо, ця форма — статистично.")

# --------------------------------------------------------------------------- #
# ТАБ 3 — Розподіли
# --------------------------------------------------------------------------- #

with tab_dist:
    c1, c2 = st.columns(2)

    with c1:
        vc = daily["count"].value_counts().sort_index()
        vc = vc.reindex(range(0, int(daily["count"].max()) + 1), fill_value=0)
        pdf = pd.DataFrame({"per_day": vc.index.astype(int), "days": vc.values})
        fig = px.bar(pdf, x="per_day", y="days", title="Скільки днів мали N подій")
        fig.update_traces(
            marker_color=CAT[0], marker_line_width=0,
            hovertemplate="%{x} подій за день → %{y} днів<extra></extra>",
        )
        fig.update_layout(showlegend=False, xaxis_title="подій за день", yaxis_title="днів", bargap=0.2)
        fig.update_xaxes(dtick=1)
        rounded_bars(fig)
        show(fig, 330)
        st.caption("Середнє ховає форму розподілу — ось вона.")

    with c2:
        if len(gap_min) >= 5:
            cap = float(np.nanpercentile(gap_min, 99))
            capped = np.minimum(gap_min.values, cap) / 60.0
            fig = px.histogram(
                pd.DataFrame({"h": capped}), x="h", nbins=40,
                title="Інтервал між подіями (годин)",
            )
            fig.update_traces(
                marker_color=CAT[0], marker_line_width=0,
                hovertemplate="%{x:.1f} год<br>%{y} випадків<extra></extra>",
            )
            fig.update_layout(showlegend=False, xaxis_title="годин", yaxis_title="випадків", bargap=0.05)
            show(fig, 330)
            st.caption(f"Обрізано по 99-му перцентилю ({cap / 60:.1f} год), щоб хвіст не з'їдав графік.")
        else:
            st.info("Замало даних для розподілу інтервалів.")

    c3, c4 = st.columns(2)

    with c3:
        if len(gap_min) >= 5:
            hours = np.sort(gap_min.values) / 60.0
            ecdf = np.arange(1, len(hours) + 1) / len(hours)
            lim = float(np.percentile(hours, 95))
            keep = hours <= lim
            fig = go.Figure()
            fig.add_scatter(
                x=hours[keep], y=ecdf[keep] * 100, mode="lines",
                line=dict(color=CAT[0], width=2), name="P(наступна ≤ x)",
                hovertemplate="протягом %{x:.1f} год → %{y:.0f}%<extra></extra>",
            )
            for q, col in ((50, CAT[2]), (80, CAT[3])):
                xq = float(np.percentile(hours, q))
                if xq <= lim:
                    fig.add_vline(x=xq, line=dict(color=col, width=1, dash="dot"),
                                  annotation_text=f"{q}% ≤ {xq:.1f} год",
                                  annotation_font_color=INK_MUTED)
            fig.update_layout(
                title="Ймовірність наступної події протягом X годин",
                xaxis_title="годин", yaxis_title="%", showlegend=False,
            )
            fig.update_yaxes(range=[0, 100])
            show(fig, 330)
        else:
            st.info("Замало даних для кривої ймовірності.")

    with c4:
        runs = zero_runs(daily)
        if not runs.empty:
            top = runs.sort_values("days", ascending=False).head(12).iloc[::-1]
            top["label"] = top.apply(lambda r: f"{r['start']:%d.%m.%y} → {r['end']:%d.%m.%y}", axis=1)
            fig = px.bar(top, x="days", y="label", orientation="h", title="Найдовші перерви")
            fig.update_traces(
                marker_color=CAT[1], marker_line_width=0,
                hovertemplate="%{y}<br>%{x} днів простою<extra></extra>",
            )
            fig.update_layout(showlegend=False, xaxis_title="днів", yaxis_title=None, bargap=0.25)
            rounded_bars(fig)
            show(fig, 330)
            st.caption("Зворотний бік стріків: де саме рвався ланцюжок.")
        else:
            st.success("У вибраному періоді немає жодного дня без подій.")

    types = df["event_type"].dropna()
    if types.nunique() >= 2:
        c5, c6 = st.columns([1.0, 1.4])
        with c5:
            tc = types.value_counts().reset_index()
            tc.columns = ["event_type", "count"]
            fig = px.pie(tc, names="event_type", values="count", hole=0.55, title="Частка типів подій")
            fig.update_traces(
                marker=dict(line=dict(color=SURFACE, width=2)),
                textposition="outside", textinfo="label+percent",
                hovertemplate="%{label}<br>%{value} (%{percent})<extra></extra>",
            )
            fig.update_layout(showlegend=False)
            show(fig, 330)
        with c6:
            typed = df.dropna(subset=["event_type"])
            tm = (
                typed.assign(m=typed["ts"].dt.tz_localize(None).dt.to_period("M").dt.to_timestamp())
                .groupby(["m", "event_type"]).size().rename("count").reset_index()
            )
            fig = px.area(tm, x="m", y="count", color="event_type", title="Мікс типів подій у часі")
            fig.update_traces(line=dict(width=0.5), hovertemplate="%{x|%Y-%m}<br>%{y} подій<extra>%{fullData.name}</extra>")
            fig.update_layout(xaxis_title=None, yaxis_title="подій")
            show(fig, 330)

    if user == "ALL" and df["username"].nunique() > 1:
        st.subheader("Порівняння користувачів")
        uc = df["username"].value_counts().head(8).reset_index()
        uc.columns = ["username", "count"]
        c7, c8 = st.columns([1.0, 1.4])
        with c7:
            fig = px.bar(uc.iloc[::-1], x="count", y="username", orientation="h", title="Всього за період")
            fig.update_traces(marker_color=CAT[0], marker_line_width=0,
                              hovertemplate="%{y}<br>%{x} подій<extra></extra>")
            fig.update_layout(showlegend=False, xaxis_title="подій", yaxis_title=None, bargap=0.3)
            rounded_bars(fig)
            show(fig, 330)
        with c8:
            top_users = uc["username"].head(3).tolist()  # 3 слоти проходять all-pairs гейт
            sub = df[df["username"].isin(top_users)]
            um = (
                sub.assign(m=sub["ts"].dt.tz_localize(None).dt.to_period("M").dt.to_timestamp())
                .groupby(["m", "username"]).size().rename("count").reset_index()
            )
            fig = px.line(um, x="m", y="count", color="username", title="Динаміка топ-3 по місяцях")
            fig.update_traces(line=dict(width=2), mode="lines+markers", marker=dict(size=8))
            fig.update_layout(xaxis_title=None, yaxis_title="подій")
            show(fig, 330)
            if len(uc) > 3:
                st.caption(f"Показано топ-3 з {len(uc)} — далі кольори перестають бути надійно розрізнюваними.")

# --------------------------------------------------------------------------- #
# ТАБ 4 — Цілі та досягнення
# --------------------------------------------------------------------------- #

DIFFICULTY = {"easy": ("🟢", "легко"), "medium": ("🟡", "середньо"), "hard": ("🟠", "важко"), "extreme": ("🔴", "екстрим")}

# метрики по ВСІЙ історії — цілі не мають залежати від вибраного діапазону
weekly_all = daily_all.set_index("d")["count"].resample("W-MON", label="left", closed="left").sum()
GOALS = [
    dict(diff="medium", goal="Стрік 50 днів", cur=current_streak, target=50, unit="дн"),
    dict(diff="hard", goal="20 разів за тиждень", cur=int(weekly_all.max()), target=20, unit=""),
    dict(diff="extreme", goal="10 разів за один день", cur=int(daily_all["count"].max()), target=10, unit=""),
    dict(diff="extreme", goal="Рік без пропусків", cur=longest_streak, target=365, unit="дн"),
    dict(diff="easy", goal="1000 подій усього", cur=int(daily_all["count"].sum()), target=1000, unit=""),
]

with tab_goals:
    left, right = st.columns([1.2, 1.0])

    with left:
        st.subheader("Цілі")
        st.caption("Рахуються автоматично по всій історії, не по вибраному діапазону.")
        for g in GOALS:
            icon, label = DIFFICULTY[g["diff"]]
            done = g["cur"] >= g["target"]
            pct = min(1.0, safe_div(g["cur"], g["target"]))
            state = "✅ виконано" if done else f"{pct * 100:.0f}%"
            st.markdown(
                f"<div class='goal-row'><span>{icon} <b>{g['goal']}</b> "
                f"<span style='color:{INK_MUTED}'>· {label}</span></span>"
                f"<span>{g['cur']}{(' ' + g['unit']) if g['unit'] else ''} / "
                f"{g['target']} · {state}</span></div>",
                unsafe_allow_html=True,
            )
            st.progress(pct)

        # окрема ціль — не числова, а «попадання»
        hit = peak_hour == 11
        st.markdown(
            f"<div class='goal-row'><span>🟠 <b>Зламати пікову годину на 11:00</b> "
            f"<span style='color:{INK_MUTED}'>· важко</span></span>"
            f"<span>{'✅ 11:00' if hit else f'зараз {peak_hour:02d}:00'}</span></div>",
            unsafe_allow_html=True,
        )
        st.progress(1.0 if hit else safe_div(int(by_hour.loc[11, 'count']), max(1, peak_hour_count)))
        st.markdown(
            f"<div class='goal-note'>О 11:00 — {int(by_hour.loc[11, 'count'])} подій, "
            f"у лідера ({peak_hour:02d}:00) — {peak_hour_count}.</div>",
            unsafe_allow_html=True,
        )

        gp = pd.DataFrame(GOALS)
        gp["pct"] = (gp["cur"] / gp["target"] * 100).clip(upper=100)
        gp = gp.sort_values("pct")
        fig = px.bar(gp, x="pct", y="goal", orientation="h", title="Прогрес по цілях")
        fig.update_traces(
            marker_color=[STATUS["good"] if p >= 100 else CAT[0] for p in gp["pct"]],
            marker_line_width=0,
            hovertemplate="%{y}<br>%{x:.0f}%<extra></extra>",
        )
        fig.update_layout(showlegend=False, xaxis_title="%", yaxis_title=None, bargap=0.3)
        fig.update_xaxes(range=[0, 100])
        rounded_bars(fig)
        show(fig, 300)

    with right:
        st.subheader("Досягнення")
        ach_path = Path(__file__).with_name("achievements.json")
        if ach_path.exists():
            try:
                achievements = json.loads(ach_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                st.warning(f"achievements.json пошкоджений: {exc}")
                achievements = []
        else:
            achievements = [{"date": "2025-07-03", "time": "16:48:52", "title": "Подрочив в горах"}]

        ach_df = pd.DataFrame(achievements)
        if ach_df.empty:
            st.info("Поки що немає досягнень.")
        else:
            times = (
                ach_df["time"].fillna("00:00:00").astype(str)
                if "time" in ach_df.columns
                else pd.Series("00:00:00", index=ach_df.index)
            )
            ach_df["_dt"] = pd.to_datetime(
                ach_df["date"].astype(str) + " " + times, errors="coerce"
            )
            ach_df = ach_df.sort_values("_dt", ascending=False).drop(columns=["_dt"])
            show_df(ach_df, hide_index=True)
        st.caption("Редагується у файлі `achievements.json` поряд з app.py — без правок коду.")

        st.subheader("Рекорди")
        show_df(
            pd.DataFrame(
                [
                    ("Найкращий день", f"{peak_day:%d.%m.%Y}", peak_day_count),
                    ("Найкращий тиждень", f"з {peak_week:%d.%m.%Y}", peak_week_count),
                    ("Найкращий місяць", f"{peak_month:%Y-%m}", peak_month_count),
                    ("Найдовший стрік", "по всій історії", longest_streak),
                ],
                columns=["Рекорд", "Коли", "Значення"],
            ),
            hide_index=True,
        )
