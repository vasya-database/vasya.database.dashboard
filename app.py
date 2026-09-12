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

import inspect
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

import statmath as sm
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

# верхній відступ: заголовок ≈34px; якщо є легенда, їй потрібен власний рядок
T_MARGIN_PLAIN = 48
T_MARGIN_LEGEND = 80

pio.templates["mb"] = go.layout.Template(
    layout=dict(
        colorway=CAT,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=INK, size=13),
        # заголовок притиснутий до верху контейнера, легенда — окремим рядком під ним;
        # без цього горизонтальна легенда (y=1.02) сідає рівно на текст заголовка
        title=dict(font=dict(size=15, color=INK), x=0, xanchor="left",
                   y=1.0, yanchor="top", pad=dict(t=14, b=0, l=0)),
        margin=dict(l=8, r=8, t=T_MARGIN_PLAIN, b=8),
        hoverlabel=dict(bgcolor=SURFACE, bordercolor=GRID, font=dict(color=INK, size=12)),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0,
            font=dict(color=INK_MUTED, size=12), bgcolor="rgba(0,0,0,0)",
        ),
        # automargin обов'язковий: свій шаблон повністю заміщає дефолтний разом із
        # його automargin=True, а при margin l/r/b=8 підписи осей просто обрізаються
        xaxis=dict(gridcolor=GRID, zeroline=False, linecolor=GRID, automargin=True,
                   tickfont=dict(color=INK_MUTED), title=dict(font=dict(color=INK_MUTED))),
        yaxis=dict(gridcolor=GRID, zeroline=False, linecolor=GRID, automargin=True,
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

def _width_kw(fn, **extra) -> dict:
    """«Розтягнути на ширину контейнера» у термінах конкретної версії Streamlit.

    Версію рахувати не можна: `width="stretch"` з'явився в st.dataframe раніше,
    ніж у st.plotly_chart (у 1.50 його там досі немає, і аргумент тихо падає
    в **kwargs, даючи deprecation-warning на кожному графіку). Тому дивимо
    на фактичну сигнатуру.
    """
    params = inspect.signature(fn).parameters
    if "width" in params:
        extra["width"] = "stretch"
    elif "use_container_width" in params:
        extra["use_container_width"] = True
    return extra


def _legend_visible(fig: go.Figure) -> bool:
    """Чи намалює Plotly легенду для цієї фігури."""
    if fig.layout.showlegend is False:
        return False
    named = sum(
        1 for tr in fig.data
        if getattr(tr, "showlegend", None) is not False and getattr(tr, "name", None)
    )
    return bool(fig.layout.showlegend) or named > 1


def show(fig: go.Figure, height: int = 340) -> None:
    """Єдина точка рендеру: висота, верхній відступ під легенду, конфіг, ширина."""
    fig.update_layout(
        height=height,
        margin=dict(t=T_MARGIN_LEGEND if _legend_visible(fig) else T_MARGIN_PLAIN),
    )
    st.plotly_chart(fig, **_width_kw(st.plotly_chart, config=PLOTLY_CFG))


def show_df(data: pd.DataFrame, **kw) -> None:
    st.dataframe(data, **_width_kw(st.dataframe, **kw))


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

# --------------------------------------------------------------------------- #
# Прогнозна модель — по ВСІЙ історії, спільна для табів «Прогноз» і «Цілі»
# --------------------------------------------------------------------------- #

FC_HORIZON = 90                      # горизонт для ймовірностей цілей
FC_SEED = 20240501                   # фіксований seed: числа не «стрибають» між рерайтами
FC_SIMS_N = 4000

y_all = daily_all["count"].to_numpy(dtype=float)
wd_all = daily_all["d"].dt.weekday.to_numpy()
gaps_all_min = gaps_minutes(ALL_EVENTS)
gaps_all_h = gaps_all_min.to_numpy(dtype=float) / 60.0
total_all = int(y_all.sum())

FC_MODEL = sm.fit_rate_model(y_all, wd_all, window=120)
FC_DATES = pd.date_range(TODAY + timedelta(days=1), periods=max(FC_HORIZON, (date(TODAY.year, 12, 31) - TODAY).days), freq="D")
FC_SIMS = sm.simulate_future(FC_MODEL, FC_DATES.weekday.to_numpy(), n_sims=FC_SIMS_N, seed=FC_SEED)
FC_CUM = np.cumsum(FC_SIMS, axis=1)
MK_ALL = sm.markov_daily(y_all > 0)

# календарні тижні в межах горизонту (лише повні) — для цілі «N разів за тиждень»
_wk_starts = np.flatnonzero(np.r_[True, FC_DATES.weekday.to_numpy()[1:] == 0])
_wk_len = np.diff(np.r_[_wk_starts, len(FC_DATES)])
FC_WEEK_SUMS = np.add.reduceat(FC_SIMS, _wk_starts, axis=1)[:, _wk_len == 7]


def num(v, spec: str = ".2f", dash: str = "—", plus: bool = False) -> str:
    """Число для UI: nan/inf ніколи не доходять до екрана."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return dash
    return format(v, ("+" if plus else "") + spec) if np.isfinite(v) else dash


def fmt_p(p: float) -> str:
    if not np.isfinite(p):
        return "p = —"
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def sig_word(p: float, alpha: float = 0.05) -> str:
    if not np.isfinite(p):
        return "не перевірявся — замало даних"
    return "статистично значущий" if p < alpha else "не відрізняється від випадкового"


def fmt_hour(h: float) -> str:
    if not np.isfinite(h):
        return "—"
    hh = int(h) % 24
    return f"{hh:02d}:{int(round((h - int(h)) * 60)) % 60:02d}"


def human_days(d: float) -> str:
    """Очікувана кількість днів у вигляді, який можна прочитати.

    Марковський час до стріку 365 днів — це 1.9e19 днів; «18 710 492 235 363 102 720 дн»
    на екрані не несе інформації, тому великі значення згортаються.
    """
    if not np.isfinite(d):
        return "не досяжно"
    if d < 400:
        return f"≈ {d:.0f} дн"
    if d < 400 * 365:
        return f"≈ {d / 365:.0f} р."
    return "практично недосяжно"


def events_word(n: int) -> str:
    """подія / події / подій — за українським правилом."""
    n = abs(int(n))
    if n % 100 in (11, 12, 13, 14):
        return "подій"
    return {1: "подія", 2: "події", 3: "події", 4: "події"}.get(n % 10, "подій")


def fc_date(days: float) -> str:
    """Номер дня горизонту → дата."""
    if not np.isfinite(days):
        return "—"
    return f"{(TODAY + timedelta(days=int(round(days)))):%d.%m.%Y}"


st.divider()

tab_overview, tab_patterns, tab_dist, tab_stats, tab_forecast, tab_goals = st.tabs(
    ["📈 Огляд", "🧭 Патерни", "📊 Розподіли", "🧮 Статистика", "🔮 Прогноз", "🏆 Цілі"]
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
# ТАБ 4 — Статистика (усе по вибраному періоду, якщо не сказано інакше)
# --------------------------------------------------------------------------- #

with tab_stats:
    y = daily["count"].to_numpy(dtype=float)
    x_idx = np.arange(y.size, dtype=float)
    gaps_h = gap_min.to_numpy(dtype=float) / 60.0 if len(gap_min) else np.array([])

    mean_d, mean_half = sm.mean_ci(y)
    vmr, vmr_chi2, vmr_p = sm.dispersion_test(y)
    burst, memory = sm.burstiness(gaps_h)
    gini_days = sm.gini(y)
    cs = sm.circ_stats(df["hour_f"].to_numpy(dtype=float))
    h_hour, h_hour_n = sm.entropy_bits(by_hour["count"].to_numpy(dtype=float))
    h_wd, h_wd_n = sm.entropy_bits(by_wd["count"].to_numpy(dtype=float))
    mk = sm.markov_daily(y > 0)
    wk_k, wk_lam, wk_r2 = sm.weibull_fit(gaps_h)

    s1 = st.columns(5)
    s1[0].metric("Подій / день", num(mean_d), f"± {num(mean_half)} (95% CI)", delta_color="off")
    s1[1].metric(
        "Індекс дисперсії", num(vmr),
        (("серіями" if vmr > 1 else "рівномірніше") + f" · {fmt_p(vmr_p)}"
         if np.isfinite(vmr) else "замало днів"),
        delta_color="off",
    )
    s1[2].metric("Burstiness B", num(burst, plus=True),
                 "−1 періодично · 0 пуассон · +1 серіями", delta_color="off")
    s1[3].metric("Пам'ять інтервалів M", num(memory, plus=True),
                 "кореляція сусідніх пауз", delta_color="off")
    s1[4].metric("Джині по днях", num(gini_days), "0 рівно · 1 усе в одному дні", delta_color="off")

    s2 = st.columns(5)
    s2[0].metric("Круговий сер. час", fmt_hour(cs.mean_hour), "центр маси доби", delta_color="off")
    s2[1].metric("Розкид по добі", f"± {num(cs.sd_hours, '.1f')} год",
                 f"концентрація R = {num(cs.R)}", delta_color="off")
    s2[2].metric("Тест Релея", fmt_p(cs.rayleigh_p).replace("p = ", "").replace("p < ", "<"),
                 ("години нерівномірні" if cs.rayleigh_p < 0.05 else "рівномірні по добі")
                 if np.isfinite(cs.rayleigh_p) else "замало подій", delta_color="off")
    s2[3].metric("Ентропія годин", f"{num(h_hour)} біт",
                 f"{num(h_hour_n * 100, '.0f')}% від максимуму", delta_color="off")
    s2[4].metric("Ентропія днів тижня", f"{num(h_wd)} біт",
                 f"{num(h_wd_n * 100, '.0f')}% від максимуму", delta_color="off")

    if not np.isfinite(vmr):
        vmr_line = "поки не рахується — замало днів у періоді"
    else:
        vmr_stat = f"{num(vmr)} (χ² = {num(vmr_chi2, '.0f')} на df = {y.size - 1}, {fmt_p(vmr_p)})"
        vmr_line = (f"{vmr_stat} — події злипаються в серії: активний день тягне за собою ще один"
                    if vmr > 1.2 else f"{vmr_stat} — розкид близький до випадкового")

    with st.expander("Як читати ці числа"):
        st.markdown(
            f"""
- **Індекс дисперсії (VMR)** = дисперсія / середнє по днях. Для чистого
  пуассонівського процесу = 1. Тут {vmr_line}.
- **Burstiness B** = (σ−μ)/(σ+μ) на інтервалах між подіями. −1 — ідеальний
  метроном, 0 — пуассон, +1 — довгі паузи вперемішку зі шквалами.
- **Пам'ять M** — кореляція сусідніх інтервалів. M > 0: за довгою паузою
  частіше йде довга (режим «то тиждень нічого, то щодня»).
- **Ентропія** — скільки бітів невизначеності в тому, коли саме станеться
  подія. 24 години рівномірно = log₂24 = 4.58 біт; менше значення = чіткіший
  розклад. Нормована — частка від цього максимуму.
- **Тест Релея** — H₀: години розкидані по добі рівномірно. Малий p означає,
  що добовий пік не випадковість.
- **Кругова статистика** трактує годину як кут (23:30 і 00:30 — сусіди),
  тому «середній час» не ламається об полуніч, як звичайне середнє.
"""
        )

    st.markdown("#### Тренд інтенсивності")
    tr = sm.trend(y)
    fig = go.Figure()
    fig.add_bar(
        x=daily["d"], y=y, name="За день", marker_color=CAT[0], opacity=0.35, marker_line_width=0,
        hovertemplate="%{x|%d.%m.%Y}<br>%{y} подій<extra></extra>",
    )
    if np.isfinite(tr.slope):
        fig.add_scatter(
            x=daily["d"], y=tr.intercept + tr.slope * x_idx, name="OLS",
            mode="lines", line=dict(color=CAT[1], width=2),
            hovertemplate="%{y:.2f}/день<extra>OLS</extra>",
        )
    if np.isfinite(tr.ts_slope):
        fig.add_scatter(
            x=daily["d"], y=tr.ts_intercept + tr.ts_slope * x_idx, name="Тейл–Сен (робастний)",
            mode="lines", line=dict(color=CAT[2], width=2, dash="dash"),
            hovertemplate="%{y:.2f}/день<extra>Тейл–Сен</extra>",
        )
    fig.update_layout(title="Тренд: чи змінюється інтенсивність", yaxis_title="подій", hovermode="x unified")
    rounded_bars(fig)
    show(fig, 360)
    parts = [f"OLS: {num(tr.slope * 30, '.2f', plus=True)} подій/день за кожні 30 днів "
             f"({fmt_p(tr.p)}, R² = {num(tr.r2, '.3f')})."]
    if np.isfinite(tr.ts_slope):
        parts.append(f"Тейл–Сен (не боїться викидів): {num(tr.ts_slope * 30, '.2f', plus=True)}/30 дн.")
    parts.append(
        f"Манн–Кендалл: τ = {num(tr.tau, '.3f', plus=True)}, {fmt_p(tr.mk_p)} → тренд {sig_word(tr.mk_p)}."
        if np.isfinite(tr.mk_p) else
        "Манн–Кендалл не рахувався: для непараметричного тесту треба ≥8 днів."
    )
    st.caption(" ".join(parts))

    st.markdown("#### Контрольна карта")
    win = int(min(28, max(7, days_in_range // 4)))
    center = daily["count"].rolling(win, center=True, min_periods=max(3, win // 3)).mean()
    c_round = np.round(np.nan_to_num(center.to_numpy(dtype=float), nan=0.0), 1)
    _lims = {v: sm.poisson_interval(float(v)) for v in np.unique(c_round)}
    lo = np.array([_lims[v][0] for v in c_round], dtype=float)
    hi = np.array([_lims[v][1] for v in c_round], dtype=float)
    flag_up = (y > hi) & (c_round > 0.2)
    flag_dn = (y < lo) & (c_round > 0.2)

    fig = go.Figure()
    fig.add_scatter(x=daily["d"], y=hi, mode="lines", line=dict(width=0), showlegend=False,
                    hoverinfo="skip", name="")
    fig.add_scatter(
        x=daily["d"], y=lo, mode="lines", line=dict(width=0), fill="tonexty",
        fillcolor="rgba(57,135,229,0.12)" if DARK else "rgba(42,120,214,0.10)",
        name="Межі Пуассона 3σ", hoverinfo="skip",
    )
    fig.add_scatter(x=daily["d"], y=center, mode="lines", name=f"Центр ({win} дн)",
                    line=dict(color=CAT[0], width=2),
                    hovertemplate="%{x|%d.%m.%Y}<br>норма %{y:.2f}/день<extra></extra>")
    fig.add_scatter(x=daily["d"], y=y, mode="markers", name="Факт",
                    marker=dict(color=INK_MUTED, size=4),
                    hovertemplate="%{x|%d.%m.%Y}<br>%{y} подій<extra></extra>")
    if flag_up.any():
        fig.add_scatter(x=daily["d"][flag_up], y=y[flag_up], mode="markers", name="Вище норми",
                        marker=dict(color=STATUS["good"], size=9, symbol="circle-open", line=dict(width=2)),
                        hovertemplate="%{x|%d.%m.%Y}<br>%{y} подій — вище 3σ<extra></extra>")
    if flag_dn.any():
        fig.add_scatter(x=daily["d"][flag_dn], y=y[flag_dn], mode="markers", name="Нижче норми",
                        marker=dict(color=STATUS["critical"], size=9, symbol="circle-open", line=dict(width=2)),
                        hovertemplate="%{x|%d.%m.%Y}<br>%{y} подій — нижче 3σ<extra></extra>")
    fig.update_layout(title="Що виходить за межі випадкової флуктуації", yaxis_title="подій", hovermode="x unified")
    show(fig, 360)
    n_out = int(flag_up.sum() + flag_dn.sum())
    st.caption(
        f"Центр — плавне ковзне середнє за {win} дн, смуга — центральний інтервал Пуассона 99.73% "
        f"(аналог ±3σ, але коректний для лічильників). Поза межами: {n_out} дн з {days_in_range} "
        f"(випадково очікувалося б ≈ {days_in_range * 0.0027:.1f})."
    )
    if flag_up.any():
        out = pd.DataFrame({
            "Дата": daily["d"][flag_up].dt.strftime("%d.%m.%Y"),
            "Подій": y[flag_up].astype(int),
            "Норма": np.round(center.to_numpy(dtype=float)[flag_up], 2),
            "p (випадково ≥)": [f"{sm.poisson_sf(v - 1, max(c, 1e-9)):.1e}"
                                for v, c in zip(y[flag_up], c_round[flag_up])],
        }).sort_values("Подій", ascending=False)
        with st.expander(f"Аномальні дні ({len(out)})"):
            show_df(out.head(25), hide_index=True)

    st.markdown("#### Точки зламу режиму")
    cps = sm.poisson_changepoints(y, min_seg=max(10, y.size // 24))
    if not cps:
        st.info(
            "Жодного зламу, який виправдав би BIC-штраф: у межах періоду інтенсивність "
            "однорідна, а всі коливання — пуассонівський шум."
        )
    else:
        bounds = [0] + cps + [int(y.size)]
        seg_rows, step_x, step_y = [], [], []
        prev_rate = None
        for a, b in zip(bounds[:-1], bounds[1:]):
            rate = float(y[a:b].mean())
            d0, d1 = daily["d"].iloc[a], daily["d"].iloc[b - 1]
            step_x += [d0, d1]
            step_y += [rate, rate]
            if prev_rate is None:
                change = "—"
            elif prev_rate > 0:
                change = f"{(rate / prev_rate - 1) * 100:+.0f}%"
            else:
                change = "з нуля"
            seg_rows.append({
                "Період": f"{d0:%d.%m.%Y} → {d1:%d.%m.%Y}",
                "Днів": b - a,
                "Подій/день": round(rate, 3),
                "Зміна": change,
            })
            prev_rate = rate
        fig = go.Figure()
        fig.add_bar(x=daily["d"], y=y, name="За день", marker_color=CAT[0], opacity=0.3,
                    marker_line_width=0, hovertemplate="%{x|%d.%m.%Y}<br>%{y} подій<extra></extra>")
        fig.add_scatter(x=step_x, y=step_y, name="Режим (сегментна інтенсивність)", mode="lines",
                        line=dict(color=CAT[1], width=3),
                        hovertemplate="%{y:.2f}/день<extra>режим</extra>")
        for cp in cps:
            fig.add_vline(x=daily["d"].iloc[cp], line=dict(color=INK_MUTED, width=1, dash="dot"))
        fig.update_layout(title=f"Знайдено зламів: {len(cps)}", yaxis_title="подій", hovermode="x unified")
        rounded_bars(fig)
        show(fig, 360)
        show_df(pd.DataFrame(seg_rows), hide_index=True)
        st.caption(
            "Бінарна сегментація з пуассонівською правдоподібністю: злам приймається лише якщо "
            "приріст правдоподібності перевищує штраф 3·ln(N) — випадкові сплески через це не проходять."
        )

    c1, c2 = st.columns(2)

    with c1:
        a_vals, band = sm.acf(y, min(60, max(2, y.size // 3)))
        lags = np.arange(1, a_vals.size + 1)
        fig = go.Figure()
        fig.add_bar(
            x=lags, y=a_vals, marker_color=[CAT[1] if L % 7 == 0 else CAT[0] for L in lags],
            marker_line_width=0, name="ACF",
            hovertemplate="лаг %{x} дн<br>r = %{y:.3f}<extra></extra>",
        )
        for sgn in (1, -1):
            fig.add_hline(y=sgn * band, line=dict(color=INK_MUTED, width=1, dash="dot"))
        fig.update_layout(title="Автокореляція по днях", xaxis_title="лаг, днів",
                          yaxis_title="r", showlegend=False)
        rounded_bars(fig)
        show(fig, 330)
        wk_lags = [L for L in (7, 14, 21, 28) if L <= a_vals.size]
        wk_txt = ", ".join(f"лаг {L}: {a_vals[L - 1]:+.2f}" for L in wk_lags) or "—"
        st.caption(
            f"Пунктир — межа випадковості ±{band:.2f}. Оранжеві — кратні тижню ({wk_txt}). "
            "Стовпчик за межами пунктиру означає, що день «пам'ятає» такий самий день N днів тому."
        )

    with c2:
        per, pw, thr = sm.periodogram(y)
        if per.size:
            order = np.argsort(per)
            fig = go.Figure()
            fig.add_scatter(x=per[order], y=pw[order], mode="lines", name="Потужність",
                            line=dict(color=CAT[0], width=2),
                            hovertemplate="період %{x:.1f} дн<br>потужність %{y:.1f}<extra></extra>")
            if np.isfinite(thr):
                fig.add_hline(y=thr, line=dict(color=CAT[3], width=1, dash="dash"),
                              annotation_text="поріг Фішера 95%", annotation_font_color=INK_MUTED)
            fig.update_layout(title="Періодограма: які цикли справді є",
                              xaxis_title="період, днів", yaxis_title="потужність", showlegend=False)
            fig.update_xaxes(type="log", tickvals=[2, 3, 5, 7, 14, 30, 60, 90])
            show(fig, 330)
            peaks = order[np.argsort(pw[order])[::-1]][:3]
            top_txt = " · ".join(
                f"{per[i]:.1f} дн ({'значущий' if np.isfinite(thr) and pw[i] > thr else 'шум'})"
                for i in peaks
            )
            st.caption(
                f"Топ-3 піки: {top_txt}. Поріг Фішера враховує, що перевіряється багато частот "
                "одразу — без нього «цикл» знаходиться завжди."
            )
        else:
            st.info("Замало днів для спектрального аналізу.")

    c3, c4 = st.columns(2)

    with c3:
        chi2_wd, df_wd, p_wd, res_wd = sm.chi2_gof(by_wd["count"].to_numpy(dtype=float))
        fig = go.Figure()
        fig.add_bar(
            x=by_wd["weekday_ua"], y=res_wd,
            marker_color=[CAT[0] if r >= 0 else CAT[1] for r in res_wd], marker_line_width=0,
            hovertemplate="%{x}<br>залишок %{y:+.2f}σ<extra></extra>", name="залишок",
        )
        for sgn in (2, -2):
            fig.add_hline(y=sgn, line=dict(color=INK_MUTED, width=1, dash="dot"))
        fig.update_layout(title="Які дні тижня ламають рівномірність",
                          yaxis_title="стандартизований залишок, σ", xaxis_title=None, showlegend=False)
        rounded_bars(fig)
        show(fig, 330)
        st.caption(
            f"χ² = {num(chi2_wd, '.1f')}, df = {df_wd}, {fmt_p(p_wd)} → розподіл по днях тижня "
            f"{'нерівномірний' if p_wd < 0.05 else 'не відрізняється від рівномірного'}. "
            "За межами ±2σ — дні, які роблять основний внесок у χ²."
        )

    with c4:
        chi2_h, df_h, p_h, res_h = sm.chi2_gof(by_hour["count"].to_numpy(dtype=float))
        fig = go.Figure()
        fig.add_bar(
            x=[f"{h:02d}" for h in range(24)], y=res_h,
            marker_color=[CAT[0] if r >= 0 else CAT[1] for r in res_h], marker_line_width=0,
            hovertemplate="%{x}:00<br>залишок %{y:+.2f}σ<extra></extra>", name="залишок",
        )
        for sgn in (2, -2):
            fig.add_hline(y=sgn, line=dict(color=INK_MUTED, width=1, dash="dot"))
        fig.update_layout(title="Які години ламають рівномірність",
                          yaxis_title="стандартизований залишок, σ", xaxis_title="година", showlegend=False)
        fig.update_xaxes(dtick=2)
        rounded_bars(fig)
        show(fig, 330)
        st.caption(f"χ² = {num(chi2_h, '.1f')}, df = {df_h}, {fmt_p(p_h)}. Тест Релея на ті самі дані: "
                   f"{fmt_p(cs.rayleigh_p)} (він враховує циклічність доби, χ² — ні).")

    c5, c6 = st.columns(2)

    with c5:
        mo = df.assign(m=df["ts"].dt.tz_localize(None).dt.to_period("M").dt.to_timestamp())
        rows = [
            (m, sm.circ_mean_hour(g["hour_f"].to_numpy(dtype=float)), len(g))
            for m, g in mo.groupby("m") if len(g) >= 5
        ]
        if len(rows) >= 4:
            months = pd.to_datetime([r[0] for r in rows])
            hours_u = sm.unwrap_hours([r[1] for r in rows])
            sizes = np.array([r[2] for r in rows], dtype=float)
            slope, intercept, se, p_dr, r2_dr = sm.ols_fit(hours_u, np.arange(len(rows), dtype=float))
            fig = go.Figure()
            fig.add_scatter(
                x=months, y=hours_u, mode="markers", name="Круговий сер. час за місяць",
                marker=dict(color=CAT[0], size=6 + 12 * sizes / sizes.max(),
                            line=dict(color=SURFACE, width=1)),
                customdata=sizes,
                hovertemplate="%{x|%Y-%m}<br>%{y:.2f} год<br>%{customdata:.0f} подій<extra></extra>",
            )
            if np.isfinite(slope):
                fig.add_scatter(
                    x=months, y=intercept + slope * np.arange(len(rows)), mode="lines",
                    name="Тренд", line=dict(color=CAT[1], width=2, dash="dash"),
                    hovertemplate="%{y:.2f} год<extra>тренд</extra>",
                )
            fig.update_layout(title="Дрейф хронотипу: куди повзе режим",
                              yaxis_title="година доби", xaxis_title=None)
            show(fig, 330)
            st.caption(
                f"Зсув {num(slope * 60, '.0f', plus=True)} хв за місяць ({fmt_p(p_dr)}, "
                f"R² = {num(r2_dr)}) — "
                f"{sig_word(p_dr)}. Місяці з <5 подіями пропущені; ряд розгорнутий через полуніч, "
                "тому значення можуть виходити за 0–24."
            )
        else:
            st.info("Для дрейфу хронотипу потрібно ≥4 місяці з даними.")

    with c6:
        fx, fy = sm.lorenz(y)
        share_for_half = 1.0 - float(np.interp(0.5, fy, fx))
        fig = go.Figure()
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Ідеальна рівність",
                        line=dict(color=INK_MUTED, width=1, dash="dot"), hoverinfo="skip")
        fig.add_scatter(
            x=fx, y=fy, mode="lines", name="Факт", line=dict(color=CAT[0], width=2),
            fill="tozeroy", fillcolor="rgba(57,135,229,0.13)" if DARK else "rgba(42,120,214,0.10)",
            hovertemplate="%{x:.0%} найспокійніших днів<br>%{y:.0%} подій<extra></extra>",
        )
        fig.update_layout(title="Крива Лоренца: наскільки все зосереджено",
                          xaxis_title="частка днів (від найспокійніших)", yaxis_title="частка подій")
        fig.update_xaxes(tickformat=".0%")
        fig.update_yaxes(tickformat=".0%")
        show(fig, 330)
        st.caption(
            f"Джині = {gini_days:.2f}. Половина всіх подій припадає на {share_for_half:.0%} "
            f"найактивніших днів ({share_for_half * days_in_range:.0f} із {days_in_range})."
        )

    st.markdown("#### Інерція днів: ланцюг Маркова")
    c7, c8 = st.columns([1.0, 1.3])
    with c7:
        if np.isfinite(mk.p01) and np.isfinite(mk.p11):
            z = np.array([[1 - mk.p01, mk.p01], [1 - mk.p11, mk.p11]])
            fig = go.Figure(
                go.Heatmap(
                    z=z, x=["пропуск", "активний"], y=["пропуск", "активний"], xgap=3, ygap=3,
                    colorscale=[[i / (len(SEQ) - 1), c] for i, c in enumerate(SEQ)], zmin=0, zmax=1,
                    text=[[f"{v:.0%}" for v in row] for row in z], texttemplate="%{text}",
                    textfont=dict(size=15),
                    hovertemplate="сьогодні %{y} → завтра %{x}<br>%{z:.1%}<extra></extra>",
                    colorbar=dict(title="P", thickness=10, len=0.8, outlinewidth=0, tickformat=".0%"),
                )
            )
            fig.update_layout(title="Матриця переходів день → наступний день",
                              xaxis=dict(title="завтра", showgrid=False),
                              yaxis=dict(title="сьогодні", showgrid=False, autorange="reversed"))
            show(fig, 300)
        else:
            st.info("Замало днів для матриці переходів.")
    with c8:
        m1 = st.columns(2)
        m1[0].metric("P(активний | був пропуск)", f"{mk.p01:.0%}" if np.isfinite(mk.p01) else "—",
                     delta_color="off")
        m1[1].metric("P(активний | був активний)", f"{mk.p11:.0%}" if np.isfinite(mk.p11) else "—",
                     delta_color="off")
        m2 = st.columns(2)
        m2[0].metric("Інерція (p₁₁ − p₀₁)", f"{mk.persistence:+.0%}" if np.isfinite(mk.persistence) else "—",
                     f"χ² = {num(mk.chi2, '.0f')}, {fmt_p(mk.p)}" if np.isfinite(mk.chi2) else "замало днів",
                     delta_color="off")
        m2[1].metric("Стаціонарна активність", f"{mk.stationary:.0%}" if np.isfinite(mk.stationary) else "—",
                     f"факт: {coverage:.0f}%", delta_color="off")
        pers_txt = f"{mk.persistence * 100:+.0f} п.п." if np.isfinite(mk.persistence) else "—"
        st.caption(
            "Дні не незалежні: активний день змінює ймовірність наступного на "
            f"{pers_txt} ({sig_word(mk.p)}). Тут — по вибраному періоду; у табі «Прогноз» "
            "та сама модель рахується по всій історії й дає очікувані стріки — тому оцінка "
            "стріків не наївна «p у степені N», а марковська."
        )

    with st.expander("Розподіл інтервалів: чи це взагалі пуассонівський процес"):
        if not np.isfinite(wk_k):
            st.info("Замало інтервалів для оцінки Вейбулла.")
        else:
            st.markdown(
                f"""
Оцінка Вейбулла за інтервалами періоду: **k = {num(wk_k)}**, λ = {num(wk_lam, ".1f")} год
(R² = {num(wk_r2, ".3f")}).

- k < 1 — чим довше триває пауза, тим менша ймовірність, що вона от-от
  скінчиться (важкий хвіст, серійність);
- k = 1 — рівно експоненційний розподіл, тобто процес без пам'яті (Пуассон);
- k > 1 — інтервали регулярніші за випадкові.

Разом з VMR = {num(vmr)} і B = {num(burst, plus=True)} це дає узгоджену картину:
{"процес понад-дисперсний і серійний — модель «просто Пуассон» його недооцінює"
 if (np.isfinite(vmr) and vmr > 1.2) else "розкид близький до пуассонівського"}.
Крива виживання й функція інтенсивності — у табі «Прогноз».
"""
        )

# --------------------------------------------------------------------------- #
# ТАБ 5 — Прогноз (модель по ВСІЙ історії, не по вибраному діапазону)
# --------------------------------------------------------------------------- #

with tab_forecast:
    st.caption(
        f"Усе на цій вкладці рахується по всій історії ({DATA_MIN:%d.%m.%Y} → {DATA_MAX:%d.%m.%Y}, "
        f"{total_all} подій), бо прогноз від вибраного діапазону не має сенсу. "
        f"Модель: інтенсивність за останні {FC_MODEL.window} дн × тижневі множники, "
        f"{'негативно-біноміальна' if FC_MODEL.vmr > 1.05 else 'пуассонівська'} дисперсія "
        f"(VMR = {FC_MODEL.vmr:.2f}), {FC_SIMS_N} симуляцій."
    )

    last_ts_all = ALL_EVENTS["ts"].max()
    since_last_all = datetime.now(KYIV_TZ) - last_ts_all.to_pydatetime()
    elapsed_h = since_last_all.total_seconds() / 3600.0
    wk_k_all, wk_lam_all, wk_r2_all = sm.weibull_fit(gaps_all_h)

    st.markdown("#### Скільки чекати наступної")
    f1 = st.columns(5)
    f1[0].metric("Минуло від останньої", human_td(since_last_all),
                 f"{last_ts_all:%d.%m %H:%M}", delta_color="off")
    for i, hrs in enumerate((6, 12, 24, 48)):
        pr = sm.cond_prob_within(gaps_all_h, elapsed_h, hrs)
        f1[i + 1].metric(
            f"P(подія ≤ {hrs} год)", "—" if not np.isfinite(pr) else f"{pr:.0%}",
            "умовно, з урахуванням паузи", delta_color="off",
        )

    med_gap = float(np.median(gaps_all_h)) if gaps_all_h.size else np.nan
    mean_gap = float(np.mean(gaps_all_h)) if gaps_all_h.size else np.nan
    lb_gap = sm.length_biased_mean(gaps_all_h)
    f2 = st.columns(4)
    f2[0].metric("Медіанний інтервал", human_minutes(med_gap * 60), "50% пауз коротші", delta_color="off")
    f2[1].metric("Середній інтервал", human_minutes(mean_gap * 60), "хвіст тягне вгору", delta_color="off")
    f2[2].metric("Парадокс очікування", human_minutes(lb_gap * 60),
                 "у яку паузу попадає випадковий момент", delta_color="off")
    f2[3].metric("Форма Вейбулла k", num(wk_k_all),
                 "k<1: чим довша пауза, тим менший шанс", delta_color="off")

    c1, c2 = st.columns(2)

    with c1:
        if gaps_all_h.size >= 20:
            t_s, s_s = sm.survival(gaps_all_h)
            keep = s_s > 0
            t_s, s_s = t_s[keep], s_s[keep]
            if t_s.size > 1500:  # прорідження суто для ваги графіка
                sel = np.unique(np.linspace(0, t_s.size - 1, 1500).astype(int))
                t_s, s_s = t_s[sel], s_s[sel]
            fig = go.Figure()
            fig.add_scatter(x=t_s, y=s_s, mode="lines", name="Факт",
                            line=dict(color=CAT[0], width=2, shape="hv"),
                            hovertemplate="через %{x:.1f} год ще нічого: %{y:.1%}<extra></extra>")
            grid = np.linspace(max(t_s.min(), 1e-3), t_s.max(), 200)
            fig.add_scatter(x=grid, y=np.exp(-grid / mean_gap), mode="lines",
                            name="Експоненційний (Пуассон)",
                            line=dict(color=CAT[1], width=2, dash="dot"),
                            hovertemplate="%{y:.1%}<extra>експоненційний</extra>")
            if np.isfinite(wk_k_all) and np.isfinite(wk_lam_all) and wk_lam_all > 0:
                fig.add_scatter(x=grid, y=np.exp(-((grid / wk_lam_all) ** wk_k_all)), mode="lines",
                                name=f"Вейбулл k={wk_k_all:.2f}",
                                line=dict(color=CAT[2], width=2, dash="dash"),
                                hovertemplate="%{y:.1%}<extra>Вейбулл</extra>")
            fig.update_layout(title="Крива виживання: P(пауза довша за X)",
                              xaxis_title="годин", yaxis_title="P")
            fig.update_yaxes(type="log", tickformat=".0%")
            show(fig, 340)
            st.caption(
                "Лог-шкала по вертикалі: експоненційний розподіл на ній — пряма. "
                f"Факт відхиляється від прямої → процес не безпам'ятний (k = {num(wk_k_all)}, "
                f"R² підгонки {num(wk_r2_all, '.3f')})."
            )
        else:
            st.info("Замало інтервалів для кривої виживання.")

    with c2:
        if gaps_all_h.size >= 30:
            hmax = float(np.percentile(gaps_all_h, 90))
            grid = np.linspace(0, max(hmax, 6.0), 60)
            hz24 = sm.hazard_curve(gaps_all_h, 24.0, grid)
            hz6 = sm.hazard_curve(gaps_all_h, 6.0, grid)
            fig = go.Figure()
            fig.add_scatter(x=grid, y=hz24 * 100, mode="lines", name="у наступні 24 год",
                            line=dict(color=CAT[0], width=2),
                            hovertemplate="минуло %{x:.0f} год → %{y:.0f}%<extra>24 год</extra>")
            fig.add_scatter(x=grid, y=hz6 * 100, mode="lines", name="у наступні 6 год",
                            line=dict(color=CAT[1], width=2, dash="dash"),
                            hovertemplate="минуло %{x:.0f} год → %{y:.0f}%<extra>6 год</extra>")
            if elapsed_h <= grid.max():
                fig.add_vline(x=elapsed_h, line=dict(color=INK_MUTED, width=1, dash="dot"),
                              annotation_text="зараз", annotation_font_color=INK_MUTED)
            fig.update_layout(title="Функція інтенсивності: шанс залежно від того, скільки минуло",
                              xaxis_title="годин без подій", yaxis_title="%")
            fig.update_yaxes(range=[0, 100])
            show(fig, 340)
            st.caption(
                "Якби процес був пуассонівським, обидві лінії були б горизонтальними. "
                "Спадання вправо — та сама серійність: чим довша пауза, тим менший шанс, "
                "що вона от-от перерветься. Точки, де в ризику <5 спостережень, не малюються."
            )
        else:
            st.info("Замало інтервалів для функції інтенсивності.")

    st.markdown("#### Прогноз накопичення")
    hist = daily_all.copy()
    hist["cum"] = hist["count"].cumsum()
    hist_tail = hist.tail(120)
    fan = sm.quantile_fan(FC_CUM)
    fx = FC_DATES
    fig = go.Figure()
    fig.add_scatter(x=fx, y=total_all + fan[0.95], mode="lines", line=dict(width=0),
                    showlegend=False, hoverinfo="skip", name="")
    fig.add_scatter(x=fx, y=total_all + fan[0.05], mode="lines", line=dict(width=0), fill="tonexty",
                    fillcolor="rgba(57,135,229,0.14)" if DARK else "rgba(42,120,214,0.12)",
                    name="90% інтервал", hoverinfo="skip")
    fig.add_scatter(x=fx, y=total_all + fan[0.75], mode="lines", line=dict(width=0),
                    showlegend=False, hoverinfo="skip", name="")
    fig.add_scatter(x=fx, y=total_all + fan[0.25], mode="lines", line=dict(width=0), fill="tonexty",
                    fillcolor="rgba(57,135,229,0.28)" if DARK else "rgba(42,120,214,0.24)",
                    name="50% інтервал", hoverinfo="skip")
    fig.add_scatter(x=hist_tail["d"], y=hist_tail["cum"], mode="lines", name="Факт",
                    line=dict(color=INK_MUTED, width=2),
                    hovertemplate="%{x|%d.%m.%Y}<br>разом %{y:.0f}<extra></extra>")
    fig.add_scatter(x=fx, y=total_all + fan[0.5], mode="lines", name="Медіана прогнозу",
                    line=dict(color=CAT[0], width=2, dash="dash"),
                    hovertemplate="%{x|%d.%m.%Y}<br>≈ %{y:.0f}<extra></extra>")
    fig.update_layout(title="Скільки буде всього: медіана і смуги невизначеності",
                      yaxis_title="подій разом (з початку історії)", hovermode="x unified")
    show(fig, 380)

    eom = date(TODAY.year + (TODAY.month == 12), (TODAY.month % 12) + 1, 1) - timedelta(days=1)
    month_left = (eom - TODAY).days
    month_done = int(ALL_EVENTS.loc[
        (ALL_EVENTS["d"] >= date(TODAY.year, TODAY.month, 1)) & (ALL_EVENTS["d"] <= TODAY)
    ].shape[0])
    year_left = (date(TODAY.year, 12, 31) - TODAY).days
    year_done = int(ALL_EVENTS.loc[ALL_EVENTS["d"] >= date(TODAY.year, 1, 1)].shape[0])

    f3 = st.columns(4)
    if month_left > 0:
        mc = FC_CUM[:, month_left - 1]
        f3[0].metric("До кінця місяця", f"+{np.median(mc):.0f}",
                     f"90%: {np.quantile(mc, 0.05):.0f}…{np.quantile(mc, 0.95):.0f}", delta_color="off")
        f3[1].metric("Разом за місяць", f"{month_done + np.median(mc):.0f}",
                     f"уже {month_done} за {TODAY.day} дн", delta_color="off")
    else:
        f3[0].metric("До кінця місяця", "—", "останній день місяця", delta_color="off")
        f3[1].metric("Разом за місяць", f"{month_done}", "факт", delta_color="off")
    if year_left > 0:
        yc = FC_CUM[:, min(year_left, FC_CUM.shape[1]) - 1]
        f3[2].metric(f"Разом за {TODAY.year}", f"{year_done + np.median(yc):.0f}",
                     f"90%: {year_done + np.quantile(yc, 0.05):.0f}…{year_done + np.quantile(yc, 0.95):.0f}",
                     delta_color="off")
    else:
        f3[2].metric(f"Разом за {TODAY.year}", f"{year_done}", "рік закінчився", delta_color="off")
    f3[3].metric("Темп моделі", f"{num(FC_MODEL.mu)}/день",
                 "макс. множник: " + WEEKDAY_UA[int(np.argmax(FC_MODEL.wd_factor))]
                 + f" ×{FC_MODEL.wd_factor.max():.2f}", delta_color="off")

    c3, c4 = st.columns([1.0, 1.0])

    with c3:
        fig = go.Figure()
        fig.add_bar(
            x=WEEKDAY_UA, y=FC_MODEL.wd_factor, marker_color=CAT[0], marker_line_width=0,
            hovertemplate="%{x}: ×%{y:.2f}<extra></extra>", name="множник",
        )
        fig.add_hline(y=1.0, line=dict(color=INK_MUTED, width=1, dash="dot"))
        fig.update_layout(title="Тижневі множники моделі", yaxis_title="× до середнього",
                          xaxis_title=None, showlegend=False)
        rounded_bars(fig)
        show(fig, 300)
        st.caption(
            f"Оцінені за останні {FC_MODEL.window} дн і стягнуті до 1 (shrinkage), щоб день "
            "із чотирма спостереженнями не отримав випадковий множник ×2."
        )

    with c4:
        steps = (100, 250, 500, 1000)
        targets = sorted({int((total_all // st_) * st_ + st_) for st_ in steps})[:4]
        rows = []
        for t_ in targets:
            frac, q = sm.eta_days(FC_CUM, t_ - total_all)
            rows.append({
                "Рубіж": f"{t_:,}".replace(",", " "),
                "Лишилось": t_ - total_all,
                "Медіанна дата": fc_date(q[0.5]) if frac >= 0.5 else "поза горизонтом",
                "90% діапазон": (f"{fc_date(q[0.05])} … {fc_date(q[0.95])}"
                                 if frac >= 0.95 else f"шанс {frac:.0%} за {len(FC_DATES)} дн"),
            })
        st.markdown("**Коли наступні рубежі**")
        show_df(pd.DataFrame(rows), hide_index=True)
        st.caption(
            f"Дати — квантилі {FC_SIMS_N} симуляцій, не «лишилось ділити на темп»: "
            "у них уже враховані і тижневий ритм, і понад-дисперсія."
        )

    st.markdown("#### Стріки: скільки чекати за поточною динамікою")
    if np.isfinite(MK_ALL.p01) and np.isfinite(MK_ALL.p11):
        rows = []
        for L in (7, 14, 30, 50, 100, 365):
            e_days = sm.expected_days_to_streak(MK_ALL.p01, MK_ALL.p11, L)
            p90 = sm.prob_streak_within(MK_ALL.p01, MK_ALL.p11, L, FC_HORIZON,
                                        start_streak=current_streak, seed=FC_SEED)
            naive = MK_ALL.p11 ** (L - 1) if MK_ALL.p11 > 0 else 0.0
            rows.append({
                "Стрік": f"{L} дн",
                "Досягнуто": "✅" if longest_streak >= L else "—",
                "E[днів з нуля]": human_days(e_days),
                f"P(за {FC_HORIZON} дн)": f"{p90:.1%}",
                "P(дотягнути з активного дня)": f"{naive:.2%}",
            })
        show_df(pd.DataFrame(rows), hide_index=True)
        st.caption(
            f"Марковська модель на всій історії: P(активний|активний) = {MK_ALL.p11:.0%}, "
            f"P(активний|пропуск) = {MK_ALL.p01:.0%}. «E[днів з нуля]» — очікуваний час до першої "
            "такої серії (поглинаючий ланцюг), «P за "
            f"{FC_HORIZON} дн» враховує поточний стрік {current_streak} дн. "
            "Остання колонка — шанс, що вже розпочата серія доживе до потрібної довжини."
        )
    else:
        st.info("Замало днів для марковської оцінки стріків.")

# --------------------------------------------------------------------------- #
# ТАБ 6 — Цілі та досягнення
# --------------------------------------------------------------------------- #

DIFFICULTY = {"easy": ("🟢", "легко"), "medium": ("🟡", "середньо"), "hard": ("🟠", "важко"), "extreme": ("🔴", "екстрим")}

# метрики по ВСІЙ історії — цілі не мають залежати від вибраного діапазону
weekly_all = daily_all.set_index("d")["count"].resample("W-MON", label="left", closed="left").sum()
by_hour_all = ALL_EVENTS.groupby("hour").size().reindex(range(24), fill_value=0)
peak_hour_all = int(by_hour_all.idxmax())
peak_hour_all_count = int(by_hour_all.max())

GOALS = [
    dict(diff="medium", goal="Стрік 50 днів", cur=current_streak, target=50, unit="дн", kind="streak"),
    dict(diff="hard", goal="20 разів за тиждень", cur=int(weekly_all.max()), target=20, unit="", kind="weekly"),
    dict(diff="extreme", goal="10 разів за один день", cur=int(daily_all["count"].max()), target=10, unit="", kind="daily"),
    dict(diff="extreme", goal="Рік без пропусків", cur=longest_streak, target=365, unit="дн", kind="streak"),
    dict(diff="easy", goal="1000 подій усього", cur=total_all, target=1000, unit="", kind="total"),
]

# тижні горизонту, що вміщаються в FC_HORIZON днів — для цілі «N разів за тиждень»
FC_WEEKS_H = FC_WEEK_SUMS[:, : max(1, FC_HORIZON // 7)]


def goal_outlook(g: dict) -> str:
    """Одна фраза: коли/з якою ймовірністю ціль досяжна за поточною динамікою.

    Усе з тієї самої моделі, що й таб «Прогноз»: симуляції для лічильників,
    марковський ланцюг для стріків. Наївного «лишилось / темп» тут немає.
    """
    kind, target, cur = g["kind"], g["target"], g["cur"]
    if cur >= target:
        return "✅ вже виконано"
    horizon = FC_HORIZON
    if kind == "total":
        frac, q = sm.eta_days(FC_CUM, target - cur)
        if frac >= 0.5:
            return (f"медіанна дата ≈ {fc_date(q[0.5])}"
                    + (f" · 90%: {fc_date(q[0.05])} … {fc_date(q[0.95])}" if frac >= 0.95 else ""))
        return f"за {len(FC_DATES)} дн шанс {frac:.0%} — поза горизонтом прогнозу"
    if kind == "streak":
        if not (np.isfinite(MK_ALL.p01) and np.isfinite(MK_ALL.p11)):
            return "замало даних для оцінки"
        pr = sm.prob_streak_within(MK_ALL.p01, MK_ALL.p11, target, horizon,
                                   start_streak=current_streak, seed=FC_SEED)
        e_days = sm.expected_days_to_streak(MK_ALL.p01, MK_ALL.p11, target)
        e_txt = (f"очікувано {human_days(e_days)} з нуля"
                 if np.isfinite(e_days) and e_days < 400 * 365
                 else "за поточної інерції практично недосяжно")
        return f"P(за {horizon} дн) = {pr:.1%} · {e_txt}"
    if kind == "weekly":
        pr = float((FC_WEEKS_H.max(axis=1) >= target).mean())
        best = float(np.median(FC_WEEKS_H.max(axis=1)))
        return f"P(за {horizon} дн) = {pr:.1%} · очікуваний найкращий тиждень ≈ {best:.0f}"
    if kind == "daily":
        sims_h = FC_SIMS[:, :horizon]
        pr = float((sims_h.max(axis=1) >= target).mean())
        best = float(np.median(sims_h.max(axis=1)))
        return f"P(за {horizon} дн) = {pr:.1%} · очікуваний рекорд дня ≈ {best:.0f}"
    return ""

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
            st.markdown(
                f"<div class='goal-note'>{goal_outlook(g)}</div>", unsafe_allow_html=True
            )

        # окрема ціль — не числова, а «попадання»
        hit = peak_hour_all == 11
        h11 = int(by_hour_all.loc[11])
        st.markdown(
            f"<div class='goal-row'><span>🟠 <b>Зламати пікову годину на 11:00</b> "
            f"<span style='color:{INK_MUTED}'>· важко</span></span>"
            f"<span>{'✅ 11:00' if hit else f'зараз {peak_hour_all:02d}:00'}</span></div>",
            unsafe_allow_html=True,
        )
        st.progress(1.0 if hit else safe_div(h11, max(1, peak_hour_all_count)))
        need11 = max(0, peak_hour_all_count - h11) + 1
        st.markdown(
            f"<div class='goal-note'>О 11:00 — {h11} {events_word(h11)}, у лідера "
            f"({peak_hour_all:02d}:00) — {peak_hour_all_count}. "
            f"Відставання об 11:00: {need11}.</div>",
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
