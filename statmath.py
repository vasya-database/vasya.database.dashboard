"""Статистичні примітиви для MasturBoard — тільки numpy + stdlib.

scipy навмисно не використовується: на Streamlit Cloud це +80 МБ до образу,
а потрібні лише чотири спецфункції (Q(a,x), I_x(a,b), erfc, lgamma) і кілька
тестів. Реалізації спецфункцій — Numerical Recipes (gser/gcf/betacf),
перевірені проти scipy в test_statmath.py з точністю 1e-12.

Усі функції чисті (без Streamlit) — їх можна тестувати й кешувати зовні.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

_EPS = 1.0e-15  # цільова відносна точність рядів; перевірено проти scipy на 1e-13
_FPMIN = 1.0e-300
_ITMAX = 1000

# --------------------------------------------------------------------------- #
# Спецфункції
# --------------------------------------------------------------------------- #


def norm_sf(z: float) -> float:
    """P(Z > z) для стандартного нормального."""
    return 0.5 * math.erfc(float(z) / math.sqrt(2.0))


def _gser(a: float, x: float) -> float:
    """Ряд для регуляризованої нижньої неповної гамми P(a, x)."""
    ap, s, d = a, 1.0 / a, 1.0 / a
    for _ in range(_ITMAX):
        ap += 1.0
        d *= x / ap
        s += d
        if abs(d) < abs(s) * _EPS:
            break
    return s * math.exp(-x + a * math.log(x) - math.lgamma(a))


def _gcf(a: float, x: float) -> float:
    """Неперервний дріб (Lentz) для верхньої неповної гамми Q(a, x)."""
    b = x + 1.0 - a
    c = 1.0 / _FPMIN
    d = 1.0 / b
    h = d
    for i in range(1, _ITMAX + 1):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = b + an / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < _EPS:
            break
    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def gamma_q(a: float, x: float) -> float:
    """Регуляризована верхня неповна гамма Q(a, x) = 1 - P(a, x)."""
    a, x = float(a), float(x)
    if x < 0 or a <= 0:
        raise ValueError("gamma_q: потрібні a > 0, x >= 0")
    if x == 0.0:
        return 1.0
    return 1.0 - _gser(a, x) if x < a + 1.0 else _gcf(a, x)


def chi2_sf(x: float, df: float) -> float:
    """P(χ²_df > x)."""
    if df <= 0:
        return float("nan")
    return gamma_q(df / 2.0, max(float(x), 0.0) / 2.0)


def _betacf(a: float, b: float, x: float) -> float:
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < _FPMIN:
        d = _FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, _ITMAX + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = 1.0 + aa / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = 1.0 + aa / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < _EPS:
            break
    return h


def beta_inc(a: float, b: float, x: float) -> float:
    """Регуляризована неповна бета I_x(a, b)."""
    a, b, x = float(a), float(b), float(x)
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    bt = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def t_sf2(t: float, df: float) -> float:
    """Двостороннє p для t-статистики з df ступенями свободи."""
    if df <= 0 or not np.isfinite(t):
        return float("nan")
    return beta_inc(df / 2.0, 0.5, df / (df + float(t) ** 2))


# --------------------------------------------------------------------------- #
# Пуассон
# --------------------------------------------------------------------------- #


def poisson_cdf(k: float, mu: float) -> float:
    """P(X <= k) для X ~ Poisson(mu). Через Q(k+1, mu)."""
    if mu <= 0:
        return 1.0
    k = math.floor(k)
    if k < 0:
        return 0.0
    return gamma_q(k + 1.0, mu)


def poisson_sf(k: float, mu: float) -> float:
    """P(X > k)."""
    return 1.0 - poisson_cdf(k, mu)


def poisson_interval(mu: float, alpha: float = 0.0027) -> tuple[int, int]:
    """Центральний інтервал Пуассона [lo, hi] — аналог ±3σ (alpha=0.0027)."""
    if mu <= 0:
        return 0, 0
    lo_t, hi_t = alpha / 2.0, 1.0 - alpha / 2.0
    lo, hi, c = 0, 0, 0.0
    k, pmf = 0, math.exp(-mu)
    found_lo = False
    while k < 10_000:
        c += pmf
        if not found_lo and c > lo_t:
            lo, found_lo = k, True
        if c >= hi_t:
            hi = k
            break
        k += 1
        pmf *= mu / k
    return lo, max(hi, lo)


def dispersion_test(counts: np.ndarray) -> tuple[float, float, float]:
    """Тест надмірної дисперсії Пуассона.

    Повертає (VMR, χ², p). Для Пуассона VMR ≈ 1 і (n-1)·VMR ~ χ²(n-1);
    VMR > 1 — події «злипаються» в серії, VMR < 1 — рівномірніше за випадковість.
    """
    x = np.asarray(counts, dtype=float)
    n = x.size
    m = float(x.mean()) if n else 0.0
    if n < 3 or m <= 0:
        return float("nan"), float("nan"), float("nan")
    vmr = float(x.var(ddof=1) / m)
    chi2 = vmr * (n - 1)
    return vmr, chi2, chi2_sf(chi2, n - 1)


# --------------------------------------------------------------------------- #
# Кореляції, тренд
# --------------------------------------------------------------------------- #


def spearman(x, y) -> tuple[float, float]:
    """(rho, двостороннє p) — p через t-наближення, df = n-2."""
    a = pd.Series(np.asarray(x, dtype=float)).rank()
    b = pd.Series(np.asarray(y, dtype=float)).rank()
    n = len(a)
    if n < 4 or a.std(ddof=1) == 0 or b.std(ddof=1) == 0:
        return float("nan"), float("nan")
    rho = float(np.corrcoef(a, b)[0, 1])
    if abs(rho) >= 1.0:
        return rho, 0.0
    t = rho * math.sqrt((n - 2) / (1.0 - rho * rho))
    return rho, t_sf2(t, n - 2)


@dataclass(frozen=True)
class Trend:
    slope: float        # одиниць на день
    intercept: float
    se: float
    t: float
    p: float
    r2: float
    ts_slope: float     # робастний нахил Тейла–Сена
    ts_intercept: float
    tau: float          # Кендаллове τ (Манн–Кендалл)
    mk_z: float
    mk_p: float
    n: int


_MAX_PAIRWISE = 1500  # межа для O(n²) статистик: 1500² ≈ 18 МБ на матрицю


def _subsample(y: np.ndarray, x: np.ndarray, max_n: int, seed: int = 0):
    if y.size <= max_n:
        return y, x
    idx = np.sort(np.random.default_rng(seed).choice(y.size, max_n, replace=False))
    return y[idx], x[idx]


def _mann_kendall(y: np.ndarray) -> tuple[float, float, float]:
    """S-тест Манна–Кендалла з поправкою на зв'язки. Повертає (tau_b, z, p)."""
    y, _ = _subsample(y, y, _MAX_PAIRWISE)
    n = y.size
    if n < 8:
        return float("nan"), float("nan"), float("nan")
    diff = np.sign(np.subtract.outer(y, y))      # y_j - y_i для j > i
    s = float(np.triu(diff, k=1).sum() * -1)     # outer дає y_i - y_j → знак навпаки
    _, cnt = np.unique(y, return_counts=True)
    ties = float(np.sum(cnt * (cnt - 1) * (2 * cnt + 5)))
    var = (n * (n - 1) * (2 * n + 5) - ties) / 18.0
    n0 = n * (n - 1) / 2.0
    n1 = float(np.sum(cnt * (cnt - 1) / 2.0))
    tau = s / math.sqrt((n0 - n1) * n0) if n0 > n1 else float("nan")
    if var <= 0:
        return tau, float("nan"), float("nan")
    z = (s - math.copysign(1.0, s)) / math.sqrt(var) if s != 0 else 0.0
    return tau, z, 2.0 * norm_sf(abs(z))


def theil_sen(y: np.ndarray, x: np.ndarray | None = None, max_n: int = _MAX_PAIRWISE,
              seed: int = 0) -> tuple[float, float]:
    """Медіанний нахил по всіх парах (робастний до викидів) + перетин."""
    y = np.asarray(y, dtype=float)
    x = np.arange(y.size, dtype=float) if x is None else np.asarray(x, dtype=float)
    if y.size < 3:
        return float("nan"), float("nan")
    y, x = _subsample(y, x, max_n, seed)  # без цього O(n²) з'їсть пам'ять
    dx = x[None, :] - x[:, None]
    dy = y[None, :] - y[:, None]
    m = np.triu(np.ones_like(dx, dtype=bool), k=1) & (dx != 0)
    slope = float(np.median(dy[m] / dx[m]))
    return slope, float(np.median(y - slope * x))


def ols_fit(y, x=None) -> tuple[float, float, float, float, float]:
    """Проста лінійна регресія: (slope, intercept, se, p, r²)."""
    y = np.asarray(y, dtype=float)
    x = np.arange(y.size, dtype=float) if x is None else np.asarray(x, dtype=float)
    n = y.size
    nan = float("nan")
    if n < 3:
        return nan, nan, nan, nan, nan
    xm, ym = x.mean(), y.mean()
    sxx = float(((x - xm) ** 2).sum())
    if sxx == 0:
        return nan, nan, nan, nan, nan
    slope = float(((x - xm) * (y - ym)).sum() / sxx)
    intercept = float(ym - slope * xm)
    sse = float(((y - (intercept + slope * x)) ** 2).sum())
    sst = float(((y - ym) ** 2).sum())
    se = math.sqrt(sse / (n - 2) / sxx) if n > 2 and sse > 0 else 0.0
    p = t_sf2(slope / se, n - 2) if se > 0 else nan
    r2 = 1.0 - sse / sst if sst > 0 else nan
    return slope, intercept, se, p, r2


def trend(y, x=None) -> Trend:
    """OLS + Тейл–Сен + Манн–Кендалл на одному наборі."""
    y = np.asarray(y, dtype=float)
    x = np.arange(y.size, dtype=float) if x is None else np.asarray(x, dtype=float)
    n = y.size
    nan = float("nan")
    slope, intercept, se, p, r2 = ols_fit(y, x)
    if n < 8 or not np.isfinite(slope):
        return Trend(slope, intercept, se, nan, p, r2, nan, nan, nan, nan, nan, n)
    t = slope / se if se and se > 0 else nan
    ts_s, ts_i = theil_sen(y, x)
    tau, z, mkp = _mann_kendall(y)
    return Trend(slope, intercept, se, t, p, r2, ts_s, ts_i, tau, z, mkp, n)


# --------------------------------------------------------------------------- #
# Часові ряди: ACF, спектр, точки зламу
# --------------------------------------------------------------------------- #


def acf(y, nlags: int) -> tuple[np.ndarray, float]:
    """Автокореляція лагів 1..nlags + межа 95% для білого шуму (±1.96/√n)."""
    y = np.asarray(y, dtype=float)
    n = y.size
    nlags = int(min(nlags, max(1, n // 3)))
    d = y - y.mean()
    denom = float((d * d).sum())
    if denom == 0:
        return np.zeros(nlags), 0.0
    out = np.array([float((d[k:] * d[:-k]).sum() / denom) for k in range(1, nlags + 1)])
    return out, 1.96 / math.sqrt(n)


def periodogram(y, min_period: float = 2.0, max_period: float = 120.0):
    """Періодограма (FFT) у координатах «період у днях».

    Повертає (periods, power, fisher_threshold_95). Порог — тест Фішера на
    найбільшу ординату: для білого шуму нормовані ординати ~Exp(1), тож
    поріг = -mean·ln(1 - 0.95^(1/m)) на m перевірених частотах.
    """
    y = np.asarray(y, dtype=float)
    n = y.size
    if n < 16:
        return np.array([]), np.array([]), float("nan")
    d = y - y.mean()
    power = np.abs(np.fft.rfft(d)) ** 2 / n
    freq = np.fft.rfftfreq(n, d=1.0)
    keep = freq > 0
    power, freq = power[keep], freq[keep]
    periods = 1.0 / freq
    sel = (periods >= min_period) & (periods <= min(max_period, n / 2.0))
    periods, power = periods[sel], power[sel]
    m = power.size
    if m == 0:
        return periods, power, float("nan")
    thr = -float(power.mean()) * math.log(1.0 - 0.95 ** (1.0 / m))
    return periods, power, thr


def _pois_seg_cost(csum: np.ndarray, i: int, j: int) -> float:
    """-2·max логправдоподібності Пуассона для counts[i:j] (без const log k!)."""
    s = csum[j] - csum[i]
    n = j - i
    if s <= 0:
        return 0.0
    return -2.0 * (s * math.log(s / n) - s)


def poisson_changepoints(counts, min_seg: int = 14, max_cp: int = 8,
                         penalty_mult: float = 3.0) -> list[int]:
    """Бінарна сегментація з пуассонівською правдоподібністю.

    Точка зламу приймається, поки приріст правдоподібності перевищує
    BIC-подібний штраф penalty_mult·ln(N). Повертає відсортовані індекси
    початків нових сегментів.
    """
    x = np.asarray(counts, dtype=float)
    n = x.size
    if n < 4 * min_seg:
        return []
    csum = np.concatenate(([0.0], np.cumsum(x)))
    penalty = penalty_mult * math.log(n)
    cps: list[int] = []
    segs = [(0, n)]
    while len(cps) < max_cp:
        best = (0.0, None, None)
        for (a, b) in segs:
            if b - a < 2 * min_seg:
                continue
            parent = _pois_seg_cost(csum, a, b)
            ks = np.arange(a + min_seg, b - min_seg + 1)
            if ks.size == 0:
                continue
            s_all = csum[b] - csum[a]
            sl = csum[ks] - csum[a]
            sr = s_all - sl
            nl = (ks - a).astype(float)
            nr = (b - ks).astype(float)
            with np.errstate(divide="ignore", invalid="ignore"):
                ll = np.where(sl > 0, sl * np.log(np.maximum(sl, 1e-12) / nl) - sl, 0.0)
                lr = np.where(sr > 0, sr * np.log(np.maximum(sr, 1e-12) / nr) - sr, 0.0)
            child = -2.0 * (ll + lr)
            gain = parent - child
            i = int(np.argmax(gain))
            if gain[i] > best[0]:
                best = (float(gain[i]), int(ks[i]), (a, b))
        if best[1] is None or best[0] <= penalty:
            break
        k, (a, b) = best[1], best[2]
        cps.append(k)
        segs.remove((a, b))
        segs += [(a, k), (k, b)]
    return sorted(cps)


# --------------------------------------------------------------------------- #
# Концентрація і різноманіття
# --------------------------------------------------------------------------- #


def gini(x) -> float:
    """Коефіцієнт Джині: 0 — рівномірно по днях, 1 — усе в одному дні."""
    a = np.sort(np.asarray(x, dtype=float))
    n, s = a.size, a.sum()
    if n == 0 or s <= 0:
        return float("nan")
    idx = np.arange(1, n + 1)
    return float((2.0 * (idx * a).sum() / (n * s)) - (n + 1.0) / n)


def lorenz(x) -> tuple[np.ndarray, np.ndarray]:
    """Крива Лоренца: (частка днів, частка подій), обидві від 0 до 1."""
    a = np.sort(np.asarray(x, dtype=float))
    n = a.size
    if n == 0 or a.sum() <= 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    cum = np.concatenate(([0.0], np.cumsum(a) / a.sum()))
    return np.arange(n + 1) / n, cum


def entropy_bits(counts) -> tuple[float, float]:
    """(Шеннонова ентропія в бітах, нормована 0..1) для розподілу частот."""
    p = np.asarray(counts, dtype=float)
    p = p[p > 0]
    tot = p.sum()
    k = np.asarray(counts, dtype=float).size
    if tot <= 0 or k < 2:
        return float("nan"), float("nan")
    p = p / tot
    h = float(-(p * np.log2(p)).sum())
    return h, h / math.log2(k)


def chi2_gof(observed) -> tuple[float, int, float, np.ndarray]:
    """χ² на рівномірність + стандартизовані пірсонівські залишки.

    Повертає (χ², df, p, residuals). Залишок > |2| — клітинка, що «ламає» модель.
    """
    o = np.asarray(observed, dtype=float)
    k = o.size
    tot = o.sum()
    if k < 2 or tot <= 0:
        return float("nan"), 0, float("nan"), np.zeros(k)
    e = np.full(k, tot / k)
    chi2 = float(((o - e) ** 2 / e).sum())
    resid = (o - e) / np.sqrt(e * (1.0 - 1.0 / k))
    return chi2, k - 1, chi2_sf(chi2, k - 1), resid


def chi2_contingency_2x2(n00: int, n01: int, n10: int, n11: int) -> tuple[float, float]:
    """χ² незалежності для 2×2 (без поправки Йейтса). Повертає (χ², p)."""
    obs = np.array([[n00, n01], [n10, n11]], dtype=float)
    tot = obs.sum()
    if tot <= 0 or (obs.sum(axis=0) == 0).any() or (obs.sum(axis=1) == 0).any():
        return float("nan"), float("nan")
    exp = np.outer(obs.sum(axis=1), obs.sum(axis=0)) / tot
    chi2 = float(((obs - exp) ** 2 / exp).sum())
    return chi2, chi2_sf(chi2, 1)


# --------------------------------------------------------------------------- #
# Кругова статистика (година доби — циклічна величина)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CircStats:
    mean_hour: float     # круговий середній час доби
    R: float             # довжина результуючого вектора, 0..1 (концентрація)
    sd_hours: float      # кругове стандартне відхилення, години
    rayleigh_p: float    # p тесту Релея: H0 — години рівномірні по добі
    n: int


def circ_stats(hours) -> CircStats:
    h = np.asarray(hours, dtype=float)
    n = h.size
    if n < 3:
        return CircStats(float("nan"), float("nan"), float("nan"), float("nan"), n)
    th = 2.0 * math.pi * h / 24.0
    c, s = float(np.cos(th).mean()), float(np.sin(th).mean())
    R = math.hypot(c, s)
    mean_h = (math.atan2(s, c) / (2.0 * math.pi) * 24.0) % 24.0
    sd = math.sqrt(-2.0 * math.log(R)) * 24.0 / (2.0 * math.pi) if R > 0 else float("inf")
    z = n * R * R
    # наближення Цара для p тесту Релея
    p = math.exp(-z) * (1.0 + (2.0 * z - z * z) / (4.0 * n)
                        - (24.0 * z - 132.0 * z**2 + 76.0 * z**3 - 9.0 * z**4) / (288.0 * n * n))
    return CircStats(mean_h, R, sd, min(max(p, 0.0), 1.0), n)


def circ_mean_hour(hours) -> float:
    h = np.asarray(hours, dtype=float)
    if h.size == 0:
        return float("nan")
    th = 2.0 * math.pi * h / 24.0
    return (math.atan2(float(np.sin(th).mean()), float(np.cos(th).mean()))
            / (2.0 * math.pi) * 24.0) % 24.0


def unwrap_hours(values) -> np.ndarray:
    """Розгортає послідовність годин доби, щоб 23:30 → 00:30 не давало -23 год."""
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return v
    return np.unwrap(v / 24.0 * 2.0 * math.pi) / (2.0 * math.pi) * 24.0


# --------------------------------------------------------------------------- #
# Процес відновлення: виживання, інтенсивність, «пам'ять»
# --------------------------------------------------------------------------- #


def survival(gaps) -> tuple[np.ndarray, np.ndarray]:
    """Емпірична функція виживання S(t) = P(інтервал > t)."""
    g = np.sort(np.asarray(gaps, dtype=float))
    n = g.size
    if n == 0:
        return np.array([]), np.array([])
    s = 1.0 - np.arange(1, n + 1) / n
    return g, s


def cond_prob_within(gaps, elapsed: float, horizon: float) -> float:
    """P(подія протягом наступних `horizon` | вже минуло `elapsed`).

    Емпірично, без припущення про розподіл: (S(t) - S(t+h)) / S(t).
    """
    g = np.asarray(gaps, dtype=float)
    if g.size < 5:
        return float("nan")
    s_t = float((g > elapsed).mean())
    if s_t <= 0:
        return float("nan")
    s_th = float((g > elapsed + horizon).mean())
    return (s_t - s_th) / s_t


def hazard_curve(gaps, horizon: float, grid: np.ndarray) -> np.ndarray:
    """P(подія в наступні `horizon` | минуло t) для кожного t із grid."""
    g = np.asarray(gaps, dtype=float)
    out = np.full(grid.size, np.nan)
    for i, t in enumerate(grid):
        at_risk = g > t
        k = int(at_risk.sum())
        if k >= 5:  # менше 5 спостережень у ризику — шум, не показуємо
            out[i] = float(((g > t) & (g <= t + horizon)).sum()) / k
    return out


def weibull_fit(gaps) -> tuple[float, float, float]:
    """Оцінка Вейбулла за Weibull-plot: ln(-ln S) = k·ln t - k·ln λ.

    Повертає (k, lambda, r²). k < 1 — «чим довша пауза, тим менша інтенсивність»
    (важкий хвіст, серійність); k = 1 — експоненційний (пуассонівський) процес;
    k > 1 — інтервали регулярніші за випадкові.
    """
    t, s = survival(gaps)
    m = (t > 0) & (s > 0)
    t, s = t[m], s[m]
    if t.size < 10:
        return float("nan"), float("nan"), float("nan")
    if t.size > 400:  # регресія по прорідженій сітці: рівні ваги, менше роботи
        sel = np.unique(np.linspace(0, t.size - 1, 400).astype(int))
        t, s = t[sel], s[sel]
    x, y = np.log(t), np.log(-np.log(s))
    k, intercept, _, _, r2 = ols_fit(y, x)
    lam = math.exp(-intercept / k) if k and np.isfinite(k) and k != 0 else float("nan")
    return float(k), float(lam), float(r2)


def burstiness(gaps) -> tuple[float, float]:
    """Площина Го–Барабаші: (burstiness B, memory M).

    B = (σ-μ)/(σ+μ) ∈ [-1, 1]: -1 — ідеально періодично, 0 — пуассонівсько,
    →1 — серіями. M — кореляція сусідніх інтервалів: >0 означає, що за довгою
    паузою частіше йде довга.
    """
    g = np.asarray(gaps, dtype=float)
    g = g[np.isfinite(g)]
    if g.size < 5:
        return float("nan"), float("nan")
    mu, sd = float(g.mean()), float(g.std(ddof=1))
    b = (sd - mu) / (sd + mu) if (sd + mu) > 0 else float("nan")
    prev, nxt = g[:-1], g[1:]
    m = (float(np.corrcoef(prev, nxt)[0, 1])
         if g.size >= 6 and prev.std() > 0 and nxt.std() > 0 else float("nan"))
    return b, m


def length_biased_mean(gaps) -> float:
    """«Парадокс часу очікування»: середній інтервал, у який попадає випадковий
    момент, = E[X²]/E[X] ≥ E[X]."""
    g = np.asarray(gaps, dtype=float)
    if g.size == 0 or g.mean() <= 0:
        return float("nan")
    return float((g ** 2).mean() / g.mean())


# --------------------------------------------------------------------------- #
# Ланцюг Маркова «активний день / пропуск»
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Markov:
    p01: float      # P(активний завтра | сьогодні пропуск)
    p11: float      # P(активний завтра | сьогодні активний)
    persistence: float   # p11 - p01: 0 — незалежні дні, >0 — інерція
    stationary: float    # стаціонарна частка активних днів
    chi2: float
    p: float
    n: int


def markov_daily(active) -> Markov:
    a = np.asarray(active).astype(int)
    if a.size < 10:
        nan = float("nan")
        return Markov(nan, nan, nan, nan, nan, nan, a.size)
    prev, nxt = a[:-1], a[1:]
    n00 = int(((prev == 0) & (nxt == 0)).sum())
    n01 = int(((prev == 0) & (nxt == 1)).sum())
    n10 = int(((prev == 1) & (nxt == 0)).sum())
    n11 = int(((prev == 1) & (nxt == 1)).sum())
    p01 = n01 / (n00 + n01) if (n00 + n01) else float("nan")
    p11 = n11 / (n10 + n11) if (n10 + n11) else float("nan")
    chi2, p = chi2_contingency_2x2(n00, n01, n10, n11)
    stat = p01 / (1.0 + p01 - p11) if np.isfinite(p01) and np.isfinite(p11) and (1.0 + p01 - p11) > 0 else float("nan")
    return Markov(p01, p11, p11 - p01, stat, chi2, p, a.size)


def expected_days_to_streak(p01: float, p11: float, length: int) -> float:
    """Очікувана кількість днів до першого стріку заданої довжини.

    Поглинаючий ланцюг: стан = поточна довжина стріку (0..L-1), поглинання в L.
    Розв'язується як система E = 1 + P·E.
    """
    L = int(length)
    if L <= 0:
        return 0.0
    if not (np.isfinite(p01) and np.isfinite(p11)) or p01 <= 0 or p11 <= 0:
        return float("inf")
    if L > 2000:
        return float("inf")
    A = np.eye(L)
    b = np.ones(L)
    A[0, 0] -= 1.0 - p01                      # зі стану 0: пропуск → знову 0
    if L > 1:
        A[0, 1] -= p01                        # активний день → стрік 1
        for s in range(1, L):
            A[s, 0] -= 1.0 - p11              # обрив → 0
            if s + 1 < L:
                A[s, s + 1] -= p11            # продовження
    try:
        e = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return float("inf")
    return float(e[0])


def prob_streak_within(p01: float, p11: float, length: int, horizon: int,
                       start_streak: int = 0, n_sims: int = 4000, seed: int = 0) -> float:
    """P(досягти стріку `length` протягом `horizon` днів) — Монте-Карло по ланцюгу."""
    if not (np.isfinite(p01) and np.isfinite(p11)):
        return float("nan")
    if start_streak >= length:
        return 1.0
    rng = np.random.default_rng(seed)
    run = np.full(n_sims, int(start_streak))
    done = np.zeros(n_sims, dtype=bool)
    for _ in range(int(horizon)):
        p = np.where(run > 0, p11, p01)
        act = rng.random(n_sims) < p
        run = np.where(act, run + 1, 0)
        done |= run >= length
        if done.all():
            break
    return float(done.mean())


# --------------------------------------------------------------------------- #
# Прогноз: пуассон/негативно-біноміальна модель із тижневою сезонністю
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RateModel:
    mu: float                 # базова інтенсивність, подій/день
    wd_factor: np.ndarray     # 7 множників (Пн..Нд), згладжені до 1
    vmr: float                # індекс дисперсії після зняття сезонності
    window: int


def fit_rate_model(daily_counts: np.ndarray, weekdays: np.ndarray,
                   window: int = 120, shrink: float = 8.0) -> RateModel:
    """Інтенсивність за останні `window` днів + тижневі множники.

    Множники стягуються до 1 за Джеймсом–Стайном-подібним правилом
    (n_d·m_d + shrink·μ)/(n_d + shrink)/μ — інакше на 4 спостереженнях
    «неділя» отримує випадковий множник 2.0.
    """
    y = np.asarray(daily_counts, dtype=float)
    wd = np.asarray(weekdays, dtype=int)
    if y.size == 0:
        return RateModel(0.0, np.ones(7), 1.0, 0)
    w = int(min(window, y.size))
    y, wd = y[-w:], wd[-w:]
    mu = float(y.mean())
    if mu <= 0:
        return RateModel(0.0, np.ones(7), 1.0, w)
    fac = np.ones(7)
    for d in range(7):
        m = wd == d
        nd = int(m.sum())
        if nd:
            fac[d] = ((nd * float(y[m].mean()) + shrink * mu) / (nd + shrink)) / mu
    fac /= fac.mean()  # нормуємо середній множник до 1
    lam = mu * fac[wd]
    resid_vmr = float(((y - lam) ** 2 / np.maximum(lam, 1e-9)).mean())
    return RateModel(mu, fac, max(resid_vmr, 1.0), w)


def simulate_future(model: RateModel, weekdays_future: np.ndarray, n_sims: int = 4000,
                    seed: int = 0) -> np.ndarray:
    """Матриця (n_sims × days) майбутніх лічильників.

    Понад-дисперсія (VMR > 1) моделюється сумішшю гамма–Пуассон
    (негативно-біноміальний розподіл) з тим самим VMR.
    """
    rng = np.random.default_rng(seed)
    wd = np.asarray(weekdays_future, dtype=int)
    lam = model.mu * model.wd_factor[wd]
    lam = np.tile(lam, (n_sims, 1))
    if model.vmr > 1.05:
        shape = lam / (model.vmr - 1.0)          # NB: Var = λ·VMR
        lam = rng.gamma(shape=np.maximum(shape, 1e-6), scale=(model.vmr - 1.0))
    return rng.poisson(lam)


def quantile_fan(sims_cum: np.ndarray, qs=(0.05, 0.25, 0.5, 0.75, 0.95)) -> dict:
    return {q: np.quantile(sims_cum, q, axis=0) for q in qs}


def mean_ci(x, level: float = 0.95) -> tuple[float, float]:
    """(середнє, напівширина CI) через SEM. Робастно до понад-дисперсії:
    sd вимірюється з даних, а не береться з пуассонівського припущення."""
    a = np.asarray(x, dtype=float)
    n = a.size
    if n < 2:
        return (float(a.mean()) if n else float("nan")), float("nan")
    # квантиль нормального для рівня (двостороннє); 1.959964 для 95%
    z = math.sqrt(2.0) * _erfinv(level)
    return float(a.mean()), float(z * a.std(ddof=1) / math.sqrt(n))


def _erfinv(p: float) -> float:
    """erf⁻¹(p) методом Ньютона — потрібен лише для квантиля нормального."""
    x = 0.0
    for _ in range(60):
        err = math.erf(x) - p
        d = 2.0 / math.sqrt(math.pi) * math.exp(-x * x)
        if d < 1e-300:
            break
        step = err / d
        x -= step
        if abs(step) < 1e-14:
            break
    return x


def eta_days(sims_cum: np.ndarray, need: float, qs=(0.05, 0.5, 0.95)) -> tuple[float, dict]:
    """Коли накопичиться `need` подій: (частка симуляцій, що дійшли, квантилі днів).

    Квантилі рахуються лише по тих симуляціях, що дійшли до цілі в межах
    горизонту — тому разом із ними завжди треба показувати частку.
    """
    if need <= 0:
        return 1.0, {q: 0.0 for q in qs}
    reached = sims_cum[:, -1] >= need
    frac = float(reached.mean())
    if frac == 0:
        return 0.0, {q: float("nan") for q in qs}
    first = np.argmax(sims_cum[reached] >= need, axis=1) + 1.0
    return frac, {q: float(np.quantile(first, q)) for q in qs}
