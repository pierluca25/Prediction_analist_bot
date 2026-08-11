"""
Engine di analisi per Prediction_Bot.

Fonti (tutte gratuite, nessuna API key, nessuno scraping):
  fixtures.csv          partite in programma con quote di molti bookmaker
  {stagione}/{div}.csv  risultati storici per stimare il modello

Riferimento di mercato: BETFAIR EXCHANGE (colonne BFE*).
Non Pinnacle: dal 23/07/2025 le sue quote su football-data sono
sistematicamente stantie e non vanno piu' usate come linea sharp.
L'exchange e' comunque un riferimento migliore, perche' e' un prezzo di
mercato vero con solo la commissione, non il margine di un bookmaker.
"""

from __future__ import annotations

import io
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import requests
from scipy.optimize import brentq, minimize
from scipy.special import gammaln
from scipy.stats import poisson

BASE = "https://www.football-data.co.uk"
FIXTURES_URL = f"{BASE}/fixtures.csv"
UA = {"User-Agent": "Mozilla/5.0 (compatible; PredictionBot/1.0)"}

# campionati seguiti
LEGHE = {"I1": "Serie A", "E0": "Premier League"}
# stagioni storiche da scaricare per stimare il modello
STAGIONI = ["2223", "2324", "2425", "2526", "2627"]

MAX_GOALS = 10
CACHE_TTL = 6 * 3600  # 6 ore


# ---------------------------------------------------------------- devigging
def devig_power(quote):
    """Rimuove il margine col metodo power. Ritorna probabilita' che sommano a 1."""
    q = np.array([1.0 / o for o in quote], dtype=float)
    s = q.sum()
    if abs(s - 1.0) < 1e-9:
        return q

    def f(k):
        return float(np.sum(q**k) - 1.0)

    try:
        k = brentq(f, 0.2, 3.0, xtol=1e-10, maxiter=200)
    except (ValueError, RuntimeError):
        return q / s
    p = q**k
    return p / p.sum()


def overround(quote):
    return float(sum(1.0 / o for o in quote) - 1.0)


def ev(p, quota):
    """Valore atteso per unita' di puntata."""
    return p * quota - 1.0


# ---------------------------------------------------------------- Dixon-Coles
@dataclass
class Modello:
    squadre: list[str]
    attacco: np.ndarray
    difesa: np.ndarray
    gamma: float
    rho: float
    n_partite: int

    def _i(self, s):
        try:
            return self.squadre.index(s)
        except ValueError:
            return None

    def mercati(self, casa, trasferta):
        """Probabilita' dei mercati. None se una squadra non e' nel modello."""
        i, j = self._i(casa), self._i(trasferta)
        if i is None or j is None:
            return None
        lam = float(np.exp(self.attacco[i] - self.difesa[j] + self.gamma))
        mu = float(np.exp(self.attacco[j] - self.difesa[i]))

        k = np.arange(MAX_GOALS + 1)
        m = np.outer(poisson.pmf(k, lam), poisson.pmf(k, mu))
        m[0, 0] *= 1.0 - lam * mu * self.rho
        m[0, 1] *= 1.0 + lam * self.rho
        m[1, 0] *= 1.0 + mu * self.rho
        m[1, 1] *= 1.0 - self.rho
        m = np.clip(m, 1e-15, None)
        m /= m.sum()

        idx = np.arange(MAX_GOALS + 1)
        h, a = np.meshgrid(idx, idx, indexing="ij")
        tot = h + a
        pH = float(m[h > a].sum())
        pD = float(np.trace(m))
        pA = float(m[h < a].sum())
        return {
            "1": pH, "X": pD, "2": pA,
            "O25": float(m[tot > 2].sum()),
            "U25": float(m[tot <= 2].sum()),
            "GG": float(m[1:, 1:].sum()),
            "1X": pH + pD, "X2": pD + pA, "12": pH + pA,
            "gol_attesi": lam + mu,
        }


def stima_modello(df, xi=0.0065, min_partite=4):
    """Massima verosimiglianza pesata con time decay."""
    d = df[["date", "home", "away", "hg", "ag"]].dropna().copy()
    ref = d["date"].max()

    cnt = pd.concat([d["home"], d["away"]]).value_counts()
    ok = set(cnt[cnt >= min_partite].index)
    d = d[d["home"].isin(ok) & d["away"].isin(ok)]

    squadre = sorted(set(d["home"]) | set(d["away"]))
    idx = {t: i for i, t in enumerate(squadre)}
    n = len(squadre)
    if n < 6 or len(d) < 100:
        raise ValueError(f"dati insufficienti: {len(d)} partite, {n} squadre")

    hi = d["home"].map(idx).to_numpy()
    ai = d["away"].map(idx).to_numpy()
    hg = d["hg"].to_numpy(dtype=int)
    ag = d["ag"].to_numpy(dtype=int)
    w = np.exp(-xi * np.clip((ref - d["date"]).dt.days.to_numpy(float), 0, None))

    m00 = (hg == 0) & (ag == 0)
    m01 = (hg == 0) & (ag == 1)
    m10 = (hg == 1) & (ag == 0)
    m11 = (hg == 1) & (ag == 1)

    def unpack(p):
        af = p[: n - 1]
        return np.concatenate([af, [-af.sum()]]), p[n - 1 : 2 * n - 1], p[-2], p[-1]

    def nll(p):
        att, dif, gamma, rho = unpack(p)
        ll_lam = att[hi] - dif[ai] + gamma
        ll_mu = att[ai] - dif[hi]
        np.clip(ll_lam, -18, 3.4, out=ll_lam)
        np.clip(ll_mu, -18, 3.4, out=ll_mu)
        lam, mu = np.exp(ll_lam), np.exp(ll_mu)
        ll = hg * ll_lam - lam + ag * ll_mu - mu
        tau = np.ones_like(lam)
        tau[m00] = 1.0 - lam[m00] * mu[m00] * rho
        tau[m01] = 1.0 + lam[m01] * rho
        tau[m10] = 1.0 + mu[m10] * rho
        tau[m11] = 1.0 - rho
        if np.any(tau <= 1e-10):
            return 1e10
        return -float(np.dot(w, ll + np.log(tau)))

    x0 = np.concatenate([np.zeros(n - 1), np.zeros(n), [0.2], [-0.05]])
    bounds = [(-3, 3)] * (2 * n - 1) + [(-1.0, 1.5), (-0.35, 0.35)]
    res = minimize(nll, x0, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": 3000, "maxfun": 60000})
    att, dif, gamma, rho = unpack(res.x)
    return Modello(squadre, att, dif, float(gamma), float(rho), len(d))


# ---------------------------------------------------------------- download
def _scarica(url, tentativi=3):
    ultimo = None
    for i in range(tentativi):
        try:
            r = requests.get(url, headers=UA, timeout=30)
            r.raise_for_status()
            return r.content
        except Exception as e:  # rete instabile: si riprova
            ultimo = e
            time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"download fallito {url}: {ultimo}")


def _leggi_csv(raw):
    return pd.read_csv(io.BytesIO(raw), encoding="latin-1", on_bad_lines="skip")


def _date(serie):
    out = pd.to_datetime(serie, format="%d/%m/%Y", errors="coerce")
    manca = out.isna()
    if manca.any():
        out.loc[manca] = pd.to_datetime(serie[manca], format="%d/%m/%y", errors="coerce")
    manca = out.isna()
    if manca.any():
        out.loc[manca] = pd.to_datetime(serie[manca], dayfirst=True, errors="coerce")
    return out


def carica_storico(divisioni=tuple(LEGHE), stagioni=tuple(STAGIONI)):
    """Scarica i risultati storici. Le stagioni non ancora esistenti si saltano."""
    frames, caricate = [], []
    for div in divisioni:
        for st in stagioni:
            try:
                raw = _scarica(f"{BASE}/mmz4281/{st}/{div}.csv", tentativi=2)
                d = _leggi_csv(raw)
                if not {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"} <= set(d.columns):
                    continue
                frames.append(pd.DataFrame({
                    "date": _date(d["Date"]),
                    "home": d["HomeTeam"].astype(str).str.strip(),
                    "away": d["AwayTeam"].astype(str).str.strip(),
                    "hg": pd.to_numeric(d["FTHG"], errors="coerce"),
                    "ag": pd.to_numeric(d["FTAG"], errors="coerce"),
                    "div": div,
                }))
                caricate.append(f"{div}/{st}")
            except Exception:
                continue  # stagione futura o file assente: normale

    if not frames:
        raise RuntimeError("nessuno storico scaricato")
    df = pd.concat(frames, ignore_index=True).dropna(subset=["date", "hg", "ag"])
    df = df.drop_duplicates(subset=["date", "home", "away"]).sort_values("date")
    return df.reset_index(drop=True), caricate


# colonne quote in fixtures.csv, in ordine di preferenza
RIF_1X2 = [("BFEH", "BFED", "BFEA"), ("PSH", "PSD", "PSA")]
MERC_1X2 = [("MaxH", "MaxD", "MaxA"), ("AvgH", "AvgD", "AvgA"), ("B365H", "B365D", "B365A")]
RIF_OU = [("BFE>2.5", "BFE<2.5"), ("P>2.5", "P<2.5")]
MERC_OU = [("Max>2.5", "Max<2.5"), ("Avg>2.5", "Avg<2.5"), ("B365>2.5", "B365<2.5")]


def _prima_valida(riga, gruppi):
    for cols in gruppi:
        if all(c in riga.index for c in cols):
            v = [pd.to_numeric(riga.get(c), errors="coerce") for c in cols]
            if all(pd.notna(x) and x > 1.0 for x in v):
                return [float(x) for x in v], cols[0]
    return None, None


def carica_partite(divisioni=tuple(LEGHE)):
    """Scarica le partite in programma con le relative quote."""
    d = _leggi_csv(_scarica(FIXTURES_URL))
    if "Div" not in d.columns:
        raise RuntimeError("fixtures.csv senza colonna Div")
    d = d[d["Div"].isin(divisioni)].copy()
    if d.empty:
        return []

    d["dt"] = _date(d["Date"])
    d = d.sort_values(["dt", "Time"] if "Time" in d.columns else ["dt"])

    out = []
    for _, r in d.iterrows():
        rif, _ = _prima_valida(r, RIF_1X2)
        mercato, fonte = _prima_valida(r, MERC_1X2)
        rif_ou, _ = _prima_valida(r, RIF_OU)
        merc_ou, _ = _prima_valida(r, MERC_OU)
        out.append({
            "div": r["Div"],
            "lega": LEGHE.get(r["Div"], r["Div"]),
            "data": r["dt"],
            "ora": str(r.get("Time", "")),
            "casa": str(r["HomeTeam"]).strip(),
            "trasferta": str(r["AwayTeam"]).strip(),
            "rif_1x2": rif,          # Betfair Exchange
            "mkt_1x2": mercato,      # miglior quota del panel
            "fonte_mkt": fonte,
            "rif_ou": rif_ou,
            "mkt_ou": merc_ou,
        })
    return out


# ---------------------------------------------------------------- analisi
@dataclass
class Analisi:
    aggiornato: float = 0.0
    modello: Modello | None = None
    partite: list = field(default_factory=list)
    storico_info: list = field(default_factory=list)
    errore: str | None = None

    def scaduta(self):
        return time.time() - self.aggiornato > CACHE_TTL


_cache = Analisi()


def aggiorna(forza=False) -> Analisi:
    """Scarica tutto e stima il modello. Usa la cache se ancora fresca."""
    global _cache
    if not forza and _cache.modello is not None and not _cache.scaduta():
        return _cache

    nuova = Analisi(aggiornato=time.time())
    try:
        storico, info = carica_storico()
        nuova.storico_info = info
        nuova.modello = stima_modello(storico)
        nuova.partite = valuta_partite(carica_partite(), nuova.modello)
    except Exception as e:
        nuova.errore = str(e)
        if _cache.modello is not None:
            return _cache  # meglio dati vecchi che nessun dato
    _cache = nuova
    return nuova


def valuta_partite(partite, modello):
    """Per ogni partita calcola probabilita' modello, mercato e disaccordo."""
    fuori = []
    for p in partite:
        mk = modello.mercati(p["casa"], p["trasferta"])
        p["modello"] = mk

        if p["rif_1x2"]:
            pr = devig_power(p["rif_1x2"])
            p["p_mercato"] = {"1": pr[0], "X": pr[1], "2": pr[2]}
            p["overround_rif"] = overround(p["rif_1x2"])
        else:
            p["p_mercato"] = None

        if p["rif_ou"]:
            po = devig_power(p["rif_ou"])
            p["p_mercato_ou"] = {"O25": po[0], "U25": po[1]}
        else:
            p["p_mercato_ou"] = None

        # disaccordo massimo fra modello e mercato sui tre esiti
        if mk and p["p_mercato"]:
            p["scarto"] = max(abs(mk[e] - p["p_mercato"][e]) for e in ("1", "X", "2"))
            p["esito_scarto"] = max(("1", "X", "2"), key=lambda e: abs(mk[e] - p["p_mercato"][e]))
        else:
            p["scarto"] = None
            p["esito_scarto"] = None

        if mk is None:
            fuori.append(f"{p['casa']}-{p['trasferta']}")

    partite.sort(key=lambda x: (x["data"] is pd.NaT, x["data"], x["casa"]))
    for i, p in enumerate(partite, 1):
        p["n"] = i
    return partite


def analizza_quota(partita, esito, quota_utente):
    """Confronta una quota inserita a mano con la linea di riferimento.

    Ritorna il quadro completo: probabilita' di mercato, del modello, EV,
    quota equa. Se manca il riferimento lo dice invece di inventarlo.
    """
    pm = partita.get("p_mercato")
    mk = partita.get("modello")
    out = {"esito": esito, "quota": quota_utente}

    if pm and esito in pm:
        out["p_mercato"] = pm[esito]
        out["quota_equa"] = 1.0 / pm[esito]
        out["ev_mercato"] = ev(pm[esito], quota_utente)
    elif partita.get("p_mercato_ou") and esito in partita["p_mercato_ou"]:
        pv = partita["p_mercato_ou"][esito]
        out["p_mercato"] = pv
        out["quota_equa"] = 1.0 / pv
        out["ev_mercato"] = ev(pv, quota_utente)
    else:
        out["p_mercato"] = None

    if mk and esito in mk:
        out["p_modello"] = mk[esito]
        out["ev_modello"] = ev(mk[esito], quota_utente)
    else:
        out["p_modello"] = None

    return out


def scalata(prob_step, quote_step, capitale):
    """Calcola una scalata a piu' step. Nessuna magia: probabilita' composte."""
    p_tot = float(np.prod(prob_step))
    moltiplicatore = float(np.prod(quote_step))
    finale = capitale * moltiplicatore
    return {
        "step": len(quote_step),
        "p_completamento": p_tot,
        "moltiplicatore": moltiplicatore,
        "capitale": capitale,
        "target": finale,
        "ev": p_tot * finale - capitale,
        "ev_pct": (p_tot * moltiplicatore) - 1.0,
    }
