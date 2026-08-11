"""
Prediction_Bot — interfaccia Telegram.

Filosofia: il bot fa il lavoro lungo (scaricare, stimare, confrontare),
l'utente fa il lavoro corto (mandare una riga di quote).

Comandi:
  /oggi              partite in programma, gia' analizzate
  /stato             bankroll e storico
  /reset             azzera il conto

Inserimento quote in UN SOLO messaggio:
  3 2.45 3.20 1.80   -> partita 3, quote Sisal per 1, X, 2
  3 1 2.45           -> partita 3, solo l'esito 1 a quota 2.45
  3 O25 1.85         -> partita 3, Over 2.5 a quota 1.85

Il riferimento e' Betfair Exchange, non il modello: il backtest su 8
stagioni di Serie A ha mostrato che il modello NON batte il mercato.
Il modello resta visibile come secondo parere, con peso zero sulle decisioni.
"""

from __future__ import annotations

import logging
import os
import re
import sqlite3
from datetime import datetime

from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

import engine as E

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("prediction_bot")

TOKEN = os.getenv("TELEGRAM_TOKEN", "")
DB_PATH = os.getenv("DATABASE_PATH", "/tmp/prediction_bot.db")
CAPITALE_DEFAULT = float(os.getenv("CAPITALE", "300"))

ESITI_1X2 = {"1", "X", "2"}
ESITI_ALTRI = {"O25", "U25", "GG", "1X", "X2", "12"}


# ------------------------------------------------------------------ database
def db():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c


def init_db():
    with db() as c:
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS utenti (
                uid INTEGER PRIMARY KEY,
                capitale REAL,
                creato TEXT
            );
            CREATE TABLE IF NOT EXISTS giocate (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                uid INTEGER, data TEXT, lega TEXT, partita TEXT,
                esito TEXT, quota REAL, quota_equa REAL,
                p_mercato REAL, p_modello REAL, ev REAL,
                stake REAL, stato TEXT DEFAULT 'aperta', pnl REAL DEFAULT 0
            );
            """
        )


def utente(uid):
    with db() as c:
        r = c.execute("SELECT * FROM utenti WHERE uid=?", (uid,)).fetchone()
        if r:
            return dict(r)
        c.execute(
            "INSERT INTO utenti (uid, capitale, creato) VALUES (?,?,?)",
            (uid, CAPITALE_DEFAULT, datetime.now().isoformat()),
        )
        return {"uid": uid, "capitale": CAPITALE_DEFAULT}


# ------------------------------------------------------------------ formato
def pct(x):
    return "n/d" if x is None else f"{x*100:.1f}%"


def riga_partita(p):
    quando = p["data"].strftime("%d/%m") if p["data"] is not None else "?"
    ora = f" {p['ora']}" if p["ora"] and p["ora"] != "nan" else ""
    testa = f"*{p['n']}.* {p['casa']} – {p['trasferta']}\n    {quando}{ora} · {p['lega']}"

    if p["p_mercato"]:
        m = p["p_mercato"]
        testa += f"\n    mercato  1 {pct(m['1'])} · X {pct(m['X'])} · 2 {pct(m['2'])}"
    else:
        testa += "\n    mercato  quote non ancora disponibili"

    if p["modello"]:
        mo = p["modello"]
        testa += f"\n    modello  1 {pct(mo['1'])} · X {pct(mo['X'])} · 2 {pct(mo['2'])}"
        if p["scarto"] is not None and p["scarto"] > 0.05:
            testa += f"\n    ⚠ disaccordo {p['scarto']*100:.0f} punti sull'esito {p['esito_scarto']}"
    else:
        testa += "\n    modello  squadra senza storico sufficiente"
    return testa


# ------------------------------------------------------------------ comandi
async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    u = utente(update.effective_user.id)
    await update.message.reply_text(
        "*Prediction\\_Bot*\n\n"
        "Il bot scarica le partite di Serie A e Premier League, le analizza "
        "e le confronta con Betfair Exchange. Tu mandi le quote di Sisal, "
        "lui ti dice se c'è valore.\n\n"
        f"Capitale: *€{u['capitale']:.0f}*\n\n"
        "*Comandi*\n"
        "/oggi — partite analizzate\n"
        "/stato — bankroll e storico\n"
        "/reset — azzera il conto\n\n"
        "*Per inserire una quota* basta un messaggio:\n"
        "`3 2.45 3.20 1.80` → partita 3, quote 1/X/2\n"
        "`3 1 2.45` → solo l'esito 1\n"
        "`3 O25 1.85` → Over 2.5\n\n"
        "_Il backtest dice −12% EV per scalata a 3 step. È un laboratorio, "
        "non una fonte di reddito._",
        parse_mode=ParseMode.MARKDOWN,
    )


async def oggi(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    avviso = await update.message.reply_text(
        "Scarico risultati e quote, stimo il modello… (30-60 secondi la prima volta)"
    )
    try:
        import asyncio

        loop = asyncio.get_running_loop()
        a = await loop.run_in_executor(None, E.aggiorna)
    except Exception as e:
        log.exception("aggiorna")
        await avviso.edit_text(f"Errore durante l'aggiornamento:\n`{e}`",
                               parse_mode=ParseMode.MARKDOWN)
        return

    if a.errore and a.modello is None:
        await avviso.edit_text(f"Non sono riuscito a scaricare i dati:\n`{a.errore}`",
                               parse_mode=ParseMode.MARKDOWN)
        return

    if not a.partite:
        await avviso.edit_text(
            "Nessuna partita di Serie A o Premier League nel file delle prossime gare.\n\n"
            "Il file viene aggiornato il venerdì pomeriggio per il weekend e il "
            "martedì per i turni infrasettimanali. Se i campionati sono fermi, "
            "è normale che sia vuoto."
        )
        return

    ctx.bot_data["partite"] = {p["n"]: p for p in a.partite}
    blocchi = [riga_partita(p) for p in a.partite[:15]]
    testa = (
        f"*{len(a.partite)} partite in programma*\n"
        f"_riferimento: Betfair Exchange · modello su {a.modello.n_partite} partite_\n\n"
    )
    coda = (
        "\n\n_Manda il numero della partita e le quote di Sisal._\n"
        "_Esempio:_ `1 2.45 3.20 1.80`"
    )
    await avviso.edit_text(testa + "\n\n".join(blocchi) + coda,
                           parse_mode=ParseMode.MARKDOWN)


async def quote(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Interpreta un messaggio di quote in una riga."""
    testo = update.message.text.strip()
    pezzi = testo.replace(",", ".").split()
    partite = ctx.bot_data.get("partite")

    if not partite:
        await update.message.reply_text("Prima manda /oggi per caricare le partite.")
        return

    try:
        n = int(pezzi[0])
    except (ValueError, IndexError):
        await update.message.reply_text(
            "Non ho capito. Formato: `3 2.45 3.20 1.80`", parse_mode=ParseMode.MARKDOWN
        )
        return

    p = partite.get(n)
    if not p:
        await update.message.reply_text(f"La partita {n} non esiste. Controlla con /oggi.")
        return

    # forma lunga: numero + tre quote 1X2
    if len(pezzi) == 4 and all(re.fullmatch(r"\d+(\.\d+)?", x) for x in pezzi[1:]):
        richieste = list(zip(("1", "X", "2"), (float(x) for x in pezzi[1:])))
    # forma corta: numero + esito + quota
    elif len(pezzi) == 3:
        es = pezzi[1].upper()
        if es not in ESITI_1X2 | ESITI_ALTRI:
            await update.message.reply_text(
                f"Esito `{pezzi[1]}` non riconosciuto.\n"
                f"Validi: 1, X, 2, O25, U25, GG, 1X, X2, 12",
                parse_mode=ParseMode.MARKDOWN,
            )
            return
        try:
            richieste = [(es, float(pezzi[2]))]
        except ValueError:
            await update.message.reply_text("La quota non è un numero valido.")
            return
    else:
        await update.message.reply_text(
            "Formato non valido.\n`3 2.45 3.20 1.80` oppure `3 1 2.45`",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    u = utente(update.effective_user.id)
    righe = [f"*{p['casa']} – {p['trasferta']}*  ({p['lega']})\n"]
    migliore = None

    for esito, q in richieste:
        if q <= 1.0:
            righe.append(f"`{esito}` quota {q} non valida.")
            continue
        r = E.analizza_quota(p, esito, q)

        if r["p_mercato"] is None:
            righe.append(
                f"*{esito}* · quota {q:.2f}\n"
                f"    nessuna linea di riferimento per questo mercato"
            )
            continue

        segno = "✅" if r["ev_mercato"] > 0.02 else ("○" if r["ev_mercato"] > 0 else "✗")
        blocco = (
            f"{segno} *{esito}* · Sisal {q:.2f}\n"
            f"    quota equa {r['quota_equa']:.2f} · mercato {pct(r['p_mercato'])}\n"
            f"    *EV {r['ev_mercato']*100:+.2f}%*"
        )
        if r["p_modello"] is not None:
            blocco += f"\n    modello {pct(r['p_modello'])} (EV {r['ev_modello']*100:+.1f}%)"
        righe.append(blocco)

        if migliore is None or r["ev_mercato"] > migliore[1]["ev_mercato"]:
            migliore = (esito, r)

    if migliore and migliore[1]["ev_mercato"] > 0.02:
        e2, r2 = migliore
        stake = min(u["capitale"] * 0.05, u["capitale"])
        s = E.scalata(
            [r2["p_mercato"]] * 3, [r2["quota"]] * 3, stake
        )
        righe.append(
            f"\n*Scalata a 3 step su {e2}*\n"
            f"    €{stake:.0f} → €{s['target']:.0f}\n"
            f"    probabilità di completarla *{s['p_completamento']*100:.1f}%*\n"
            f"    EV complessivo *{s['ev_pct']*100:+.1f}%*"
        )
        if s["ev_pct"] < 0:
            righe.append(
                "    _EV negativo: il margine si compone su tre step e "
                "si mangia il vantaggio del singolo._"
            )
    elif migliore:
        righe.append("\n_Nessun esito con valore sufficiente. Meglio lasciar stare._")

    await update.message.reply_text("\n\n".join(righe), parse_mode=ParseMode.MARKDOWN)


async def stato(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    u = utente(uid)
    with db() as c:
        g = c.execute(
            "SELECT COUNT(*) n, COALESCE(SUM(pnl),0) pnl, COALESCE(AVG(ev),0) ev "
            "FROM giocate WHERE uid=?",
            (uid,),
        ).fetchone()

    await update.message.reply_text(
        f"*Conto*\n\n"
        f"Capitale: €{u['capitale']:.2f}\n"
        f"Giocate registrate: {g['n']}\n"
        f"P/L: €{g['pnl']:+.2f}\n"
        f"EV medio all'inserimento: {g['ev']*100:+.2f}%\n\n"
        "_Con pochi dati il P/L è quasi tutto rumore. "
        "Il numero da guardare, quando ce ne saranno abbastanza, è il CLV._",
        parse_mode=ParseMode.MARKDOWN,
    )


async def reset(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    with db() as c:
        c.execute("DELETE FROM giocate WHERE uid=?", (uid,))
        c.execute("UPDATE utenti SET capitale=? WHERE uid=?", (CAPITALE_DEFAULT, uid))
    await update.message.reply_text(f"Conto azzerato. Capitale €{CAPITALE_DEFAULT:.0f}.")


async def errore(update, ctx: ContextTypes.DEFAULT_TYPE):
    log.error("errore non gestito", exc_info=ctx.error)


def main():
    if not TOKEN:
        raise SystemExit("Manca la variabile TELEGRAM_TOKEN")
    init_db()

    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("oggi", oggi))
    app.add_handler(CommandHandler("stato", stato))
    app.add_handler(CommandHandler("reset", reset))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, quote))
    app.add_error_handler(errore)

    log.info("bot avviato")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
