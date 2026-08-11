"""
Prediction_Bot — Sistema di value betting con scalate a 3 step.

Uso:
  python bot.py

Variabili di ambiente:
  TELEGRAM_TOKEN — token del bot
  DATABASE_PATH — percorso del database SQLite (default: ./bot.db)
"""

import os
import sqlite3
import json
from datetime import datetime, timedelta
from decimal import Decimal

import numpy as np
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
    ConversationHandler,
)

# ========== CONFIG ==========
TOKEN = os.getenv("TELEGRAM_TOKEN", "YOUR_TOKEN_HERE")
DB_PATH = os.getenv("DATABASE_PATH", "./bot.db")
CAPITALE_INIZIALE = 300.0
COSTO_ORA = 9.0

# Stati della conversation
ATTESA_PARTITA, ATTESA_QUOTE_H, ATTESA_QUOTE_D, ATTESA_QUOTE_A = range(4)
ATTESA_QUOTE_O25, ATTESA_QUOTE_U25, ATTESA_QUOTE_BTTS, ATTESA_SCALATA = range(4, 8)
ATTESA_RISULTATO_PARTITA, ATTESA_RISULTATO_ESITO = range(8, 10)


def devig_power(odds):
    """Devigging metodo power."""
    q = np.array([1.0 / o for o in odds], dtype=float)
    s = q.sum()
    if abs(s - 1.0) < 1e-9:
        return q
    from scipy.optimize import brentq

    def f(k):
        return np.sum(q**k) - 1.0

    try:
        k = brentq(f, 0.2, 3.0, xtol=1e-10, maxiter=200)
    except ValueError:
        return q / s
    p = q**k
    return p / p.sum()


def kelly_frac(p, odds, cap=0.05):
    """Kelly frazionario con cap."""
    if p <= 0 or p >= 1 or odds <= 1:
        return 0.0
    b = odds - 1.0
    f = (p * b - (1.0 - p)) / b
    return min(max(f, 0.0), cap)


def ev_calc(p, odds):
    """Expected value per unità di stake."""
    return p * odds - 1.0


# ========== DATABASE ==========
class DB:
    def __init__(self, path=DB_PATH):
        self.path = path
        self._init()

    def _init(self):
        conn = sqlite3.connect(self.path)
        c = conn.cursor()
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS partite (
                id INTEGER PRIMARY KEY,
                user_id INTEGER,
                data TEXT,
                squadre TEXT,
                stato TEXT DEFAULT 'aperta'
            )
        """
        )
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS quote (
                id INTEGER PRIMARY KEY,
                partita_id INTEGER,
                mercato TEXT,
                book TEXT,
                quota REAL,
                FOREIGN KEY(partita_id) REFERENCES partite(id)
            )
        """
        )
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS scalate (
                id INTEGER PRIMARY KEY,
                user_id INTEGER,
                partita_id INTEGER,
                mercato TEXT,
                stake REAL,
                target REAL,
                quote TEXT,
                data TEXT,
                stato TEXT DEFAULT 'in corso',
                risultato TEXT,
                pnl REAL,
                clv REAL
            )
        """
        )
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS stato_utente (
                user_id INTEGER PRIMARY KEY,
                capitale REAL DEFAULT {},
                totale_scalate INTEGER DEFAULT 0,
                vinte INTEGER DEFAULT 0,
                profitto REAL DEFAULT 0,
                ultima_aggiornamento TEXT
            )
        """.format(
                CAPITALE_INIZIALE
            )
        )
        conn.commit()
        conn.close()

    def conn_ctx(self):
        return sqlite3.connect(self.path)

    def get_user_state(self, user_id):
        conn = self.conn_ctx()
        c = conn.cursor()
        c.execute("SELECT capitale, totale_scalate, vinte, profitto FROM stato_utente WHERE user_id=?", (user_id,))
        row = c.fetchone()
        conn.close()
        if row:
            return {"capitale": row[0], "scalate": row[1], "vinte": row[2], "profitto": row[3]}
        return {"capitale": CAPITALE_INIZIALE, "scalate": 0, "vinte": 0, "profitto": 0}

    def save_user_state(self, user_id, capitale, scalate, vinte, profitto):
        conn = self.conn_ctx()
        c = conn.cursor()
        c.execute(
            """
            INSERT OR REPLACE INTO stato_utente (user_id, capitale, totale_scalate, vinte, profitto, ultima_aggiornamento)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            (user_id, capitale, scalate, vinte, profitto, datetime.now().isoformat()),
        )
        conn.commit()
        conn.close()

    def add_partita(self, user_id, squadre):
        conn = self.conn_ctx()
        c = conn.cursor()
        c.execute(
            "INSERT INTO partite (user_id, squadre, data) VALUES (?, ?, ?)",
            (user_id, squadre, datetime.now().isoformat()),
        )
        pid = c.lastrowid
        conn.commit()
        conn.close()
        return pid

    def add_quote(self, partita_id, mercato, book, quota):
        conn = self.conn_ctx()
        c = conn.cursor()
        c.execute(
            "INSERT INTO quote (partita_id, mercato, book, quota) VALUES (?, ?, ?, ?)",
            (partita_id, mercato, book, quota),
        )
        conn.commit()
        conn.close()

    def get_partita_quote(self, partita_id):
        conn = self.conn_ctx()
        c = conn.cursor()
        c.execute("SELECT mercato, book, quota FROM quote WHERE partita_id=? ORDER BY mercato", (partita_id,))
        rows = c.fetchall()
        conn.close()
        out = {}
        for m, b, q in rows:
            if m not in out:
                out[m] = {}
            out[m][b] = q
        return out

    def add_scalata(self, user_id, partita_id, mercato, stake, target, quote_list, clv):
        conn = self.conn_ctx()
        c = conn.cursor()
        c.execute(
            """
            INSERT INTO scalate (user_id, partita_id, mercato, stake, target, quote, clv, data)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (user_id, partita_id, mercato, stake, target, json.dumps(quote_list), clv, datetime.now().isoformat()),
        )
        sid = c.lastrowid
        conn.commit()
        conn.close()
        return sid

    def update_scalata_result(self, scalata_id, vinta, pnl):
        conn = self.conn_ctx()
        c = conn.cursor()
        c.execute(
            "UPDATE scalate SET stato=?, risultato=?, pnl=? WHERE id=?",
            ("vinta" if vinta else "persa", "vinto" if vinta else "perso", pnl, scalata_id),
        )
        conn.commit()
        conn.close()

    def get_user_scalate(self, user_id, limit=10):
        conn = self.conn_ctx()
        c = conn.cursor()
        c.execute(
            "SELECT id, mercato, stake, target, quote, stato, pnl, clv FROM scalate WHERE user_id=? ORDER BY id DESC LIMIT ?",
            (user_id, limit),
        )
        rows = c.fetchall()
        conn.close()
        return rows


db = DB()

# ========== HANDLERS ==========
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Inizia il bot."""
    user_id = update.effective_user.id
    state = db.get_user_state(user_id)
    msg = f"""
Benvenuto in Prediction_Bot.

Questo è un laboratorio di value betting: inserisci le quote, il sistema calcola le scalate, tu decidi se giocare.

**Capitale:** €{state['capitale']:.0f}
**Scalate:** {state['scalate']} (vinte: {state['vinte']})
**Profitto:** €{state['profitto']:+.0f}

Comandi:
/valuta — analizza una nuova partita
/stato — vedi statistiche
/risultato — registra l'esito di una scalata
/reset — ricomincia

⚠️ Ricorda: −12% EV sulle scalate a 3 step. È un esperimento.
"""
    await update.message.reply_text(msg)


async def valuta_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Inizia la valutazione di una partita."""
    await update.message.reply_text("Come si chiama la partita? (es: Napoli-Roma)")
    return ATTESA_PARTITA


async def valuta_partita(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Ricevi il nome della partita."""
    user_id = update.effective_user.id
    squadre = update.message.text.strip()
    partita_id = db.add_partita(user_id, squadre)
    context.user_data["partita_id"] = partita_id
    context.user_data["quote"] = {}

    msg = (
        f"Partita: {squadre}\n\n"
        "Inserisci la quota per il **1** (vittoria casa).\n"
        "Formato: 2.45 o 2,45"
    )
    await update.message.reply_text(msg)
    return ATTESA_QUOTE_H


async def quote_h(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        q = float(update.message.text.replace(",", "."))
        if q < 1.01:
            raise ValueError
        context.user_data["quote"]["H"] = q
        db.add_quote(context.user_data["partita_id"], "1X2", "sisal", q)
    except ValueError:
        await update.message.reply_text("Quota non valida. Riprova (es: 2.45)")
        return ATTESA_QUOTE_H

    await update.message.reply_text("Quota per il **pareggio** (X)?")
    return ATTESA_QUOTE_D


async def quote_d(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        q = float(update.message.text.replace(",", "."))
        if q < 1.01:
            raise ValueError
        context.user_data["quote"]["D"] = q
        db.add_quote(context.user_data["partita_id"], "1X2", "sisal", q)
    except ValueError:
        await update.message.reply_text("Quota non valida. Riprova.")
        return ATTESA_QUOTE_D

    await update.message.reply_text("Quota per il **2** (vittoria trasferta)?")
    return ATTESA_QUOTE_A


async def quote_a(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        q = float(update.message.text.replace(",", "."))
        if q < 1.01:
            raise ValueError
        context.user_data["quote"]["A"] = q
        db.add_quote(context.user_data["partita_id"], "1X2", "sisal", q)
    except ValueError:
        await update.message.reply_text("Quota non valida. Riprova.")
        return ATTESA_QUOTE_A

    await update.message.reply_text("Quota per **Over 2.5**? (premi 0 per saltare)")
    return ATTESA_QUOTE_O25


async def quote_o25(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        val = update.message.text.strip()
        if val == "0":
            context.user_data["quote"]["O25"] = None
        else:
            q = float(val.replace(",", "."))
            if q < 1.01:
                raise ValueError
            context.user_data["quote"]["O25"] = q
            db.add_quote(context.user_data["partita_id"], "Over2.5", "sisal", q)
    except ValueError:
        await update.message.reply_text("Non valido. Riprova o scrivi 0.")
        return ATTESA_QUOTE_O25

    await update.message.reply_text("Quota per **BTTS** (entrambe segnano)? (0 per saltare)")
    return ATTESA_QUOTE_BTTS


async def quote_btts(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        val = update.message.text.strip()
        if val == "0":
            context.user_data["quote"]["BTTS"] = None
        else:
            q = float(val.replace(",", "."))
            if q < 1.01:
                raise ValueError
            context.user_data["quote"]["BTTS"] = q
            db.add_quote(context.user_data["partita_id"], "BTTS", "sisal", q)
    except ValueError:
        await update.message.reply_text("Non valido. Riprova o scrivi 0.")
        return ATTESA_QUOTE_BTTS

    # calcola e proponi scalate
    await calcola_scalate(update, context)
    return ConversationHandler.END


async def calcola_scalate(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Calcola e propone le scalate."""
    q_dict = context.user_data["quote"]
    user_state = db.get_user_state(update.effective_user.id)
    capitale = user_state["capitale"]

    # 1X2
    q_1x2 = [q_dict["H"], q_dict["D"], q_dict["A"]]
    p_sharp = devig_power(q_1x2)

    msg = "**ANALISI 1X2**\n\n"
    for label, prob in zip(["1", "X", "2"], p_sharp):
        msg += f"{label}: {prob*100:.1f}%  quota sharp {1/prob:.2f}\n"

    # Calcola scalate per ogni esito
    scalate_proposte = []
    for i, label in enumerate(["1", "X", "2"]):
        p = p_sharp[i]
        odds = q_1x2[i]
        ev_base = ev_calc(p, odds)
        clv_val = odds * p - 1.0

        if ev_base > 0.02:  # soglia minima EV
            for stake_pct in [0.05, 0.10, 0.15]:
                stake = capitale * stake_pct
                if stake < 10:
                    continue
                target = stake * (odds**3)
                if target > capitale * 2:
                    continue
                scalate_proposte.append({
                    "mercato": f"1X2-{label}",
                    "stake": stake,
                    "target": target,
                    "quote": [odds] * 3,
                    "clv": clv_val,
                    "ev": ev_base,
                })

    if scalate_proposte:
        msg += "\n**SCALATE A 3 STEP PROPOSTE**\n\n"
        for i, s in enumerate(sorted(scalate_proposte, key=lambda x: x["clv"], reverse=True)[:3]):
            msg += (
                f"{i+1}. {s['mercato']}\n"
                f"   Stake: €{s['stake']:.0f} → target €{s['target']:.0f}\n"
                f"   EV: {s['ev']*100:+.1f}%  CLV: {s['clv']*100:+.1f}%\n\n"
            )
        context.user_data["scalate_proposte"] = scalate_proposte
    else:
        msg += "\nNessuna scalata con EV positivo."

    msg += f"\n_Capitale disponibile: €{capitale:.0f}_"
    await update.message.reply_text(msg, parse_mode="Markdown")


async def stato(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Mostra lo stato del conto."""
    user_id = update.effective_user.id
    state = db.get_user_state(user_id)

    msg = f"""
**STATO CONTO**

Capitale: €{state['capitale']:.0f}
Scalate totali: {state['scalate']}
Vinte: {state['vinte']}
Win rate: {state['scalate'] > 0 and (state['vinte']*100/state['scalate']):.0f or 0}%
Profitto netto: €{state['profitto']:+.0f}

---

Le ultime 5 scalate:
"""

    scalate = db.get_user_scalate(user_id, limit=5)
    if scalate:
        for sid, merc, stake, target, quote_str, stato_s, pnl, clv in scalate:
            stato_emoji = "✓" if stato_s == "vinta" else "✗" if stato_s == "persa" else "⏳"
            msg += f"\n{stato_emoji} {merc} · stake €{stake:.0f} → €{target:.0f}"
            if pnl:
                msg += f" · {pnl:+.0f}"
    else:
        msg += "\n(nessuna ancora)"

    await update.message.reply_text(msg, parse_mode="Markdown")


async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Reset conto."""
    user_id = update.effective_user.id
    db.save_user_state(user_id, CAPITALE_INIZIALE, 0, 0, 0)
    await update.message.reply_text(f"✓ Conto resettato. Capitale: €{CAPITALE_INIZIALE:.0f}")


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    """Log errors."""
    print(f"Update {update} caused error {context.error}")


# ========== MAIN ==========
def main():
    app = Application.builder().token(TOKEN).build()

    conv_valuta = ConversationHandler(
        entry_points=[CommandHandler("valuta", valuta_start)],
        states={
            ATTESA_PARTITA: [MessageHandler(filters.TEXT & ~filters.COMMAND, valuta_partita)],
            ATTESA_QUOTE_H: [MessageHandler(filters.TEXT & ~filters.COMMAND, quote_h)],
            ATTESA_QUOTE_D: [MessageHandler(filters.TEXT & ~filters.COMMAND, quote_d)],
            ATTESA_QUOTE_A: [MessageHandler(filters.TEXT & ~filters.COMMAND, quote_a)],
            ATTESA_QUOTE_O25: [MessageHandler(filters.TEXT & ~filters.COMMAND, quote_o25)],
            ATTESA_QUOTE_BTTS: [MessageHandler(filters.TEXT & ~filters.COMMAND, quote_btts)],
        },
        fallbacks=[CommandHandler("cancel", lambda u, c: ConversationHandler.END)],
    )

    app.add_handler(CommandHandler("start", start))
    app.add_handler(conv_valuta)
    app.add_handler(CommandHandler("stato", stato))
    app.add_handler(CommandHandler("reset", reset))
    app.add_error_handler(error_handler)

    print("Bot in ascolto...")
    app.run_polling()


if __name__ == "__main__":
    main()
