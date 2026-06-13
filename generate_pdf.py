"""
Generate PDF report of today's predictions.
"""
from __future__ import annotations
import json
import sys
from datetime import datetime
from pathlib import Path
from fpdf import FPDF


CACHE_DIR = Path("cache/apifootball")


class PredictionPDF(FPDF):
    def __init__(self, date_str: str):
        super().__init__(orientation="L", unit="mm", format="A4")
        self.date_str = date_str
        self.set_auto_page_break(auto=True, margin=15)

    def header(self):
        self.set_font("Helvetica", "B", 14)
        self.cell(0, 8, f"BetSightly Predictions - {self.date_str}", ln=True, align="C")
        self.set_font("Helvetica", "", 8)
        self.cell(0, 5, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}  |  Powered by 244K+ historical matches", ln=True, align="C")
        self.ln(2)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 7)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}  |  BetSightly - AI Sports Predictions  |  Disclaimer: For informational purposes only", align="C")

    def add_league_header(self, league_name: str, country: str):
        self.set_fill_color(30, 60, 120)
        self.set_text_color(255, 255, 255)
        self.set_font("Helvetica", "B", 9)
        safe = self._safe(f"  {league_name} ({country})")
        self.cell(0, 7, safe, ln=True, fill=True)
        self.set_text_color(0, 0, 0)

    def add_table_header(self):
        self.set_font("Helvetica", "B", 7)
        self.set_fill_color(220, 220, 230)
        cols = [
            ("Time", 14), ("Home Team", 48), ("Away Team", 48),
            ("Prediction", 22), ("Home%", 14), ("Draw%", 14), ("Away%", 14),
            ("O1.5", 12), ("O2.5", 12), ("O3.5", 12), ("BTTS", 12),
            ("xHG", 12), ("xAG", 12), ("Conf", 14), ("Data", 14),
        ]
        for name, w in cols:
            self.cell(w, 6, name, border=1, align="C", fill=True)
        self.ln()

    def add_match_row(self, p: dict, row_idx: int):
        # Alternating row colors
        if row_idx % 2 == 0:
            self.set_fill_color(245, 245, 250)
        else:
            self.set_fill_color(255, 255, 255)

        self.set_font("Helvetica", "", 7)

        # Color-code prediction
        pred = p["prediction"]
        if pred == "Home Win":
            pred_short = "1 - HOME"
        elif pred == "Away Win":
            pred_short = "2 - AWAY"
        else:
            pred_short = "X - DRAW"

        # Color-code confidence
        conf = p["confidence"]
        data_q = p["data_quality"]

        home = self._safe(p["home_team"][:25])
        away = self._safe(p["away_team"][:25])

        cols = [
            (p["kick_off"], 14),
            (home, 48),
            (away, 48),
            (pred_short, 22),
            (f'{p["home_win"]}', 14),
            (f'{p["draw"]}', 14),
            (f'{p["away_win"]}', 14),
            (f'{p["over_1.5"]}', 12),
            (f'{p["over_2.5"]}', 12),
            (f'{p["over_3.5"]}', 12),
            (f'{p["btts"]}', 12),
            (f'{p["expected_home_goals"]}', 12),
            (f'{p["expected_away_goals"]}', 12),
            (f'{conf}%', 14),
            (data_q, 14),
        ]

        # Highlight prediction cell
        for i, (val, w) in enumerate(cols):
            if i == 3:  # Prediction column
                if pred == "Home Win":
                    self.set_fill_color(200, 230, 200)
                elif pred == "Away Win":
                    self.set_fill_color(200, 210, 240)
                else:
                    self.set_fill_color(255, 240, 200)
                self.cell(w, 5, val, border=1, align="C", fill=True)
                # Reset fill
                if row_idx % 2 == 0:
                    self.set_fill_color(245, 245, 250)
                else:
                    self.set_fill_color(255, 255, 255)
            elif i == 14:  # Data quality
                if data_q == "HIGH":
                    self.set_fill_color(180, 230, 180)
                elif data_q == "MEDIUM":
                    self.set_fill_color(240, 230, 180)
                else:
                    self.set_fill_color(240, 200, 200)
                self.cell(w, 5, val, border=1, align="C", fill=True)
                if row_idx % 2 == 0:
                    self.set_fill_color(245, 245, 250)
                else:
                    self.set_fill_color(255, 255, 255)
            else:
                align = "L" if i in (1, 2) else "C"
                self.cell(w, 5, val, border=1, align=align, fill=True)
        self.ln()

    def add_top_picks_page(self, predictions: list):
        self.add_page()
        self.set_font("Helvetica", "B", 14)
        self.cell(0, 10, "TOP PICKS - High Confidence Selections", ln=True, align="C")
        self.ln(3)

        # Filter good picks
        good = [p for p in predictions if p["data_quality"] in ("HIGH", "MEDIUM") and p["confidence"] >= 35]
        good.sort(key=lambda x: -x["confidence"])

        if not good:
            self.set_font("Helvetica", "", 10)
            self.cell(0, 10, "No high-confidence picks today.", ln=True, align="C")
            return

        # Top picks table
        self.set_font("Helvetica", "B", 8)
        self.set_fill_color(30, 60, 120)
        self.set_text_color(255, 255, 255)
        top_cols = [
            ("#", 8), ("Time", 14), ("Home Team", 52), ("Away Team", 52),
            ("League", 45), ("Pick", 25), ("Home%", 14), ("Draw%", 14),
            ("Away%", 14), ("O2.5", 14), ("BTTS", 14), ("Conf", 14),
        ]
        for name, w in top_cols:
            self.cell(w, 7, name, border=1, align="C", fill=True)
        self.ln()
        self.set_text_color(0, 0, 0)

        for i, p in enumerate(good):
            if i % 2 == 0:
                self.set_fill_color(235, 245, 235)
            else:
                self.set_fill_color(255, 255, 255)

            self.set_font("Helvetica", "B" if p["data_quality"] == "HIGH" else "", 7)

            pred = p["prediction"]
            if pred == "Home Win": pick = "1 - HOME"
            elif pred == "Away Win": pick = "2 - AWAY"
            else: pick = "X - DRAW"

            home = self._safe(p["home_team"][:27])
            away = self._safe(p["away_team"][:27])
            league = self._safe(p["league"][:23])

            vals = [
                (str(i+1), 8), (p["kick_off"], 14), (home, 52), (away, 52),
                (league, 45), (pick, 25), (f'{p["home_win"]}', 14),
                (f'{p["draw"]}', 14), (f'{p["away_win"]}', 14),
                (f'{p["over_2.5"]}', 14), (f'{p["btts"]}', 14),
                (f'{p["confidence"]}%', 14),
            ]
            for j, (val, w) in enumerate(vals):
                align = "L" if j in (2, 3, 4) else "C"
                self.cell(w, 6, val, border=1, align=align, fill=True)
            self.ln()

    def add_summary_page(self, predictions: list):
        self.add_page()
        self.set_font("Helvetica", "B", 14)
        self.cell(0, 10, "PREDICTION SUMMARY", ln=True, align="C")
        self.ln(5)

        total = len(predictions)
        high = sum(1 for p in predictions if p["confidence"] >= 60)
        med = sum(1 for p in predictions if 40 <= p["confidence"] < 60)
        low = sum(1 for p in predictions if p["confidence"] < 40)
        d_high = sum(1 for p in predictions if p["data_quality"] == "HIGH")
        d_med = sum(1 for p in predictions if p["data_quality"] == "MEDIUM")
        d_low = sum(1 for p in predictions if p["data_quality"] == "LOW")
        home_wins = sum(1 for p in predictions if p["prediction"] == "Home Win")
        away_wins = sum(1 for p in predictions if p["prediction"] == "Away Win")
        draws = sum(1 for p in predictions if p["prediction"] == "Draw")

        # Count leagues
        leagues = set()
        countries = set()
        for p in predictions:
            leagues.add(p["league"])
            countries.add(p["country"])

        stats = [
            ("Total Predictions", str(total)),
            ("Leagues Covered", str(len(leagues))),
            ("Countries", str(len(countries))),
            ("", ""),
            ("CONFIDENCE BREAKDOWN", ""),
            ("High Confidence (>=60%)", str(high)),
            ("Medium Confidence (40-59%)", str(med)),
            ("Low Confidence (<40%)", str(low)),
            ("", ""),
            ("DATA QUALITY", ""),
            ("HIGH (both teams found, 10+ matches)", str(d_high)),
            ("MEDIUM (partial team data)", str(d_med)),
            ("LOW (no historical data)", str(d_low)),
            ("", ""),
            ("PREDICTION DISTRIBUTION", ""),
            ("Home Wins", f"{home_wins} ({home_wins/total*100:.1f}%)"),
            ("Draws", f"{draws} ({draws/total*100:.1f}%)"),
            ("Away Wins", f"{away_wins} ({away_wins/total*100:.1f}%)"),
        ]

        self.set_font("Helvetica", "", 10)
        for label, value in stats:
            if not label and not value:
                self.ln(3)
                continue
            if not value:
                self.set_font("Helvetica", "B", 10)
                self.cell(0, 7, label, ln=True)
                self.set_font("Helvetica", "", 10)
                continue
            self.cell(100, 7, f"  {label}:", border=0)
            self.set_font("Helvetica", "B", 10)
            self.cell(50, 7, value, ln=True, border=0)
            self.set_font("Helvetica", "", 10)

        # Legend
        self.ln(10)
        self.set_font("Helvetica", "B", 9)
        self.cell(0, 7, "LEGEND", ln=True)
        self.set_font("Helvetica", "", 8)
        legend = [
            "Home%/Draw%/Away% = Probability of each outcome",
            "O1.5/O2.5/O3.5 = Probability of Over 1.5/2.5/3.5 total goals",
            "BTTS = Both Teams To Score probability",
            "xHG/xAG = Expected Home/Away Goals",
            "Conf = Model confidence based on data availability",
            "Data = HIGH (both teams, 10+ matches), MEDIUM (partial), LOW (no data)",
            "",
            "Historical data: 244,273 matches from GitHub + API-Football datasets",
            "Prediction method: ELO ratings + form analysis + H2H + Poisson goal model",
        ]
        for line in legend:
            if line:
                self.cell(0, 5, f"  {line}", ln=True)
            else:
                self.ln(2)

    def _safe(self, text: str) -> str:
        """Make text safe for PDF (ASCII fallback)."""
        try:
            text.encode("latin-1")
            return text
        except UnicodeEncodeError:
            return text.encode("ascii", "replace").decode()


def main():
    date_str = datetime.now().strftime("%Y-%m-%d")
    if len(sys.argv) > 1:
        date_str = sys.argv[1]

    pred_file = CACHE_DIR / f"predictions_{date_str}.json"
    if not pred_file.exists():
        print(f"No predictions found at {pred_file}")
        print("Run predict_today.py first!")
        return

    with open(pred_file, "r", encoding="utf-8") as f:
        predictions = json.load(f)

    print(f"Loaded {len(predictions)} predictions for {date_str}")

    pdf = PredictionPDF(date_str)
    pdf.alias_nb_pages()

    # Page 1: Top Picks
    pdf.add_top_picks_page(predictions)

    # Page 2: Summary
    pdf.add_summary_page(predictions)

    # Remaining pages: All matches grouped by league
    # Sort predictions by league then kick_off
    by_league = {}
    for p in predictions:
        key = (p["country"], p["league"])
        by_league.setdefault(key, []).append(p)

    pdf.add_page()
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 10, "ALL PREDICTIONS BY LEAGUE", ln=True, align="C")
    pdf.ln(2)

    row_idx = 0
    for (country, league), preds in sorted(by_league.items()):
        # Check if we need a new page (need at least 30mm for header + 1 row)
        if pdf.get_y() > 170:
            pdf.add_page()

        pdf.add_league_header(league, country)
        pdf.add_table_header()

        for p in sorted(preds, key=lambda x: x["kick_off"]):
            if pdf.get_y() > 190:
                pdf.add_page()
                pdf.add_league_header(league + " (cont.)", country)
                pdf.add_table_header()
            pdf.add_match_row(p, row_idx)
            row_idx += 1

        pdf.ln(2)

    # Save
    output_path = Path(f"BetSightly_Predictions_{date_str}.pdf")
    pdf.output(str(output_path))
    print(f"\nPDF saved to: {output_path.absolute()}")


if __name__ == "__main__":
    main()
