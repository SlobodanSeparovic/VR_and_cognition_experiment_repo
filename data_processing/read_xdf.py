import pyxdf
import pandas as pd

def load_xdf(fname):
    streams, _ = pyxdf.load_xdf(fname, synchronize_clocks=True)
    return {
        'eeg': next(s for s in streams if "eeg" in s["info"]["name"][0].lower()),
        'game': next(s for s in streams if any(kw in s["info"]["name"][0].lower() for kw in ("urban", "nature", "control"))),
        'nasatlx': next(s for s in streams if s["info"]["type"][0] == "Responses"),
        'motion': next(s for s in streams if "motion" in s["info"]["name"][0].lower())
    }