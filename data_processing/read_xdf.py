import pyxdf
import pandas as pd

def load_xdf(fname):
    streams, _ = pyxdf.load_xdf(fname, synchronize_clocks=True)
    
    return {
        'eeg': _find_stream(streams, lambda s: "eeg" in s["info"]["name"][0].lower()),
        'game': _find_stream(streams, lambda s: any(
            kw in s["info"]["name"][0].lower() 
            for kw in ("urban", "nature", "control")
        )),
        'nasatlx': _find_stream(streams, lambda s: s["info"]["type"][0] == "Responses"),
        'motion': _find_stream(streams, lambda s: "motion" in s["info"]["name"][0].lower())
    }
    
def _find_stream(streams, predicate):
        try:
            return next(s for s in streams if predicate(s))
        except StopIteration:
            print(f"Warning: Missing {predicate.__name__} stream")
            return None
