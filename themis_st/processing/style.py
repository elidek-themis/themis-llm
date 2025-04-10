from typing import List
import pandas as pd

def nll_styler(df: pd.DataFrame) -> pd.DataFrame.style:
    return df.style.background_gradient(
       cmap="RdYlGn_r",
       vmin=0,
       vmax=10 
    )


def prob_styler(df: pd.DataFrame) -> pd.DataFrame.style:
    def highlight_min_max(s):
        min_val = s.min()
        max_val = s.max()
        return [
            "background-color: #ea9999; color: black" if v == min_val else 
            "background-color: #a4c2f4; color: black" if v == max_val else 
            "" 
            for v in s
        ]
    
    return df.style.apply(highlight_min_max)


def norm_prob_styler(df: pd.DataFrame) -> pd.DataFrame.style:
    return df.style.background_gradient(
        cmap="Greens",
        vmin=0,
        vmax=1,
        text_color_threshold=0.3
    )


def diff_styler(df: pd.DataFrame, subset: List):
    fn = lambda x: "background-color: %s; color:black" % ("#ea9999", "#a4c2f4")[x>0]
    return df.style.map(
            func=fn,
            subset=pd.IndexSlice[slice(None), subset]
        )


def stats_styler(df:pd.DataFrame):
    fn = lambda x: "background-color: %s; color:black" % ("#ea9999", "#a4c2f4")[x>0]
    
    us_subset = pd.IndexSlice[["U.S. prompt", "state_avg"], :]
    max_subset = pd.IndexSlice["avg_agreement", :]
    min_subset = pd.IndexSlice[["avg_abs_pct_diff", "avg_relative_error"], :]
    
    
    styled_df = (
        df.style
        .map(fn, subset=us_subset)
        .highlight_max(axis=1, subset=max_subset)
        .highlight_min(axis=1, subset=min_subset)
    )
    
    return styled_df