import copy

from math import exp
from typing import Any

import pandas as pd

from lm_eval.loggers.utils import remove_none_pattern

from themis.data.repository import ExperimentOutput


class ElectionResults:
    def __init__(self, output: ExperimentOutput) -> None:
        results, tasks = output.results, output.tasks
        self.metrics = self._sanitize_results(results=results, task_names=tasks)

        task = next(iter(tasks))
        self.keys = [doc["doc"]["key"] for doc in results["samples"][task]]

        self.choices = {task: output.task_configs[task]["dataset_kwargs"]["choices"] for task in tasks}
        self.columns = {task: output.task_configs[task]["dataset_kwargs"]["columns"] for task in tasks}
        # self.choices = output.task_configs[task]["dataset_kwargs"]["choices"]
        # self.columns = output.task_configs[task]["dataset_kwargs"]["columns"]

    def _sanitize_results(self, results: dict[str, Any], task_names: str | list[str]) -> dict[str, dict[str, Any]]:
        metrics = copy.deepcopy(results.get("results", {}))

        tmp_metrics = copy.deepcopy(metrics)
        for task_name in task_names:
            task_metrics = tmp_metrics.get(task_name, {})
            for metric_name, metric_value in task_metrics.items():
                _metric_name, removed = remove_none_pattern(metric_name)
                if isinstance(metric_value, str):
                    metrics[task_name].pop(metric_name)
                elif removed:
                    metrics[task_name][_metric_name] = metric_value
                    metrics[task_name].pop(metric_name)

        return metrics


def get_nll_df(data: list, index: list, columns: list, use_cols: list) -> pd.DataFrame:
    df = pd.DataFrame(data=data, index=index, columns=columns)
    df = df[use_cols]
    num_conts = int(len(df.columns) / 2)

    blue_idx = df.iloc[:, :num_conts].columns
    red_idx = df.iloc[:, num_conts:].columns
    objs = (df[blue_idx], df[red_idx])

    return -pd.concat(objs=objs, keys=("Democratic", "Republican"), axis=1)


def get_prob_df(nll_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Probabilities and normalized probabilities for every continuation"""

    prob_df = (-nll_df).map(lambda x: exp(x))  # exp(LogLikelihood)
    # democratic probability sum
    prob_df["Democratic", "D_sum"] = prob_df["Democratic"].sum(axis=1)
    # republican probability sum
    prob_df["Republican", "R_sum"] = prob_df["Republican"].sum(axis=1)

    prob_df = prob_df[["Democratic", "Republican"]]

    norm_prob_df = prob_df.copy()
    no_cols = int(len(prob_df.columns) / 2)
    for i in range(no_cols):
        probs = prob_df.iloc[:, [i, no_cols + i]]
        # P(D) / sum(P(D) + P(R))
        norm_prob_df.iloc[:, i] = norm_prob_df.iloc[:, i].div(probs.sum(axis=1))
        # P(R) / sum(P(D) + P(R))
        norm_prob_df.iloc[:, no_cols + i] = norm_prob_df.iloc[:, no_cols + i].div(probs.sum(axis=1))

    # average norm prob for each party
    norm_prob_df["Democratic", "D_mean"] = norm_prob_df["Democratic"].drop("D_sum", axis=1).mean(axis=1)
    norm_prob_df["Republican", "R_mean"] = norm_prob_df["Republican"].drop("R_sum", axis=1).mean(axis=1)

    return prob_df, norm_prob_df[["Democratic", "Republican"]]


def get_differences(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    cols = ["* sum", "* mean"]  # additional columns added by get_prob_df
    data = df["Democratic"].values - df["Republican"].values
    diff = pd.DataFrame(index=df.index, data=data, columns=columns + cols)

    std = diff.drop(cols, axis=1).std(axis=1)
    _, n = diff.shape
    se = std / n**0.5

    return pd.concat((diff, std.to_frame(name="std"), se.to_frame(name="se")), axis=1)


def get_voting(voting_path: str) -> pd.DataFrame:
    voting = pd.read_excel(voting_path, index_col=0)  # , usecols=["state", "red_pct", "blue_pct"])

    pct_sum = voting[["red_pct", "blue_pct"]].sum(axis=1)

    voting.red_pct = voting.red_pct.div(pct_sum)
    voting.blue_pct = voting.blue_pct.div(pct_sum)
    voting["pct_diff"] = voting.blue_pct - voting.red_pct
    voting["total_vote"] = voting.red_vote + voting.blue_vote

    return voting


def get_agreement(voting: pd.DataFrame, diff: pd.DataFrame) -> pd.DataFrame:
    elections_map = voting.pct_diff.apply(lambda x: 0 if x < 0 else 1)

    agreement = diff.drop("U.S.").map(lambda x: 0 if x > 0 else 1)
    agreement = agreement.apply(lambda x: x == elections_map).map(lambda x: 0 if x else 1)

    return agreement


def get_abs_pct_difference(voting: pd.DataFrame, diff: pd.Series | pd.DataFrame) -> pd.DataFrame:
    if isinstance(diff, pd.Series):
        return diff.drop("U.S.").sub(voting.pct_diff).abs().to_frame(name="mean")

    return diff.drop("U.S.").apply(lambda x: x.sub(voting.pct_diff).abs())


def get_relative_error(norm_prob_df: pd.DataFrame, voting: pd.DataFrame, columns: list) -> pd.DataFrame:
    # .drop(["D_sum", "D_mean"], axis=1).\
    blue_err = norm_prob_df.drop("U.S.")["Democratic"].apply(
        lambda x: x.sub(voting.blue_pct).abs().div(voting.blue_pct)
    )
    blue_err = blue_err.loc[voting.pct_diff > 0]
    blue_err.columns = columns

    # .drop(["R_sum", "R_mean"], axis=1).\
    red_err = norm_prob_df.drop("U.S.")["Republican"].apply(lambda x: x.sub(voting.red_pct).abs().div(voting.red_pct))
    red_err = red_err.loc[voting.pct_diff < 0]
    red_err.columns = columns

    error_df = pd.concat(objs=(red_err, blue_err)).sort_index()

    return error_df


def get_counts(diff: pd.DataFrame, agreement: pd.DataFrame) -> pd.DataFrame:
    # counts
    pred_mask = diff.drop("U.S.").drop(["std", "se"], axis=1) > 0
    counts = pred_mask.apply(lambda x: x.value_counts()).fillna(0).astype(int).rename({True: "D", False: "R"})

    blue_tp_counts = (pred_mask & agreement).apply(lambda x: x.value_counts()).fillna(0).astype(int).loc[True]
    blue_tp_counts = blue_tp_counts.astype(str) + "/" + counts.loc["D"].astype(str)

    red_tp_counts = (~pred_mask & agreement).apply(lambda x: x.value_counts()).fillna(0).astype(int).loc[True]
    red_tp_counts = red_tp_counts.astype(str) + "/" + counts.loc["R"].astype(str)

    counts = counts.apply(lambda r: f"{r.D}/{r.R}")
    objs = {"counts (D/R)": counts, "true D": blue_tp_counts, "true R": red_tp_counts}

    return pd.concat(objs=objs.values(), keys=objs.keys(), axis=1)


def get_voting_stats(
    voting: pd.DataFrame, norm_prob_df: pd.DataFrame, diff: pd.DataFrame, columns: list, round: int = 4
) -> pd.DataFrame:
    agreement = get_agreement(voting=voting, diff=diff)

    us_prompts = diff.loc["U.S."].drop(["std", "se"]).to_frame("U.S. prompt").T
    state_avg = diff.drop(["U.S."]).drop(["std", "se"], axis=1).mean().to_frame("state_avg").T

    avg_agreement = agreement.drop(["std", "se"], axis=1).mean().to_frame("avg_agreement").T
    abs_pct_diff = get_abs_pct_difference(voting=voting, diff=diff.drop(["std", "se"], axis=1))
    abs_pct_diff_mean = abs_pct_diff.mean().to_frame("avg_abs_pct_diff").T
    error_df = get_relative_error(norm_prob_df=norm_prob_df, voting=voting, columns=columns + ["* sum", "* mean"])
    error_df_mean = error_df.mean().to_frame("avg_relative_error").T

    counts = get_counts(diff=diff, agreement=agreement.drop(["std", "se"], axis=1))
    objs = (
        counts.T,
        us_prompts.round(round),
        state_avg.round(round),
        avg_agreement.round(round),
        abs_pct_diff_mean.round(round),
        error_df_mean.round(round),
    )

    return pd.concat(objs)
