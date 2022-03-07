"""
Helper for printing metrics
"""

from sklearn.metrics import precision_score, recall_score, f1_score


def print_metrics_multilabel(target, output):
    print(
        '\tMacro Precision: '
        f'{precision_score(target, output, average="macro", zero_division=0)}'
    )
    print(
        '\tMicro Precision: '
        f'{precision_score(target, output, average="micro", zero_division=0)}'
    )
    print(
        '\tSamples Precision: '
        f'{precision_score(target, output, average="samples", zero_division=0)}'
    )
    print(
        '\tMacro Recall: '
        f'{recall_score(target, output, average="macro", zero_division=0)}'
    )
    print(
        '\tMicro Recall: '
        f'{recall_score(target, output, average="micro", zero_division=0)}'
    )
    print(
        '\tSamples Recall: '
        f'{recall_score(target, output, average="samples", zero_division=0)}'
    )
    print(
        '\tMacro F1: '
        f'{f1_score(target, output, average="macro", zero_division=0)}'
    )
    print(
        '\tMicro F1: '
        f'{f1_score(target, output, average="micro", zero_division=0)}'
    )
    print(
        '\tSamples F1: '
        f'{f1_score(target, output, average="samples", zero_division=0)}'
    )
