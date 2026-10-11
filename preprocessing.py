"""Prepare a local dataset from ticket headers and the first user message."""

import argparse
import re
from pathlib import Path

import pandas as pd

from io_utils import read_dataset, require_columns

PROJECT_ROOT = Path(__file__).resolve().parent


def clean_text(text):
    text = re.sub(r"\n+", "\n", str(text)).strip()
    return re.sub(r" {2,}", " ", text)


def prepare_dataset(tickets, messages, min_text_len=10):
    require_columns(tickets, ["ticket_id", "subject"])
    require_columns(messages, ["ticket_id", "ticket_message", "ticket_message_author"])
    user_messages = messages.loc[messages["ticket_message_author"] == "User"]
    first = user_messages.drop_duplicates("ticket_id", keep="first")
    frame = pd.merge(
        first[["ticket_id", "ticket_message"]],
        tickets[["ticket_id", "subject"]],
        on="ticket_id",
        validate="many_to_one",
    )
    frame["text"] = (
        frame["subject"].fillna("").astype(str)
        + "\n"
        + frame["ticket_message"].fillna("").astype(str)
    ).map(clean_text)
    return frame.loc[frame["text"].str.len() > min_text_len, ["ticket_id", "text"]].reset_index(
        drop=True
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickets", type=Path, default=PROJECT_ROOT / "datasets/tickets_anon.csv")
    parser.add_argument(
        "--messages", type=Path, default=PROJECT_ROOT / "datasets/ticket_messages_anon.csv"
    )
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "datasets/cleaned_dataset.csv"
    )
    parser.add_argument(
        "--sample-output", type=Path, default=PROJECT_ROOT / "datasets/sample_dataset.csv"
    )
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--min-text-len", type=int, default=10)
    args = parser.parse_args()
    if args.sample_size < 1 or args.min_text_len < 0:
        parser.error("Sample size must be positive; minimum text length must be non-negative.")
    frame = prepare_dataset(
        read_dataset(args.tickets), read_dataset(args.messages), args.min_text_len
    )
    if frame.empty:
        raise ValueError("No tickets remain after filtering.")
    for target in (args.output, args.sample_output):
        target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False, encoding="utf-8-sig")
    frame.sample(n=min(args.sample_size, len(frame)), random_state=42).to_csv(
        args.sample_output, index=False, sep=";", encoding="utf-8-sig"
    )
    print(f"Prepared {len(frame)} tickets; sample has {min(args.sample_size, len(frame))} rows.")


if __name__ == "__main__":
    main()
