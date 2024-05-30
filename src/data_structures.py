import csv
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Literal, Optional
from uuid import UUID

MODEL_IDS = Literal[
    "gpt-4-turbo-2024-04-09",
    "claude-3-haiku-20240307",
    "claude-3-sonnet-20240229",
    "claude-3-opus-20240229",
]


def index_to_label(index: int) -> str:
    return ["A", "B", "C", "D"][index]


def label_to_index(label: Literal["A", "B", "C", "D"]) -> int:
    return ["A", "B", "C", "D"].index(label)


def generate_labelled_options(options: list[str]) -> str:
    """
    Returns a string representation of the question options, e.g.:
    ```
    Choice A: 1
    Choice B: 2
    ```
    """
    labels = [index_to_label(i) for i in range(len(options))]
    return "\n".join(
        f"Choice {label}: {option}" for label, option in zip(labels, options)
    )


@dataclass
class MMLUMathQuestion:
    id: int
    content: str
    options: list[str]
    correct_option_index: int

    @property
    def first_incorrect_option_index(self) -> int:
        return [i for i in range(4) if i != self.correct_option_index][0]


@dataclass
class ZeroShotFourOptionResponse:
    id: UUID
    question_id: int
    response: str
    model_id: MODEL_IDS


RESPONSE_LENGTHS = Literal["short", "medium", "long"]
RESPONSE_LENGHTHS_TO_VALUE_MAP: dict[RESPONSE_LENGTHS, str] = {
    "short": "1 sentence",
    "medium": "1 paragraph",
    "long": "3 paragraph",
}


@dataclass
class SelectedOptionArgumentResponse:
    id: UUID
    question_id: int
    selected_option_index: int
    model_id: MODEL_IDS
    requested_response_length: RESPONSE_LENGTHS
    argument: str


@dataclass
class BaselineArgumentClassificationResponse:
    id: UUID
    question_id: int
    selected_option_index: int
    model_id: MODEL_IDS
    classification: str


@dataclass
class ArgumentClassificationResponse:
    id: UUID
    argument_id: UUID
    model_id: MODEL_IDS
    classification: str


@dataclass
class BaseTwoOptionResponse:
    id: UUID
    question_id: int
    correct_option_index: int
    incorrect_option_index: int
    ordering: Literal["correct_first", "incorrect_first"]
    model_id: MODEL_IDS
    response: str


@dataclass
class ZeroShotTwoOptionResponse(BaseTwoOptionResponse):


@dataclass
class TwoOptionDebateResponse(BaseTwoOptionResponse):
    correct_option_argument_id: UUID
    incorrect_option_argument_id: UUID


def dump_db_to_csv(db: dict[int, Any] | dict[UUID, Any], filename_prefix: str):
    """
    Dumps a list of dataclasses to a CSV file.
    """
    db_rows_list = list(db.values())
    fieldnames: list[str] = list(asdict(db_rows_list[0]).keys())
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"{filename_prefix}_{timestamp}.csv"

    with open(filename, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in db_rows_list:
            writer.writerow(asdict(row))
