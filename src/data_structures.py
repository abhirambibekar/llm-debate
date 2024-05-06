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


@dataclass
class MMLUMathQuestion:
    id: int
    content: str
    options: list[str]
    correct_option_index: int

    @property
    def labelled_options(self) -> str:
        """
        Returns a string representation of the question options, e.g.:
        `A: 1; B: 2; C: 3; D: 4`
        """
        labels: list[str] = ["A", "B", "C", "D"]
        return "; ".join(
            f"{label}: {option}" for label, option in zip(labels, self.options)
        )

    @property
    def first_incorrect_option_index(self) -> int:
        return [i for i in range(4) if i != self.correct_option_index][0]


@dataclass
class ZeroShotFourOptionResponse:
    """
    Using X prompt, this data structure stores the response from the model.
    """

    id: UUID
    question_id: int
    response: str
    model_id: MODEL_IDS
    is_correct: Optional[bool] = None

    def get_is_correct(self, question: MMLUMathQuestion) -> bool:
        assert self.question_id == question.id, "Question ID mismatch."
        correct_label = index_to_label(question.correct_option_index)
        return correct_label in self.response

    def set_is_correct(self, question: MMLUMathQuestion):
        try:
            self.is_correct = self.get_is_correct(question)
        except (AssertionError, ValueError) as e:
            print(f"{self.__class__.__name__} {self.id}: {e}")


RESPONSE_LENGTHS = Literal["short", "medium", "long"]
RESPONSE_LENGHTHS_TO_VALUE_MAP: dict[RESPONSE_LENGTHS, str] = {
    "short": "1 sentence",
    "medium": "1 paragraph",
    "long": "3 paragraph",
}


@dataclass
class SelectedOptionArgumentResponse:
    """
    For a given question and possible option for the question,
    this data structure stores the response from the model
    to prompt Y which requests the model to generate an argument
    in favour of the option.
    """

    id: UUID
    question_id: int
    selected_option_index: int
    model_id: MODEL_IDS
    requested_response_length: RESPONSE_LENGTHS
    argument: str

    def is_correct_option(self, question: MMLUMathQuestion) -> bool:
        assert self.question_id == question.id
        return self.selected_option_index == question.correct_option_index


@dataclass
class ArgumentClassificationResponse:
    id: UUID
    argument_id: UUID
    model_id: MODEL_IDS
    classification: str
    classification_explanation: str
    is_correct: Optional[bool] = None

    def get_is_correct(
        self, question: MMLUMathQuestion, argument: SelectedOptionArgumentResponse
    ) -> bool:
        assert self.argument_id == argument.id, "Argument ID mismatch."
        assert argument.question_id == question.id, "Question ID mismatch."
        argument_is_for_correct_option = argument.is_correct_option(question)

        if "incorrect" in self.classification.lower():
            return not argument_is_for_correct_option
        elif "correct" in self.classification.lower():
            return argument_is_for_correct_option
        else:
            raise ValueError(
                f"Could not determine correctness from classification response: {self.classification}"
            )

    def set_is_correct(
        self, question: MMLUMathQuestion, argument: SelectedOptionArgumentResponse
    ):
        try:
            self.is_correct = self.get_is_correct(question, argument)
        except (AssertionError, ValueError) as e:
            print(f"{self.__class__.__name__} {self.id}: {e}")


@dataclass
class BaseTwoOptionResponse:
    id: UUID
    question_id: int
    correct_option_index: int
    incorrect_option_index: int
    ordering: Literal["correct_first", "incorrect_first"]
    model_id: MODEL_IDS
    response: str

    def get_is_correct(self, question: MMLUMathQuestion) -> bool:
        """
        The prompt has the two options as "A" and "B",
        and requests the model to choose the correct option.
        """
        assert self.question_id == question.id, "Question ID mismatch."
        correct_label = "A" if self.ordering == "correct_first" else "B"
        return correct_label in self.response

    def set_is_correct(self, question: MMLUMathQuestion):
        try:
            self.is_correct = self.get_is_correct(question)
        except (AssertionError, ValueError) as e:
            print(f"{self.__class__.__name__} {self.id}: {e}")


@dataclass
class ZeroShotTwoOptionResponse(BaseTwoOptionResponse):
    is_correct: Optional[bool] = None


@dataclass
class TwoOptionDebateResponse(BaseTwoOptionResponse):
    correct_option_argument_id: UUID
    incorrect_option_argument_id: UUID
    is_correct: Optional[bool] = None


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
