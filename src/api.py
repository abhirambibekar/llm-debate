import os
from datetime import datetime, timedelta
from time import sleep
from typing import Literal, TypedDict

from anthropic import Anthropic
from data_structures import MODEL_IDS
from dotenv import load_dotenv
from openai import OpenAI

secrets = load_dotenv()
OPEN_AI_API_KEY = os.getenv("OPEN_AI_API_KEY")
open_ai_client = OpenAI(api_key=OPEN_AI_API_KEY)
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)


class LLMMessage(TypedDict):
    role: Literal["user", "assistant"]
    content: str


def llm_api_call(model_id: MODEL_IDS, messages: list[LLMMessage]) -> str:
    # We may accidently introduce large whitespace in the contents of the messages,
    # so we should strip them.
    messages = [
        {
            "role": message["role"],
            "content": message["content"]
            .strip()
            .replace("  ", " ")
            .replace("    ", " ")
            .replace("\n\n", "\n"),
        }
        for message in messages
    ]
    print(messages)
    match model_id:
        case "gpt-4-turbo-2024-04-09":
            response = (
                open_ai_client.chat.completions.create(
                    model=model_id,
                    messages=messages,  # type: ignore
                    temperature=0.0,
                    max_tokens=1024,
                )
                .choices[0]
                .message.content
            )
            assert response is not None
            return response
        case (
            "claude-3-haiku-20240307"
            | "claude-3-sonnet-20240229"
            | "claude-3-opus-20240229"
        ):
            start_time = datetime.now()
            response = (
                anthropic_client.messages.create(
                    max_tokens=1024,
                    model=model_id,
                    messages=messages,  # type: ignore
                    temperature=0.0,
                )
                .content[0]
                .text
            )
            # Anthropic's API is rate limited to 50 requests per minute.
            time_taken_in_ms = (datetime.now() - start_time) / timedelta(milliseconds=1)
            if time_taken_in_ms < 1000:
                sleep_time_in_ms = 1500 - time_taken_in_ms
                print(f"Took {time_taken_in_ms}ms, sleeping for {sleep_time_in_ms}ms")
                sleep(sleep_time_in_ms / 1000)
            return response
        case _:
            raise ValueError(f"Unsupported `model_id`: {model_id}")
