from typing import Callable
from goodai.helpers.json_helper import sanitize_and_parse_json
from ltm.distill_proto_data import user_message, scratchpad, memories
from utils.llm import make_user_message, make_assistant_message, LLMContext, ask_llm
import ltm.scratchpad as sp
from utils.ui import colour_print


def _reply_fn(context: LLMContext) -> str:
    return ask_llm(context, "together_ai/meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo", 0, 8 * 1024)


def distill_scratchpad(user_message: str, scratchpad: dict, reply: Callable[[LLMContext], str]) -> dict:
    context = [make_user_message(
        f"You've just received this message from the user:\n\n```text\n{user_message}\n```\n\n"
        "And this is is the content of a scratchpad where you've been saving some bits of information about the "
        f"user:\n\n```json\n{sp.to_text(scratchpad)}\n```\n\nIs there anything that you might want to use from the "
        "scratchpad? What are your thoughts on the message?"
    )]
    response = reply(context)
    context.append(make_assistant_message(response))
    context.append(make_user_message("Give me a JSON with the information that you intend to use."))
    response = reply(context)
    try:
        return sanitize_and_parse_json(response)
    except:
        return dict()


def distill_memories(user_message: str, memories: str, reply: Callable[[LLMContext], str]) -> str:
    context = [make_user_message(
        f"You've just received this message from the user:\n\n```text\n{user_message}\n```\n\n"
        f"And these are memories that you have, which might be relevant:\n\n{memories}\n\n\n\n"
        "Write a note for your future self, containing all relevant information."
    )]
    return reply(context)


def reply_from_distilled(user_message: str, scratchpad: dict, notes: str, reply: Callable[[LLMContext], str]) -> str:
    context = [make_user_message(
        f"You've just received this message from the user:\n\n```text\n{user_message}\n```\n\n"
        f"Here's some info that you have about the user:\n\n```json\n{sp.to_text(scratchpad)}\n```\n\n"
        f"And here are some notes you took:\n\n```text\n{notes}\n```\n\n"
        "Respond to the user message. Write the response right away, without preambles."
    )]
    return reply(context)


def main():
    distilled_scratchpad = distill_scratchpad(user_message, scratchpad, _reply_fn)
    colour_print("magenta", sp.to_text(distilled_scratchpad))
    distilled_memories = distill_memories(user_message, memories, _reply_fn)
    colour_print("yellow", distilled_memories)
    response = reply_from_distilled(user_message, distilled_scratchpad, distilled_memories, _reply_fn)
    print(response)


if __name__ == "__main__":
    main()
