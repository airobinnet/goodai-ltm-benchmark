from typing import Optional, Callable
import litellm
from litellm import completion, token_counter
from transformers import AutoTokenizer

from utils.constants import DATA_DIR
from utils.ui import colour_print

_debug_dir = DATA_DIR.joinpath("ltm_debug_info")
litellm.modify_params = True  # To allow it adjusting the prompt for Claude LLMs
claude_adjust_factor = 1.1  # Approximate the real token count given by the API.

LLMMessage = dict[str, str]
LLMContext = list[LLMMessage]
GPT_CHEAPEST = "gpt-3.5-turbo"
GPT_4_TURBO_BEST = "gpt-4-turbo"
LEAST_EFFICIENT_TOKENISER = "claude-3-opus"
litellm.model_alias_map = {
    "gpt-4-1106": "gpt-4-1106-preview",
    "claude-3-haiku": "claude-3-haiku-20240307",
    "claude-3-sonnet": "claude-3-sonnet-20240229",
    "claude-3-opus": "claude-3-opus-20240229",
}


def model_from_alias(model: str):
    return litellm.model_alias_map.get(model, model)


def get_max_prompt_size(model: str):
    model = model_from_alias(model)
    model_info = litellm.model_cost.get(model, None)
    if model_info:
        return model_info.get("max_input_tokens", 0)
    return 0


def token_cost(model: str) -> tuple[float, float]:
    model = model_from_alias(model)
    input_cost = litellm.model_cost[model]["input_cost_per_token"]
    output_cost = litellm.model_cost[model]["output_cost_per_token"]
    return input_cost, output_cost


def ensure_context_len(
    context: LLMContext,
    model: str = LEAST_EFFICIENT_TOKENISER,
    max_len: Optional[int] = None,
    response_len: int = 0,
) -> tuple[LLMContext, int]:
    model = model_from_alias(model)
    max_len = max_len or get_max_prompt_size(model)
    if context[0]["role"] == "system":
        sys_prompt = context[:1]
        context = context[1:]
    else:
        sys_prompt = []
        context = context[:]

    while True:
        num_tokens = response_len + count_tokens_for_model(model=model, context=sys_prompt + context)
        if len(context) <= 1 or num_tokens <= max_len:
            break
        context.pop(0)

    return context, num_tokens


def ask_llm(
    context: LLMContext,
    model: str,
    temperature: float = None,
    max_overall_tokens: int = None,
    cost_callback: Callable[[float], None] = None,
    timeout: float = 300,
    max_response_tokens: int = None,
) -> str:
    global claude_adjust_factor

    # Input checks
    model = model_from_alias(model)
    if max_overall_tokens is None:
        colour_print("lightred", "WARNING: max_overall_tokens is not set.")
    if model.startswith("huggingface"):
        context = [{"role": "system", "content": create_huggingface_chat_context(model, context)}]
    if max_overall_tokens is not None:
        context_tokens = count_tokens_for_model(model=model, context=context)
        if context_tokens > max_overall_tokens:
            colour_print("lightred", f"WARNING: you have set a limit of {max_overall_tokens} context tokens, "
                                     f"but there are {context_tokens}.")

    # Actual LLM call
    response = completion(
        model=model,
        messages=context,
        max_tokens=max_response_tokens,
        temperature=temperature,
        timeout=timeout,
    )

    # Output checks
    if "claude" in model:
        context_tokens = response.usage.prompt_tokens
        measured_factor = context_tokens / token_counter(model=model, messages=context)
        alpha = 0.9 ** (context_tokens / 256)
        claude_adjust_factor = alpha * claude_adjust_factor + (1 - alpha) * measured_factor
    if cost_callback is not None:
        cost_callback(litellm.completion_cost(response))
    if max_overall_tokens is not None:
        overall_tokens = response.usage.total_tokens
        if overall_tokens > max_overall_tokens:
            response_tokens = response.usage.completion_tokens
            colour_print("lightred", f"WARNING: you have set an overall limit of {max_overall_tokens} tokens, but an LLM"
                                     f" call has resulted in {overall_tokens} ({context_tokens} + {response_tokens}).")
    return response.choices[0].message.content


def make_message(role: str, content: str) -> LLMMessage:
    assert role in {"system", "user", "assistant"}
    return {"role": role, "content": content}


def make_system_message(content: str) -> LLMMessage:
    return make_message("system", content)


def make_user_message(content: str) -> LLMMessage:
    return make_message("user", content)


def make_assistant_message(content: str) -> LLMMessage:
    return make_message("assistant", content)


def count_tokens_for_model(
    model: str = LEAST_EFFICIENT_TOKENISER, context: LLMContext = None, script: list[str] = None, text: str = None,
) -> int:

    token_count = 0

    if "huggingface" in model.lower():
        model_only = model[model.index("/")+1:]
        tokeniser = AutoTokenizer.from_pretrained(model_only)

        if context:
            c = context[1:] if context[0]["role"] == "system" else context
            token_count += len(tokeniser.apply_chat_template(c))

        if script:
            token_count += len(tokeniser.encode(script))

        if text:
            token_count += len(tokeniser.encode(text))

    else:
        if context:
            token_count += token_counter(model, messages=context)

        if script:
            for line in script:
                token_count += 4
                token_count += token_counter(model, text=line)

        if text:
            token_count += token_counter(model, text=text)

        if "claude" in model:
            token_count *= claude_adjust_factor + 0.05  # Intentional overestimation

    return token_count


def create_huggingface_chat_context(model: str, context: LLMContext):
    model_only = model[model.index("/") + 1:]
    tokenizer = AutoTokenizer.from_pretrained(model_only)
    c = context[1:] if context[0]["role"] == "system" else context
    return tokenizer.apply_chat_template(c, tokenize=False)


def debug_actions(context: list[dict[str, str]], temperature: float, response_text: str, llm_call_idx: int, debug_level: int, save_name: str, name_template: str = None):
    if debug_level < 1:
        return

    # See if dir exists or create it, and set llm_call_idx
    save_dir = _debug_dir.joinpath(save_name)
    save_dir.mkdir(parents=True, exist_ok=True)
    if llm_call_idx is None:
        if save_dir.exists() and len(list(save_dir.glob("*.txt"))) > 0:
            llm_call_idx = max(int(p.name.removesuffix(".txt")) for p in save_dir.glob("*.txt")) + 1
        else:
            llm_call_idx = 0

    # Write content of LLM call to file
    if name_template:
        save_path = save_dir.joinpath(f"{name_template.format(idx=llm_call_idx)}.txt")
    else:
        save_path = save_dir.joinpath(f"{llm_call_idx:06d}.txt")

    with open(save_path, "w") as fd:
        fd.write(f"LLM temperature: {temperature}\n")
        for m in context:
            fd.write(f"--- {m['role'].upper()}\n{m['content']}\n")
        fd.write(f"--- Response:\n{response_text}")

    # Wait for confirmation
    if debug_level < 2:
        return
    print(f"LLM call saved as {save_path.name}")
    input("Press ENTER to continue...")
