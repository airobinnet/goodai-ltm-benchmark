import ltm.scratchpad as sp
from dataclasses import dataclass
from typing import Callable
from datetime import datetime
from ltm.agent import LTMAgent
from utils.ui import colour_print
from utils.llm import LLMContext, make_user_message, make_assistant_message
from goodai.ltm.agent import Message
from ltm.distill_proto import distill_scratchpad, distill_memories, reply_from_distilled


CostCallback = Callable[[float], None]


@dataclass
class LTMReflexionAgent(LTMAgent):

    def reply(self, user_content: str, cost_callback: Callable[[float], None] = None) -> str:
        self.now = datetime.now()
        session = self.session
        context = self._build_llm_context(session.message_history, user_content,
                                          cost_callback)
        response = self._completion(context, temperature=self.config.llm_temperature, label="reply",
                                    cost_callback=cost_callback)
        for role, content in [("user", user_content), ("assistant", response)]:
            message = Message(role=role, content=content, timestamp=self.now.timestamp())
            session.add(message)
            self._add_to_convo_memory(message)
        return response

    def _distill_scratchpad(self, user_message: str, cost_cb: CostCallback) -> dict:
        reply = lambda ctx: self._completion(ctx, 0, "distill_scratchpad", cost_cb)
        return distill_scratchpad(user_message, self.user_info, reply)

    def _distill_memories(self, user_message: str, memories: str, cost_cb: CostCallback) -> str:
        reply = lambda ctx: self._completion(ctx, 0, "distill_memories", cost_cb)
        return distill_memories(user_message, memories, reply)

    def _truncated_completion(self, context: LLMContext, max_messages: int = None, **kwargs) -> str:
        max_messages = max_messages or len(context) + 1
        return super()._truncated_completion(context, max_messages=max_messages, **kwargs)

    def _reflect(self, context: LLMContext, response: str, cost_callback: Callable[[float], None] = None) -> str:
        context = context[:]
        reflect = lambda q: self._elaborate(context, q, cost_callback=cost_callback)
        user_content = context[-1]["content"]
        colour_print("lightblue", response)
        context.append(make_assistant_message(response))
        r = reflect(
            "In relation to your last response, what information do you have that is related to it? Give me just "
            "a short list of summarized points. A very short text.",
        )
        colour_print("lightyellow", r)
        r = reflect("Did you use that information effectively? Is there anything that you missed?")
        colour_print("lightblue", r)
        r = reflect("Based on this reflection, would you like to change or improve your response? Answer simply 'yes' or 'no'.")
        colour_print("lightyellow", r)
        if "yes" in r.lower():
            response = reflect(f"Alright. Let's try again then.\n(going back to the original message... act as if nothing happened)\n\n{user_content}")
            colour_print("lightblue", response)
        return response

    def _elaborate(self, context: LLMContext, question: str, cost_callback: Callable[[float], None] = None) -> str:
        context.append(make_user_message(question))
        reflection = self._truncated_completion(
            context, temperature=self.config.llm_temperature, label="reflection", cost_callback=cost_callback,
        )
        context.append(make_assistant_message(reflection))
        return reflection
