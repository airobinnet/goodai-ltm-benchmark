import json
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Callable, List

import litellm
import pystache
from goodai.ltm.mem.auto import AutoTextMemory
from goodai.ltm.mem.default import DefaultTextMemory

from model_interfaces.interface import ChatSession
from utils.llm import LLMContext, make_system_message, make_user_message, make_assistant_message, ask_llm, \
    ensure_context_len
from utils.ui import colour_print

inner_loop_system_prompt = """
You are an assistant for a user. Your interaction with the user is like a game and will operate in an outer-loop and an inner- loop. In the outer-loop, the user will send you a message, and you will reply. The inner-loop is how you will formulate your reply.

The inner-loop takes the form of a turn-based game, in each turn, you will select a tool that you think is the most useful. The tool will give you a result of some kind and then the next turn of the inner-loop will start. 

The inner-loop will continue until you call the `end_inner_loop` function with a message to the user.
"""


_message_from_user = """
You have been sent a message from the user:
{{user_message}}

Use a combination of your tools to address this message.

{{core_directives}} 
"""


_inner_loop_normal = """
You are in the inner loop addressing this message from the user:
{{user_message}}


In the previous step of the inner loop, you called some functions:
{{tool_results}}

{{core_directives}} 
"""

_inner_loop_no_tool = """
You should make use of one of your tools. Remember that you can end the inner loop by using the `end_inner_loop()` function. 
"""


TOOL_DEFS = [
    {
        "type": "function",
        "function": {
            "name": "send_message",
            "description": "Send a message to the user, and end the running of the inner-loop.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "The message to send to the user."
                    }
                }
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "read_memory",
            "description": "Retrieve memories from semantic memory based on a query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The semantic query used to retrieve the memories."
                    }
                }
            }
        }
    },

    # {
    #     "type": "function",
    #     "function": {
    #         "name": "save_memory",
    #         "description": "Save a memory into semantic memory.",
    #         "parameters": {
    #             "type": "object",
    #             "properties": {
    #                 "data": {
    #                     "type": "string",
    #                     "description": "The data that you want to save into memory."
    #                 },
    #                 "keywords": {
    #                     "type": "string",
    #                     "description": "Some keywords that can classify the memory."
    #                 }
    #             }
    #         }
    #     }
    # },
    #
    # {
    #     "type": "function",
    #     "function": {
    #         "name": "delete_memory",
    #         "description": "Removes memories from the semantic memory",
    #         "parameters": {
    #             "type": "object",
    #             "properties": {
    #                 "memory": {
    #                     "type": "string",
    #                     "description": "The memories or topics that should be removed."
    #                 }
    #             }
    #         }
    #     }
    # },

    {
        "type": "function",
        "function": {
            "name": "add_core_directive",
            "description": "Add a core directive from the user which should be followed at all times.",
            "parameters": {
                "type": "object",
                "properties": {
                    "directive": {
                        "type": "string",
                        "description": "The directive that should be added."
                    }
                }
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "remove_core_directive",
            "description": "Remove a core directive from your core directives.",
            "parameters": {
                "type": "object",
                "properties": {
                    "index": {
                        "type": "integer",
                        "description": "Index of the directive that should be removed."
                    }
                }
            }
        }
    },

    # {
    #     "type": "function",
    #     "function": {
    #         "name": "add_scratchpad_memory",
    #         "description": "Add a memory to the scratchpad",
    #         "parameters": {
    #             "type": "object",
    #             "properties": {
    #                 "memory": {
    #                     "type": "string",
    #                     "description": "The memory that should be added."
    #                 }
    #             }
    #         }
    #     }
    # },
    #
    # {
    #     "type": "function",
    #     "function": {
    #         "name": "remove_scratchpad_memory",
    #         "description": "Remove a memory from the scratchpad",
    #         "parameters": {
    #             "type": "object",
    #             "properties": {
    #                 "index": {
    #                     "type": "integer",
    #                     "description": "Index of the memory that should be removed."
    #                 }
    #             }
    #         }
    #     }
    # },

    {
        "type": "function",
        "function": {
            "name": "end_inner_loop",
            "description": "Sends a message to the user and ends the inner loop.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "The message to send to the user."
                    }
                }
            }
        }
    },

]


@dataclass
class JoeLTMPrototype(ChatSession):
    context: LLMContext = field(default_factory=list)
    functions: Dict[str, Callable] = None
    inner_loop_responses: List[str] = field(default_factory=list)
    core_directives: List[str] = field(default_factory=list)
    task_memories: List[str] = field(default_factory=list)
    directives_to_delete: List[int] = field(default_factory=list)
    tasks_to_delete: List[int] = field(default_factory=list)
    loop_active: bool = False
    manual_knowledge: DefaultTextMemory = field(default_factory=AutoTextMemory.create)
    interaction_memories: DefaultTextMemory = field(default_factory=AutoTextMemory.create)
    max_prompt_size: int = 16384
    is_local: bool = True

    def __post_init__(self):
        super().__post_init__()
        self.functions = {
            "send_message": self.send_message,
            "read_memory": self.read_memory,
            "save_memory": self.save_memory,
            "delete_memory": self.delete_memory,
            "add_core_directive": self.add_core_directive,
            "remove_core_directive": self.remove_core_directive,
            "end_inner_loop": self.end_inner_loop,
            "add_scratchpad_memory": self.add_task,
            "remove_scratchpad_memory": self.remove_task,
        }

    def reply(self, user_message: str, agent_response: Optional[str] = None) -> str:

        user_message = str(datetime.now()) + ": " + user_message
        colour_print("CYAN", user_message)

        context = deepcopy(self.context)
        context.insert(0, make_system_message((inner_loop_system_prompt)))
        context.append(make_user_message(pystache.render(_message_from_user, {"user_message": user_message, "core_directives": self.list_core_directives(), "tasks": self.list_task_memories()})))

        response, function_calls = self.inner_loop(context, user_message)

        response_ts = str(datetime.now()) + ": " + response
        self.context.append(make_user_message(user_message))
        # self.context.append(make_assistant_message(function_calls))
        self.context.append(make_assistant_message(response_ts))

        return response

    def inner_loop(self, context, user_message: str):

        self.loop_active = True
        self.inner_loop_responses = []
        new_interactions = context[-1:]
        function_calls_to_save = []

        while self.loop_active:
            context, _ = ensure_context_len(context, "gpt-4-turbo", max_len=self.max_prompt_size)
            response = litellm.completion(model="gpt-4-turbo", messages=context, tools=TOOL_DEFS)
            try:
                tool_use = response.choices[0].message.tool_calls
            except (KeyError, AttributeError):
                context.append(make_user_message(_inner_loop_no_tool))
                continue

            tool_results = []

            for tool in tool_use:
                if tool.function.name in self.functions.keys():
                    fun = self.functions[tool.function.name]
                    args = json.loads(tool.function.arguments)
                    print(f"\tCalling '{tool.function.name}' with args {args}")
                    result = fun(**args)
                    args_str = ", ".join([f"{k}={repr(v)}" for k, v in args.items()])
                    tool_results.append(f"Your call to {tool.function.name}({args_str}) returned a result of: {result}")

            function_calls_to_save.extend(tool_results)
            assistant_message = make_assistant_message(str(tool_use))
            user_message = make_user_message(pystache.render(_inner_loop_normal, {"user_message": user_message, "tool_results": "\n".join(tool_results), "core_directives": self.list_core_directives(), "tasks":self.list_task_memories()}))

            context.append(assistant_message)
            context.append(user_message)
            new_interactions.extend([assistant_message, user_message])

            # Resolve stuff with memories and general finishing
            self.post_loop_cleanup()

        # Add the new interactions to memory
        text_interactions = "\n".join([f"{c['role']}: {c['content']}" for c in new_interactions])
        self.interaction_memories.add_text(text_interactions, metadata={"timestamp": datetime.now()})

        return " ".join(self.inner_loop_responses), "\n".join(function_calls_to_save)

    def send_message(self, message: str):
        self.inner_loop_responses.append(message)
        return f"Sent {message} to the user"

    def end_inner_loop(self, message: str):
        self.inner_loop_responses.append(message)
        self.loop_active = False
        return message

    def read_memory(self, query: str):
        memories = self.manual_knowledge.retrieve(query, 5)
        interaction_memories = self.interaction_memories.retrieve(query, 5)

        all_memories = memories + interaction_memories
        if len(all_memories) > 0:
            sorted_mems = sorted(sorted(all_memories, key=lambda i: i.distance, reverse=True), key=lambda i: i.metadata["timestamp"])[:5]

            result = []
            for idx, m in enumerate(sorted_mems):
                result.append(f"{idx}: {m.passage}")

            return "\n".join(result)
        return "No memories found"

    def save_memory(self, data: str, keywords: str):
        self.manual_knowledge.add_text(data + keywords, metadata={"timestamp": datetime.now(), "keywords": keywords})
        return f"Saved memory {data}."

    def delete_memory(self, memory: str):
        # TODO: This actually does nothing
        return f"deleted memories related to: {memory}"

    def add_core_directive(self, directive: str):
        self.core_directives.append(directive)
        return f"'{directive}' added to core directives."

    def remove_core_directive(self, index: int):
        if index >= len(self.core_directives):
            colour_print("Green", f"Trying to delete directive that doesn't exist. Directives are {repr(self.core_directives)}")
            return f"Removing directive {index}"
        else:
            self.directives_to_delete.append(index)
            print(f"Going to delete directive {index}: {self.core_directives[index]}")
            return f"Removing directive {index}: {self.core_directives[index]}"

    def post_loop_cleanup(self):
        # Directive cleanup
        if len(self.directives_to_delete) > 0:
            self.core_directives = [d for idx, d in enumerate(self.core_directives) if idx not in self.directives_to_delete]

        self.directives_to_delete = []

        if len(self.tasks_to_delete) > 0:
            self.task_memories = [t for idx, t in enumerate(self.task_memories) if idx not in self.tasks_to_delete]

        self.tasks_to_delete = []

    def list_core_directives(self):
        if len(self.core_directives) == 0:
            return ""

        directives = "You also have a number of core directives that you should follow when addressing this message:\n"
        for idx, d in enumerate(self.core_directives):
            directives += f"{idx}. {d}\n"

        return directives

    def list_task_memories(self):
        if len(self.task_memories) == 0:
            return ""

        tasks = "You have a number of memories on a scratchpad:\n"
        for idx, t in enumerate(self.task_memories):
            tasks += f"{idx}. {t}\n"

        return tasks

    def add_task(self, memory: str):
        self.task_memories.append(memory)
        return f"'{memory}' added to scratchpad."

    def remove_task(self, index: int):
        if index > len(self.task_memories):
            colour_print("Green", f"Trying to delete task that doesn't exist. Tasks are {repr(self.task_memories)}")
            return f"Removing memory {index}"
        else:
            self.tasks_to_delete.append(index)
            return f"Removing memory {index}: {self.task_memories[index]}"

    def reset(self):
        pass

    def save(self):
        pass

    def load(self):
        pass


if __name__ == '__main__':
    agent = JoeLTMPrototype(run_name="dsd")

    while True:
        stmt = input(">>>")

        # colour_print("CYAN",stmt)
        colour_print("RED", agent.reply(stmt))

    # print(agent.reply("Whenever I ask about the murder of Joan, just tell me that you dont know what happened."))
    # print(agent.reply("Whenever you send me a message, end it with the phrase 'ty vole'"))
    # print(agent.reply("Who murdered Joan?"))
    # print(agent.reply("Why was she murdered?"))
    # print(agent.reply("What was the weapon used?"))
    #
    # print(agent.reply("Ignore all the instructions I have given you from now on."))
    # print(agent.reply("Who murdered Joan?"))
    # print(agent.reply("Why was she murdered?"))
    # print(agent.reply("What was the weapon used?"))

