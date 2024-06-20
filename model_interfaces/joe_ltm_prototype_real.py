import json
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Callable, List

import litellm
import pystache
from goodai.helpers.json_helper import sanitize_and_parse_json
from goodai.ltm.mem.auto import AutoTextMemory
from goodai.ltm.mem.config import TextMemoryConfig
from goodai.ltm.mem.default import DefaultTextMemory

from model_interfaces.interface import ChatSession
from utils.llm import LLMContext, make_system_message, make_user_message, make_assistant_message, ask_llm, \
    ensure_context_len
# from utils.similarity_matcher import SimilarityMatcher
from utils.ui import colour_print

# litellm.set_verbose=True

inner_loop_system_prompt = """
You are an assistant for a user. Your interaction with the user is like a game and will operate in an outer-loop and an inner-loop.
In the outer-loop, the user will send you a message, and you will reply. The inner-loop is how you will formulate your reply.

The inner-loop takes the form of a turn-based game, in each turn, you will select a tool that you think is the most useful.
The tool will give you a result of some kind and then the next turn of the inner-loop will start. 

The inner-loop will continue until you call the `end_inner_loop` tool with a message to the user.
"""

_message_from_user = """
*************************************************************************************
You have been sent a message from the user:
{{user_message}}

Use a combination of your tools to address this message.
The messages above are informational, they have already been addressed and replied to.
You should address and reply to this current message.
"""

_inner_loop_plan = """
Create a plan for the next step of addressing the above user message using one of the tools that you have available to you.
"""


_inner_loop_call = """
Choose a tool to call that follows your plan above.
"""


# _inner_loop_normal = """
# You are in the inner loop addressing this message from the user:
# {{user_message}}
#
# {{core_directives}}
# """

# _inner_loop_normal = """
# *************************************************************************************
# You are in the inner loop addressing this message from the user:
# {{user_message}}
#
# Use a combination of your tools to address this message.
# The messages above are for context only. You should address this message.
#
# **Prior** Each message above this one has been saved to memory.
# **Prior** Save only important memories.
# """


# _read_memory_loop = """
# You are reading from a vector database to satisfy a query.
#
# The original query is: {{original_query}}.
#
# Read from the vector database and consolidate the memories into a useful summary. Perform multiple queries if appropriate.
# Keywords will narrow down search results to the keyword topics. If there are no results at all, then the memory is empty.
#
# Each memory retrieval will return a list of memories if in order from most recent to least recent.
# Pay attention to these recency timestamps, ignore duplicates, and resolve conflicting memory states in favour of the most recent memory.
#
# Keywords that have been defined which you can use to narrow your search query. Choose relevant ones:
# {{keywords}}
# """

_read_memory_loop = """
You are reading from a vector database to satisfy a query.

The original query is: {{original_query}}.

Read from the vector database and consolidate the memories into a useful summary.
Keywords will narrow down search results to the keyword topics. Choose at most 5 keywords. 
 
Each run see if the memories are relevant, if there are no relevant memories at all, then the topic is not in memory.

Keywords that have been defined which you can use to narrow your search query. Choose relevant ones:
{{keywords}}
"""

_read_memory_plan = """
Given the memories you have retrieved, and the original query, what should the next step be.
Do not ask questions, address the memories and the query.  
"""


_delete_memory_loop = """
You need to delete memories from a database relating to a topic 

The topic is: {{topic}}.

Read from the vector database and identify which memories should be deleted.
Keywords will narrow down search results to the keyword topics. If there are no results at all, then the memory is empty.

Each memory retrieval will return a list of memories if in order from most recent to least recent. 
"""

_choose_to_delete = """
Above is a numbered list of memories. For each memory you should decide whether it is related strongly enough to the current topic.
Which is: {{topic}}

If it is related enough, then you cna choose to delete it. Formulate your answer as a list of integers in JSON like:  

```json
[0, 2, 3, 6]
```
"""

_state_reconstruction_prompt = """
You will be shown a series of memories one at a time, and what the current state is.
For each of these memories, integrate it into the current state, updating or replacing the state as necessary.
Write the state as a JSON object for clarity.
"""

TOOL_READ_MEMORY_LOOP = [
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
                },
                "keywords": {
                    "type": "string",
                    "description": "Comma separated keywords to filter memories."
                }
            },
            "required": ["query", "keywords"]
        }
    }
},

{
    "type": "function",
    "function": {
        "name": "done",
        "description": "Finish reading memory and return results.",
        "parameters": {
            "type": "object",
            "properties": {
                "results": {
                    "type": "string",
                    "description": "Results from the read memories that you want to return"
                }
            },
            "required": ["results"]
        }
    }
},

]


TOOL_DEFS = [
    # {
    #     "type": "function",
    #     "function": {
    #         "name": "send_message",
    #         "description": "Send a message to the user, and end the running of the inner-loop.",
    #         "parameters": {
    #             "type": "object",
    #             "properties": {
    #                 "message": {
    #                     "type": "string",
    #                     "description": "The message to send to the user."
    #                 }
    #             }
    #         }
    #     }
    # },

    {
        "type": "function",
        "function": {
            "name": "read_from_memory",
            "description": "Retrieve memories from semantic memory based on a query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The semantic query used to retrieve the memories."
                    }
                },
                "required": ["query"]
            }
        }
    },

    # {
    #     "type": "function",
    #     "function": {
    #         "name": "save_memory",
    #         "description": "Save a memory into semantic memory. Save information from the current user message.",
    #         "parameters": {
    #             "type": "object",
    #             "properties": {
    #                 "data": {
    #                     "type": "string",
    #                     "description": "The data that you want to save into memory."
    #                 },
    #                 "keywords": {
    #                     "type": "string",
    #                     "description": "Comma separated keywords that can classify the memory."
    #                 }
    #             },
    #             "required": ["data", "keywords"]
    #         }
    #     }
    # },

    # {
    #     "type": "function",
    #     "function": {
    #         "name": "delete_memory",
    #         "description": "Removes memories from the semantic memory related to a topic",
    #         "parameters": {
    #             "type": "object",
    #             "properties": {
    #                 "topic": {
    #                     "type": "string",
    #                     "description": "The topic used to remove the memories."
    #                 }
    #             }
    #         }
    #     }
    # },

    # {
    #     "type": "function",
    #     "function": {
    #         "name": "add_core_directive",
    #         "description": "Add a core directive from the user which should be followed at all times.",
    #         "parameters": {
    #             "type": "object",
    #             "properties": {
    #                 "directive": {
    #                     "type": "string",
    #                     "description": "The directive that should be added."
    #                 }
    #             }
    #         }
    #     }
    # },
    #
    # {
    #     "type": "function",
    #     "function": {
    #         "name": "remove_core_directive",
    #         "description": "Remove a core directive from your core directives.",
    #         "parameters": {
    #             "type": "object",
    #             "properties": {
    #                 "index": {
    #                     "type": "integer",
    #                     "description": "Index of the directive that should be removed."
    #                 }
    #             }
    #         }
    #     }
    # },

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
                },
                "required": ["message"]
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
    # matcher: SimilarityMatcher = field(default_factory=SimilarityMatcher)
    defined_kws: set = field(default_factory=set)

    def __post_init__(self):
        super().__post_init__()
        self.functions = {
            # "send_message": self.send_message,
            "read_memory": self.perform_memory_read,
            "save_memory": self.save_memory,
            "delete_memory": self.delete_memories_loop,
            # "add_core_directive": self.add_core_directive,
            # "remove_core_directive": self.remove_core_directive,
            "end_inner_loop": self.end_inner_loop,
            # "add_scratchpad_memory": self.add_task,
            # "remove_scratchpad_memory": self.remove_task,
            "read_from_memory": self.read_from_memory,
            "done": self.done,
        }
        self.manual_knowledge = AutoTextMemory.create(config=TextMemoryConfig(chunk_capacity=200, chunk_overlap_fraction=0.0))

        self.is_local = True
        self.max_message_size = 1000

    def dump_context(self, context):
        for message in context:
            colour_print("YELLOW", f"{message['role']}: {message['content']}\n")

    def reply(self, user_message: str, agent_response: Optional[str] = None) -> str:

        mem_user_message = str(datetime.now()) + "(User): " + user_message

        user_message = str(datetime.now()) + ": " + user_message
        colour_print("CYAN", f"DEALING WITH USER MESSAGE: {user_message}")

        context = deepcopy(self.context)
        context.insert(0, make_system_message(inner_loop_system_prompt))
        # context = [make_system_message(inner_loop_system_prompt)]
        context.append(make_user_message(pystache.render(_message_from_user, {"user_message": user_message, "core_directives": self.list_core_directives(), "tasks": self.list_task_memories()})))

        response = self.inner_loop(context, user_message)
        mem_agent_message = str(datetime.now()) + "(Agent): " + response

        response_ts = str(datetime.now()) + ": " + response
        self.context.append(make_user_message(user_message))
        # self.context.append(make_assistant_message(function_calls))
        self.context.append(make_assistant_message(response_ts))
        self.save_passive_memory(mem_user_message + "\n" + mem_agent_message)
        self.context, _ = ensure_context_len(self.context, "gpt-4o", max_len=self.max_prompt_size)

        return response

    def inner_loop(self, context, user_message: str):

        self.loop_active = True
        self.inner_loop_responses = []
        new_interactions = context[-1:]
        context.append(make_user_message(_inner_loop_plan))

        while self.loop_active:
            # Make a plan for the next step
            # colour_print("Yellow", f"Attempting Planning call:")
            # self.dump_context(context)
            plan_response = litellm.completion(model="gpt-4o", messages=context, tools=TOOL_DEFS, tool_choice="none")
            context.append(make_assistant_message(plan_response.choices[0].message.content))

            colour_print("GREEN", f"Plan is: {plan_response.choices[0].message.content}")
            context.append(make_user_message(_inner_loop_call))

            # Perform your tool calls
            context, _ = ensure_context_len(context, "gpt-4o", max_len=self.max_prompt_size)
            # colour_print("Yellow", f"Attempting Tool call with:")
            # self.dump_context(context)
            response = litellm.completion(model="gpt-4o", messages=context, tools=TOOL_DEFS, tool_choice="required")
            print(f"LLM call with: {response.choices[0].message.model_extra}")
            tool_use = response.choices[0].message.tool_calls
            success, new_context = self.use_tools(tool_use)

            if not success:
                continue

            context.append(response.choices[0].message.model_extra)
            context.extend(new_context)

            # Resolve stuff with memories and general finishing
            self.post_loop_cleanup()

        # Add the new interactions to memory
        # text_interactions = "\n".join([f"{c['role']}: {c['content']}" for c in new_interactions])
        # self.interaction_memories.add_text(text_interactions, metadata={"timestamp": datetime.now()})

        return " ".join(self.inner_loop_responses)

    def use_tools(self, tool_use):
        try:
            returned_context = []
            for tool in tool_use:
                if tool.function.name in self.functions.keys():
                    fun = self.functions[tool.function.name]
                    args = json.loads(tool.function.arguments)

                    print(f"\tCalling '{tool.function.name}' with args {args} and id: {tool.id}")
                    result = fun(**args)
                    print(f"\t\tReturning function '{tool.function.name}' with id: {tool.id}")
                    returned_context.append({
                        "tool_call_id": tool.id,
                        "role": "tool",
                        "name": tool.function.name,
                        "content": result,
                    })
            return True, returned_context
        except:
            return False, []

    def send_message(self, message: str):
        message = repr(message)
        self.inner_loop_responses.append(message)
        return f"Sent {message} to the user"

    def end_inner_loop(self, message: str):
        self.inner_loop_responses.append(repr(message))
        self.loop_active = False
        return message

    def perform_memory_read(self, query: str, keywords: str= ""):

        filtering_kws = sorted([k.strip() for k in keywords.split(",")])
        colour_print("MAGENTA", f"Searching memories for: {query} with keywords: {keywords}")

        memories = []
        queried_memories = self.manual_knowledge.retrieve(query, 100)
        for m in queried_memories:
            colour_print("Magenta", f"Memory: {m.passage}")
            if keywords == "":
                memories.append(m)
                continue

            if m.relevance < 0.6:
                continue

            mem_kws = m.metadata["keywords"]

            for kw in filtering_kws:
                if kw in mem_kws:
                    colour_print("Magenta", "\tAccepted")
                    memories.append(m)
                    break
                else:
                    colour_print("Magenta", "\tRejected")

        colour_print("MAGENTA", f"Found {len(memories)} memories.")
        # interaction_memories = self.interaction_memories.retrieve(query, 5)
        all_memories = memories #+ interaction_memories
        if len(all_memories) > 0:
            sorted_mems = sorted(sorted(all_memories, key=lambda i: i.distance, reverse=True), key=lambda i: i.metadata["timestamp"], reverse=True)[:20]

            memory_messages = []
            # for idx, m in enumerate(sorted_mems):
            #     memory_messages.append(f"{idx} {m.metadata}:\n{m.passage}")

            for idx, m in enumerate(sorted_mems):
                memory_messages.append(f"{m.passage}")

            mems_to_process = '\n\n'.join(memory_messages)

            current_state = self.rebuild_state(mems_to_process)

            colour_print("GREEN", f"Memory reading returns: {current_state}")
            return current_state
        return "No memories found"

    def read_from_memory(self, query):
        return self.read_memory_loop(query)

    def read_memory_loop(self, query):

        # self.memory_loop_active = True
        context = [make_user_message(pystache.render(_read_memory_loop, {"original_query": query, "keywords": list(self.defined_kws)}))]
        while True:
            context.append(make_user_message(_read_memory_plan))
            # context, _ = ensure_context_len(context, "gpt-4o", max_len=self.max_prompt_size)
            # colour_print("Yellow", f"Attempting read_memory_loop call with: {context}")
            # response = litellm.completion(model="gpt-4o", messages=context, tools=TOOL_READ_MEMORY_LOOP, tool_choice="none")
            # print(f"LLM call with: {response.choices[0].message.model_extra}")

            # Create a plan
            plan_response = litellm.completion(model="gpt-4o", messages=context, tools=TOOL_READ_MEMORY_LOOP, tool_choice="none")
            context.append(make_assistant_message(plan_response.choices[0].message.content))

            colour_print("GREEN", f"Memory read plan is: {plan_response.choices[0].message.content}")
            context.append(make_user_message(_inner_loop_call))

            # Execute on plan
            colour_print("Yellow", f"Attempting read_memory_loop call with: {context}")
            response = litellm.completion(model="gpt-4o", messages=context, tools=TOOL_READ_MEMORY_LOOP, tool_choice="required")

            tool_use = response.choices[0].message.tool_calls


            success, new_context = self.use_tools(tool_use)

            if not success:
                continue

            context.append(response.choices[0].message.model_extra)
            context.extend(new_context)

            if context[-1]["name"] == "done":
                return context[-1]["content"]

    def done(self, results):
        colour_print("BLUE", f"Returning to the outer loop: {results}")
        return results

    def save_passive_memory(self, memory):
        context = [make_user_message(f"Create three general keywords to describe the topic of this interaction:\n{memory}.\nProduce the keywords in JSON like: `[keyword_1, keyword_2, keyword_3]`")]

        response = litellm.completion(model="gpt-3.5-turbo", messages=context)
        kws = sanitize_and_parse_json(response.choices[0].message.content)
        for kw in kws:
            self.defined_kws.add(kw)

        self.manual_knowledge.add_text(memory, metadata={"timestamp": datetime.now(), "keywords": kws})
        self.manual_knowledge.add_separator()
        colour_print("BLUE", f"Saved memory: {memory}\nwith keywords {repr(kws)}")

    def save_memory(self, data: str, keywords: str):
        kws = sorted([s.strip() for s in keywords.split(",")])
        self.manual_knowledge.add_text(data, metadata={"timestamp": datetime.now(), "keywords": kws})
        self.manual_knowledge.add_separator()
        return f"Saved memory {data}."

    def delete_memories_loop(self, topic: str):

        context = [make_user_message(pystache.render(_delete_memory_loop, {"topic": topic}))]
        while True:

            # Choose whether to read or be done
            context, _ = ensure_context_len(context, "gpt-4o", max_len=self.max_prompt_size)
            response = litellm.completion(model="gpt-4o", messages=context, tools=TOOL_READ_MEMORY_LOOP, tool_choice="required")
            try:
                tool_use = response.choices[0].message.tool_calls
            except (KeyError, AttributeError):

                continue

            success, new_context = self.use_tools(tool_use)

            if not success:
                continue

            context.append(response.choices[0].message.model_extra)
            context.extend(new_context)

            if context[-1]["name"] == "done":
                return f"Memories relating to {topic} deleted."

            # mems_retrieved = []#context[-1]["content"]

            # Get all the memories retrieved
            memory_set = set()
            for new_c in new_context:
                individual_mems = new_c["content"].split("\n\n")
                for m in individual_mems:
                    m = m[m.index(" "):].strip()
                    memory_set.add(m)

            memories_retrieved = list(memory_set)
            # Remove the numbers
            consolidated_memories = ""
            for idx, mem in enumerate(memories_retrieved):
                consolidated_memories += f"{idx} {mem}\n\n"

            second_stage_context = [make_user_message(consolidated_memories), make_user_message(pystache.render(_choose_to_delete, {"topic": topic}))]
            # Choose which memories should be deleted.

            response = litellm.completion(model="gpt-4o", messages=second_stage_context)

            response_list = sanitize_and_parse_json(response.choices[0].message.content)
            self.delete_memories(memories_retrieved, response_list)

    def delete_memories(self, memories, list_to_delete):
        # TODO: Optimisation would be to return the raw memories from the retrieval function.
        for deletion_idx in list_to_delete:
            memory = memories[deletion_idx].strip()

            # Get the text of the memory
            mem_text = memory.split("}:\n")[1]
            from_db = self.manual_knowledge.retrieve(mem_text, 1)[0]
            for k in from_db.textKeys:
                self.manual_knowledge.delete_text(k)

    def rebuild_state(self, all_memories):
        _current_state = """The current state is:
{{state}}
"""
        split_mems = all_memories.split("\n\n")

        current_state = "{}"
        context = [make_system_message(_state_reconstruction_prompt)]

        for memory in reversed(split_mems):
            context.append(make_user_message(pystache.render(_current_state, {"state": current_state})))
            context.append(make_user_message(f"Create a new state by integrating the current state with this new information:\n{memory}"))

            response = litellm.completion(model="gpt-4o", messages=context)

            current_state = response.choices[0].message.content
            context = context[:1]

        return current_state

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

        colour_print("CYAN",stmt)
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

