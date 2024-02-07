from typing import Tuple, List
from uuid import uuid4
from random import choice
from inspect import getsource
from dataclasses import dataclass
from dataset_interfaces.interface import DatasetInterface, TestExample


def generate_kv_pairs(
    num_pairs: int,
    nested_levels: int = 0,
) -> tuple[list[str], dict[str, str]]:
    if nested_levels == 0:
        root_keys = [str(uuid4()) for _ in range(num_pairs)]
        return root_keys, {k: str(uuid4()) for k in root_keys}
    else:
        num_keys = num_pairs // (nested_levels + 1)
        lower_keys, lower_pairs = generate_kv_pairs(
            num_pairs=num_pairs - num_keys,
            nested_levels=nested_levels - 1,
        )
        root_keys = [str(uuid4()) for _ in range(num_keys)]
        return root_keys, {k: lk for k, lk in zip(root_keys, lower_keys)} | lower_pairs


def gold_values(root_keys: list[str], kv_pairs: dict[str, str]) -> tuple[str, str]:
    value = key = choice(root_keys)
    while value in kv_pairs:
        value = kv_pairs[value]
    return key, value


@dataclass
class KVPairsDataset(DatasetInterface):
    name: str = "Key-Value Pairs"
    description: str = (
        "Ask the agent to retrieve the leaf value corresponding to a key in the "
        "provided JSON."
    )
    question: str = ""
    nested_levels: int = 0
    num_kv_pairs: int = 140  # To fill up 8k-token context (see 3.2.2 in MemGPT paper)

    def generate_examples(self, num_examples: int) -> list[TestExample]:
        examples = list()
        for _ in range(num_examples):
            root_keys, kv_pairs = generate_kv_pairs(
                num_pairs=self.num_kv_pairs,
                nested_levels=self.nested_levels,
            )
            key, value = gold_values(root_keys, kv_pairs)
            script = [
                f"Take a look at this JSON file:\n\n{kv_pairs}",
                f"In the JSON file I showed you before, what is the leaf value corresponding to the key {repr(key)}?",
            ]
            is_question = [False, True]
            test_example = TestExample(
                dataset_name=self.name,
                description=self.description,
                dataset_generator=self,
                script=script,
                expected_responses=[value],
                evaluation_fn=self.evaluate_correct,
                number_of_questions=self.count_questions(is_question),
                is_question=is_question,
            )
            examples.append(test_example)
        return examples

    def evaluate_correct(
        self, questions: List[str], responses: List[str], expected_answers: List[str]
    ) -> Tuple[int, int, List[str]]:
        reasoning = getsource(KVPairsDataset.evaluate_correct)
        return int(expected_answers[0] in responses[0]), 1, [reasoning]

    def answer_statement_idx(self, example: TestExample) -> Tuple[int, int]:
        # This is a single message test, so the relevant information is in that message.
        # Need to get the position after the relevant information inside the statement
        string_pos = example.script[0].find(example.expected_responses[0]) + len(
            example.expected_responses[0]
        )
        return 0, string_pos
