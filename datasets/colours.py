from dataclasses import dataclass
from random import choice
from typing import List, Tuple
import pystache
from dataset_interfaces.interface import DatasetInterface, TestExample


COLOURS = [
    "Red",
    "White",
    "Blue",
    "Green",
    "Yellow",
    "Pink",
    "Magenta",
    "Cyan",
    "Purple",
    "Tan",
    "Khaki",
    "Beige",
    "Cream",
    "Brown",
    "Olive",
    # "None",
]

STATEMENTS = [
    "My favourite colour is {{colour}}.",
    "{{colour}} is my favourite colour.",
    "My favourite colour could be described as {{colour}}.",
    "The name of my favourite colour is {{colour}}.",
]


@dataclass
class ColourDataset(DatasetInterface):
    name: str = "Colours"
    description: str = "Tell the agent what your favourite colour is multiple time, then ask it what that colour is."
    question: str = "What is my favourite colour?"
    colour_changes: int = 3

    def generate_examples(self, num_examples):
        examples = []

        for _ in range(num_examples):
            is_question = []
            colours = []
            script = []
            renderer = pystache.Renderer()

            for change in range(self.colour_changes):
                colour = choice(COLOURS)
                if colour == "None":
                    name_stmt = "I have no favourite colour."
                else:
                    name_stmt = renderer.render(choice(STATEMENTS), {"colour": colour})
                colours.append(colour)
                script.append(name_stmt)
                is_question.append(False)

            script.append(self.question)
            is_question.append(True)
            answer_list = [colours[-1]]
            example = TestExample(
                dataset_generator=self,
                script=script,
                token_spacings=self.create_filler(is_question),
                expected_responses=answer_list,
                is_question=is_question,
            )
            examples.append(example)

        return examples

    def evaluate_correct(
        self, questions: List[str], responses: List[str], expected_answers: List[str]
    ) -> Tuple[int, int, List[str]]:
        color = expected_answers[0].lower()
        if color in responses[-1].lower():
            return 1, 1, [f'"{color}" is in the response.']
        return 0, 1, [f'"{color}" is NOT in the response.']

    def answer_statement_idx(self, example: TestExample) -> Tuple[int, int]:
        # The correct answer is the last colour that we told the agent
        # In this test all statements are atomic
        current = len(example.script) - 1
        for idx, text in enumerate(example.script[::-1]):
            if example.expected_responses[0] in text:
                current -= idx
                break
        return current, len(example.script[current])
