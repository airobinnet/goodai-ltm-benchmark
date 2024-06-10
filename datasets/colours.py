from dataclasses import dataclass
from typing import List, Tuple
import pystache
from dataset_interfaces.interface import DatasetInterface, TestExample
import re
import json

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
]

STATEMENTS = [
    "My favourite colour is {{colour}}.",
    "{{colour}} is my favourite colour.",
    "My favourite colour could be described as {{colour}}.",
    "The name of my favourite colour is {{colour}}.",
]


def match_color(text, colors):
    """
    Check if any of the exact words in 'colors' is in the text.
    Returns True if any color is found as a whole word, otherwise False.
    """
    colors_pattern = '|'.join([re.escape(color) for color in colors])
    pattern = rf'(?<!\S)(?:\"|\')??(?:{colors_pattern})(?:\"|\')??(?!\S)|(?<!\S)(?:\"|\')??(?:{colors_pattern})(\
    ?:\"|\')??(?=[,.!?])'
    match = re.search(pattern, text, re.IGNORECASE)
    return True if match else False


@dataclass
class ColourDataset(DatasetInterface):
    name: str = "Colours"
    description: str = "Tell the agent what your favourite colour is multiple times, then ask it what that colour is."
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
                colour = self.random.choice(COLOURS)
                if colour == "None":
                    name_stmt = "I have no favourite colour."
                else:
                    name_stmt = renderer.render(self.random.choice(STATEMENTS), {"colour": colour})
                colours.append(colour)
                script.append(name_stmt)
                is_question.append(False)

            script.append(self.question)
            is_question.append(True)
            answer_list = [colours[-1]]
            example = TestExample(
                dataset_generator=self,
                script=script,
                expected_responses=answer_list,
                is_question=is_question,
            )
            examples.append(example)

        return examples

    # def evaluate_correct(
    #         self, questions: List[str], responses: List[str], expected_answers: List[str]
    # ) -> tuple[float, int, list[str]]:
    #     score = 0
    #     max_score = 1
    #     response_messages = []
    #
    #     for expected_color, response_text in zip(expected_answers, responses):
    #         result = match_color(response_text, [expected_color])
    #         if expected_color == "" and result is False:
    #             score += 1
    #             response_messages.append("No color expected in the response.")
    #         elif result:
    #             score += 1
    #             response_messages.append(f'"{expected_color}" is in the response.')
    #         else:
    #             response_messages.append(f'"{expected_color}" is NOT in the response.')
    #
    #     score = score / len(expected_answers)
    #
    #     return score, max_score, response_messages

    def evaluate_correct(
            self, questions: List[str], responses: List[str], expected_answers: List[str]
    ) -> tuple[float, int, list[str]]:
        if not expected_answers:
            if all(not match_color(response_text, COLOURS) for response_text in responses):
                return 0.0, 0, ["No colors expected, and no colors found in the responses."]
            else:
                return 0.0, 0, ["No colors expected, but colors found in the responses."]

        max_score = len(expected_answers)
        score = 0
        response_messages = []

        for response_text in responses:
            colors_found = [color for color in expected_answers if match_color(response_text, [color])]
            colors_missing = [color for color in expected_answers if color not in colors_found]

            score += len(colors_found)
            response_messages.append(f'Found colors: {colors_found}')
            if colors_missing:
                response_messages.append(f'Missing colors: {colors_missing}')

        score_ratio = score / max_score

        return score_ratio, 1, response_messages
