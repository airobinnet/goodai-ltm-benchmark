import re
import json
import random
from inspect import getsource
from typing import List, Tuple

from utils.data import get_gdrive_file, get_data_path
from dataclasses import dataclass, field
from utils.ui import ordinal
from utils.tokens import token_len
from dataset_interfaces.interface import DatasetInterface, TestExample


# Extracted from gdrive folder
# https://drive.google.com/drive/folders/1JkFHspT56_yRWwXVj47Fw0PzHtitODt5
GDRIVE_8K_ID = "15AcGiC4wIglru2gK2MHSX5Fie7gYxTTS"


def split_in_pages(text: str, max_tokens_per_split: int) -> list[str]:
    separator = ". "
    separator_len = token_len(separator)
    page_list = list()
    page_sentences = list()
    page_len = 0
    for sentence in text.split(separator):
        sentence_len = token_len(sentence)
        if page_len == 0:
            page_sentences.append(sentence)
            page_len += sentence_len
            continue
        if page_len + separator_len + sentence_len > max_tokens_per_split:
            page_list.append(separator.join(page_sentences))
            page_sentences.clear()
            page_len = 0
            continue
        page_sentences.append(sentence)
        page_len += separator_len + sentence_len
    if page_len > 0:
        page_list.append(separator.join(page_sentences))
    return page_list


def deliver_in_pages(text: str, max_page_tokens: int, first_page_prefix: str = "") -> list[str]:
    script = list()
    pages = split_in_pages(text, max_page_tokens)
    last_i = len(pages) - 1
    for i, page in enumerate(pages):
        last_page_str = " and last" if i == last_i else ""
        script.append(f"{first_page_prefix}{ordinal(i + 1)}{last_page_str} page:\n\n{page}")
    return script


@dataclass
class ChapterBreakDataset(DatasetInterface):
    name: str = "ChapterBreak"
    description: str = (
        "The agent is given a text that corresponds to the content of many book chapters, in addition to 6 possible "
        "beginnings for the next chapter. It is then asked to say which of the six continuations is the right one."
    )
    reset_message: str = "Forget the current chapter and its potential continuations."
    # The GoodAI split is a selection of samples that have been inspected by us to ensure that they are solvable, also
    # removing ordering hints from the chapter titles.
    split: str = "goodai"  # goodai / pg19 / ao3 / all
    max_message_tokens: int = 1024
    selection_info: dict = field(default_factory=dict)

    def __post_init__(self):
        assert self.split in {"goodai", "pg19", "ao3", "all"}

    def load_data(self) -> dict:
        filename = f"chapterbreak_ctx_8192.json"
        path = get_gdrive_file(self.name, GDRIVE_8K_ID, filename)
        with open(path) as fd:
            return json.load(fd)

    def apply_sample_selection(self, samples: dict) -> dict:
        with open(get_data_path(self.name, "chapterbreak-goodai-selection.json")) as fd:
            selection = json.load(fd)
        sample_selection = dict()
        for sel in selection:
            if not sel["solvable"]:
                continue
            sample = samples[sel["id"]]
            reg_expr = sel.get("chapter_cleanup", None)
            if reg_expr:
                right_value = re.match(reg_expr, sample["pos"]).group(1)
                false_beginnings = sample["negs"]
                for i in range(len(false_beginnings)):
                    wrong_value = re.match(reg_expr, false_beginnings[i]).group(1)
                    if wrong_value is not None:
                        false_beginnings[i] = false_beginnings[i].replace(wrong_value, right_value)
            sample_selection[sel["id"]] = sample
            self.selection_info[sel["id"]] = sel
        return sample_selection

    def get_samples(self, raw_dataset: dict) -> list[dict]:
        samples = dict()
        split_keys = ["pg19", "ao3"] if self.split in ["goodai", "all"] else [self.split]
        for split in split_keys:
            for book_id, chapter_samples in raw_dataset[split].items():
                for i, chapter in enumerate(chapter_samples):
                    samples[f"{book_id}_{i}"] = chapter
        if self.split == "goodai":
            samples = self.apply_sample_selection(samples)
        sample_list = [samples[k] | {"id": k} for k in sorted(samples.keys())]
        random.Random(self.seed).shuffle(sample_list)
        return sample_list

    def generate_examples(self, num_examples: int) -> list[TestExample]:
        data = self.load_data()
        sample_list = self.get_samples(data)
        example_list = list()

        for sample_idx, sample in zip(range(num_examples), sample_list):
            beginnings = [(True, sample["pos"])] + [(False, s) for s in sample["negs"]]
            random.Random(self.seed + sample_idx).shuffle(beginnings)

            script = ["I am going to read you the final pages of a book chapter. Okay?"]
            max_page_content_tokens = self.max_message_tokens - 20  # Leave some margin for text decorations
            script.extend(deliver_in_pages(sample["ctx"], max_page_content_tokens))

            answer = 0
            script.append(f"Now I will give you {len(beginnings)} options for the beginning of the next chapter. Ready?")
            for i, (is_true_suffix, option) in enumerate(beginnings):
                script.extend(deliver_in_pages(option, max_page_content_tokens, first_page_prefix=f"Option {i + 1}. "))
                if is_true_suffix:
                    answer = i + 1
            assert answer > 0

            script.append((
                "Which option is the true next-chapter beginning?\n"
                "Answer with a single-digit number, corresponding to the option number."
            ))

            is_question = [False] * len(script)
            is_question[-1] = True

            example = TestExample(
                dataset_name=self.name,
                description=self.description,
                example_id=sample["id"],
                dataset_generator=self,
                script=script,
                expected_responses=[str(answer)],
                evaluation_fn=self.evaluate_correct,
                is_question=is_question,
                number_of_questions=1,
                reset_message=self.reset_message,
            )
            example_list.append(example)

        return example_list

    def answer_statement_idx(self, example: TestExample) -> tuple[int, int]:

        # Find the message that goes after the last context page
        last_page_idx = None
        for i, script_line in enumerate(example.script):
            if script_line.endswith("options for the beginning of the next chapter. Ready?"):
                last_page_idx = i - 1
                break
        assert last_page_idx is not None
        context_script = example.script[1:last_page_idx + 1]

        # GoodAI-selected samples contain meta-data for these cases
        # Find the latest appearance of relevant information
        if example.example_id in self.selection_info:
            relevant_sentences = self.selection_info[example.example_id]["ctx"]
            script_index = char_index = None
            for line_idx, script_line in reversed(list(enumerate(context_script))):
                for sentence in relevant_sentences:
                    i = script_line.find(sentence)
                    if i < 0:
                        continue
                    last_char_i = i + len(sentence)
                    if script_index is None or char_index < last_char_i:
                        script_index, char_index = line_idx, last_char_i
                if script_index is not None:
                    return script_index + 1, char_index

        # Otherwise, we'll just assume that the key information lies somewhere around the middle of the context.
        middle_point_chars = sum(len(msg) for msg in context_script) // 2
        counted_chars = 0
        for script_answer_index, script_line in enumerate(context_script):
            counted_chars += len(script_line)
            if counted_chars > middle_point_chars:
                break
        answer_end_char = counted_chars - middle_point_chars
        return script_answer_index + 1, answer_end_char

    def evaluate_correct(
        self, questions: List[str], responses: List[str], expected_answers: List[str]
    ) -> Tuple[int, int, List[str]]:
        reasoning = getsource(ChapterBreakDataset.evaluate_correct)
        right_answer = expected_answers[0].strip()
        wrong_answers = [str(i + 1) for i in range(6) if str(i + 1) != right_answer]
        correct = right_answer in responses[0]
        for wrong_ans in wrong_answers:
            if wrong_ans in responses[0]:
                correct = False
                break
        score = int(correct)
        max_score = 1
        return score, max_score, [reasoning]
