
import json
import json
import argparse
import re
import string
import numpy as np
from collections import Counter
from typing import List, Set, Tuple, Union
from scipy.optimize import linear_sum_assignment
from word2number.w2n import word_to_num

ALL_QUESTION_TYPES = [
    'TextQ',
    'TableQ',
    'ImageQ',
    'ImageListQ',
    'Compose(TableQ,ImageListQ)',
    'Compose(TextQ,ImageListQ)',
    'Compose(ImageQ,TableQ)',
    'Compose(ImageQ,TextQ)',
    'Compose(TextQ,TableQ)',
    'Compose(TableQ,TextQ)',
    'Intersect(TableQ,TextQ)',
    'Intersect(ImageListQ,TableQ)',
    'Intersect(ImageListQ,TextQ)',
    'Compare(Compose(TableQ,ImageQ),TableQ)',
    'Compare(Compose(TableQ,ImageQ),Compose(TableQ,TextQ))',
    'Compare(TableQ,Compose(TableQ,TextQ))',
]

TEXT_SINGLE_HOP_QUESTION_TYPES = [
    'TextQ',
]
TEXT_AS_FIRST_HOP_QUESTION_TYPES = [
    'Compare(TableQ,Compose(TableQ,TextQ))',
    'Compose(ImageQ,TextQ)',
    'Compose(TableQ,TextQ)',
    'Intersect(TableQ,TextQ)',
    'Intersect(ImageListQ,TextQ)',
]
TEXT_AS_SECOND_HOP_QUESTION_TYPES = [
    'Compare(Compose(TableQ,ImageQ),Compose(TableQ,TextQ))',
    'Compose(TextQ,ImageListQ)',
    'Compose(TextQ,TableQ)',
]

TABLE_SINGLE_HOP_QUESTION_TYPES = [
    "TableQ"
]
TABLE_AS_FIRST_HOP_QUESTION_TYPES = [
    'Compose(ImageQ,TableQ)',
    'Compose(TextQ,TableQ)',
]
TABLE_AS_SECOND_HOP_QUESTION_TYPES = [
    'Compare(Compose(TableQ,ImageQ),TableQ)',
    'Compare(TableQ,Compose(TableQ,TextQ))',
    'Compose(TableQ,ImageListQ)',
    'Compose(TableQ,TextQ)',
    'Intersect(ImageListQ,TableQ)',
    'Intersect(TableQ,TextQ)',
]

IMAGE_SINGLE_HOP_QUESTION_TYPES = [
    'ImageQ',
    'ImageListQ'
]
IMAGE_AS_FIRST_HOP_QUESTION_TYPES = [
    'Compare(Compose(TableQ,ImageQ),Compose(TableQ,TextQ))',
    'Compare(Compose(TableQ,ImageQ),TableQ)',
    'Compose(TableQ,ImageListQ)',
    'Compose(TextQ,ImageListQ)',
    'Intersect(ImageListQ,TableQ)',
]
IMAGE_AS_SECOND_HOP_QUESTION_TYPES = [
    'Compose(ImageQ,TableQ)',
    'Compose(ImageQ,TextQ)',
    'Intersect(ImageListQ,TextQ)',
]


# every question should be answered either as a single hop question, or two-hop question
assert set(TEXT_SINGLE_HOP_QUESTION_TYPES + TEXT_AS_SECOND_HOP_QUESTION_TYPES
           + TABLE_SINGLE_HOP_QUESTION_TYPES + TABLE_AS_SECOND_HOP_QUESTION_TYPES
           + IMAGE_SINGLE_HOP_QUESTION_TYPES + IMAGE_AS_SECOND_HOP_QUESTION_TYPES) == set(ALL_QUESTION_TYPES)
assert len(set(TEXT_SINGLE_HOP_QUESTION_TYPES) & set(TEXT_AS_SECOND_HOP_QUESTION_TYPES)) == 0
assert len(set(TABLE_SINGLE_HOP_QUESTION_TYPES) & set(TABLE_AS_SECOND_HOP_QUESTION_TYPES)) == 0
assert len(set(IMAGE_SINGLE_HOP_QUESTION_TYPES) & set(IMAGE_AS_SECOND_HOP_QUESTION_TYPES)) == 0

SINGLE_HOP_QUESTION_TYPES = TEXT_SINGLE_HOP_QUESTION_TYPES \
                            + TABLE_SINGLE_HOP_QUESTION_TYPES \
                            + IMAGE_SINGLE_HOP_QUESTION_TYPES
MULTI_HOP_QUESTION_TYPES = TEXT_AS_SECOND_HOP_QUESTION_TYPES \
                           + TABLE_AS_SECOND_HOP_QUESTION_TYPES + \
                           IMAGE_AS_SECOND_HOP_QUESTION_TYPES
# no duplicated multi-hop question types
assert len(MULTI_HOP_QUESTION_TYPES) == len(set(MULTI_HOP_QUESTION_TYPES))
# no duplication for the first hop
assert set(TEXT_AS_FIRST_HOP_QUESTION_TYPES + TABLE_AS_FIRST_HOP_QUESTION_TYPES + IMAGE_AS_FIRST_HOP_QUESTION_TYPES) \
       == set(MULTI_HOP_QUESTION_TYPES)
# single + multi = all
assert set(SINGLE_HOP_QUESTION_TYPES + MULTI_HOP_QUESTION_TYPES) == set(ALL_QUESTION_TYPES)


def process_question_for_implicit_decomp(question, question_type, hop=0, bridge_entity='', sep_token='[SEP]'):
    if isinstance(bridge_entity, list) or isinstance(bridge_entity, set):
        bridge_entity = "; ".join(bridge_entity)
    return (
        f'{question_type} {sep_token} '
        f'HOP={hop} {sep_token} '
        f'{bridge_entity} {sep_token} '
        f'{question}')


def extract_numbers_from_str(s):
    numbers = []
    for token in s.split():
        try:
            num = int(token.replace(",", ""))
        except:
            try:
                num = float(token)
            except:
                num = None
        if num:
            numbers.append(num)
    return numbers


def read_jsonl(filename):
    with open(filename, 'r') as f:
        data = [json.loads(l.strip()) for l in f.readlines()]
    return data
    
    
def _remove_articles(text: str) -> str:
    regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
    return re.sub(regex, " ", text)


def _white_space_fix(text: str) -> str:
    return " ".join(text.split())


EXCLUDE = set(string.punctuation)


def _remove_punc(text: str) -> str:
    if not _is_number(text):
        return "".join(ch for ch in text if ch not in EXCLUDE)
    else:
        return text


def _lower(text: str) -> str:
    return text.lower()


def _tokenize(text: str) -> List[str]:
    return re.split(" |-", text)


def _normalize_answer(text: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""

    parts = [
        _white_space_fix(_remove_articles(_normalize_number(_remove_punc(_lower(token)))))
        for token in _tokenize(text)
    ]
    parts = [part for part in parts if part.strip()]
    normalized = " ".join(parts).strip()
    return normalized


def _is_number(text: str) -> bool:
    try:
        float(text)
        return True
    except ValueError:
        return False


def _is_word_number(text: str) -> bool:
    try:
        word_to_num(text)
        return True
    except ValueError:
        return False


def _normalize_number(text: str) -> str:
    if _is_number(text):
        return str(float(text))
    #TODO: this is not included in the original drop evaluation script, we need to have our own in the end anyways.
    elif _is_word_number(text):
        return str(float(word_to_num(text)))
    else:
        return text


def _answer_to_bags(
    answer: Union[str, List[str], Tuple[str, ...]]
) -> Tuple[List[str], List[Set[str]]]:
    if isinstance(answer, (list, tuple)):
        raw_spans = answer
    else:
        raw_spans = [answer]
    normalized_spans: List[str] = []
    token_bags = []
    for raw_span in raw_spans:
        normalized_span = _normalize_answer(raw_span)
        normalized_spans.append(normalized_span)
        token_bags.append(set(normalized_span.split()))
    return normalized_spans, token_bags


def _align_bags(predicted: List[Set[str]], gold: List[Set[str]]) -> List[float]:
    """
    Takes gold and predicted answer sets and first finds the optimal 1-1 alignment
    between them and gets maximum metric values over all the answers.
    """
    scores = np.zeros([len(gold), len(predicted)])
    for gold_index, gold_item in enumerate(gold):
        for pred_index, pred_item in enumerate(predicted):
            if _match_numbers_if_present(gold_item, pred_item):
                scores[gold_index, pred_index] = _compute_f1(pred_item, gold_item)
    row_ind, col_ind = linear_sum_assignment(-scores)

    max_scores = np.zeros([max(len(gold), len(predicted))])
    for row, column in zip(row_ind, col_ind):
        max_scores[row] = max(max_scores[row], scores[row, column])
    return max_scores


def _compute_f1(predicted_bag: Set[str], gold_bag: Set[str]) -> float:
    intersection = len(gold_bag.intersection(predicted_bag))
    if not predicted_bag:
        precision = 1.0
    else:
        precision = intersection / float(len(predicted_bag))
    if not gold_bag:
        recall = 1.0
    else:
        recall = intersection / float(len(gold_bag))
    f1 = (
        (2 * precision * recall) / (precision + recall)
        if not (precision == 0.0 and recall == 0.0)
        else 0.0
    )
    return f1


def _match_numbers_if_present(gold_bag: Set[str], predicted_bag: Set[str]) -> bool:
    gold_numbers = set()
    predicted_numbers = set()
    for word in gold_bag:
        if _is_number(word):
            gold_numbers.add(word)
    for word in predicted_bag:
        if _is_number(word):
            predicted_numbers.add(word)
    if (not gold_numbers) or gold_numbers.intersection(predicted_numbers):
        return True
    return False



def list_em(predicted, gold):
    predicted_bags = _answer_to_bags(predicted)
    gold_bags = _answer_to_bags(gold)
    if set(predicted_bags[0]) == set(gold_bags[0]) and len(predicted_bags[0]) == len(gold_bags[0]):
        return 1.0
    else:
        return 0.0


def list_f1(predicted, gold):
    predicted_bags = _answer_to_bags(predicted)
    gold_bags = _answer_to_bags(gold)
    f1_per_bag = _align_bags(predicted_bags[1], gold_bags[1])
    f1 = np.mean(f1_per_bag)
    f1 = round(f1, 2)
    return f1