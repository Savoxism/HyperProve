"""Prompt templates specified in the HyperProve paper appendix."""

from __future__ import annotations

import json


ENTITY_EXTRACTION_PROMPT = """Task:
Given a passage, identify the entities that are needed to represent the factual content of the passage.

Output format:
Return only a valid JSON object with the following structure:
{{
  "entities": ["entity_1", "entity_2", "..."]
}}

Guidelines:
1. Include all named entities mentioned in the passage.
2. Include all dates, temporal expressions, and events.
3. Include generic concepts when they are central to the passage topic.
4. Include entities that participate in meaningful relations with the extracted entities.
5. Preserve the surface form used in the passage whenever possible.
6. Do not return an empty list.
7. Do not include explanations outside the JSON object.

Example passage:
Radio City is India's first private FM radio station and was started on 3 July 2001. It plays Hindi, English and regional songs. Radio City recently forayed into New Media in May 2008 with the launch of a music portal, PlanetRadiocity.com, that offers music-related news, videos, songs, and other features.

Example output:
{{
  "entities": [
    "Radio City",
    "India",
    "private FM radio station",
    "3 July 2001",
    "Hindi",
    "English",
    "regional songs",
    "New Media",
    "May 2008",
    "PlanetRadiocity.com",
    "music portal",
    "music-related news",
    "videos",
    "songs"
  ]
}}

Input:
Passage: {passage}"""


HYPEREDGE_EXTRACTION_PROMPT = """Task:
Given a passage and a predefined list of entities, decompose the passage into atomic facts. Each atomic fact should express one self-contained claim and explicitly specify the entities involved in that fact.

Output format:
Return only a valid JSON object with the following structure:
{{
  "hyperedges": [
    {{
      "text": "a complete atomic fact",
      "entities": ["entity_1", "entity_2", "..."]
    }}
  ]
}}

Guidelines:
1. Each hyperedge must contain exactly one factual claim or relation.
2. Each hyperedge must be understandable without requiring the original passage.
3. Use only entities from the provided entity list. Do not introduce new entities.
4. Make the relation among entities explicit and unambiguous.
5. Preserve temporal, causal, comparative, and modifier information when it affects the meaning of the fact.
6. When a statement involves more than two entities, keep them together in a single hyperedge rather than splitting the fact into disconnected binary relations.
7. Include all relevant entities from the provided entity list in both the hyperedge text and the entity array.
8. Ensure that the set of hyperedges covers all important factual information in the passage.
9. Do not include explanations outside the JSON object.

Example passage:
In 2020, after Apple launched the M1 chip, major software companies like Adobe optimized their applications, improving performance by up to 80 percent compared to Intel-based Macs.

Entity list:
[
  "Apple",
  "M1 chip",
  "2020",
  "Adobe",
  "Adobe's applications",
  "Intel-based Macs",
  "80% performance improvement"
]

Example output:
{{
  "hyperedges": [
    {{
      "text": "Apple launched the M1 chip in 2020.",
      "entities": ["Apple", "M1 chip", "2020"]
    }},
    {{
      "text": "Adobe optimized Adobe's applications for the M1 chip after the chip was launched.",
      "entities": ["Adobe", "Adobe's applications", "M1 chip"]
    }},
    {{
      "text": "Adobe's applications on the M1 chip achieved up to an 80% performance improvement compared with Adobe's applications on Intel-based Macs.",
      "entities": [
        "Adobe",
        "Adobe's applications",
        "M1 chip",
        "80% performance improvement",
        "Intel-based Macs"
      ]
    }}
  ]
}}

Input:
Passage: {passage}
Entity list: {entities_json_list}"""


QUESTION_DECOMPOSITION_PROMPT = """Task:
Break the input question into the smallest number of retrieval-friendly sub-questions needed to answer it reliably.

Main principle:
Decompose the question according to bridge facts, not according to surface length. Each sub-question should normally resolve one unknown bridge entity, one bridge relation, or the final requested attribute.

Guidelines:
1. If the question can be answered with a single retrieval step, return a list containing only the original question.
2. For sequential reasoning, create one sub-question for each bridge step when a later step depends on the answer to an earlier step.
3. A good sub-question should usually do one of the following:
   - identify a bridge entity;
   - identify a relation involving a known bridge entity;
   - ask for the final attribute once the target entity is known.
4. Avoid sub-questions that require resolving multiple new bridge entities at the same time.
5. Prefer two sub-questions for one-bridge questions, three for two-bridge questions, and four for three-bridge questions when needed.
6. Do not force every question into two sub-questions. Use more steps when this makes retrieval clearer.
7. Preserve the reasoning order. Later sub-questions may refer to earlier answers using phrases such as "that person", "that city", or "that work".
8. For parallel or comparative questions, retrieve the required attributes separately before asking for the comparison.
9. Avoid redundant or overlapping sub-questions.
10. Return at most four sub-questions.

Output format:
Return only a valid JSON object:
{{
  "sub_questions": [
    "first sub-question",
    "second sub-question"
  ]
}}

Examples:
Question:
What county is Erik Hort's birthplace a part of?

Output:
{{
  "sub_questions": [
    "What is Erik Hort's birthplace?",
    "Which county is that birthplace part of?"
  ]
}}

Question:
When was the start of the battle of the birthplace of the performer of III?

Output:
{{
  "sub_questions": [
    "Who is the performer of III?",
    "What is that performer's birthplace?",
    "When did the battle at that birthplace start?"
  ]
}}

Input:
Question: {question}"""





SUBQUERY_REWRITE_PROMPT = """Task:
Rewrite a sub-question into a standalone retrieval query by replacing its references to earlier sub-questions with their validated answers.

Guidelines:
1. Replace referring expressions such as "that person", "that city", "that work", "it", "he", "she", "they", "its", or "their" with the answer they refer to.
2. A reference may point to any earlier resolved sub-question, not only the most recent one.
3. Use only the listed answers. Do not answer the sub-question and do not add facts that are not given.
4. Preserve the unresolved relation and the requested attribute; change only the referring expressions.
5. If the sub-question contains no reference to a resolved answer, return it unchanged.
6. Do not include explanations outside the JSON object.

Output format:
Return only a valid JSON object:
{{
  "rewritten_query": "..."
}}

Example:
Resolved sub-questions and answers:
Sub-question: Who is the performer of III?
Answer: Stanton Moore
Sub-question: What is that performer's birthplace?
Answer: New Orleans

Sub-question: When did the battle at that birthplace start?

Output:
{{
  "rewritten_query": "When did the battle at New Orleans start?"
}}

Input:
Resolved sub-questions and answers:
{qa_block}

Sub-question: {sub_query}"""


SUBQUERY_EVALUATION_PROMPT = """Task:
Given a sub-query, retrieved atomic-fact hyperedges, and source passages, decide whether the evidence answers the sub-query.

Output format:
Return only a valid JSON object:
{{
  "status": "ANSWERED" or "UNANSWERED",
  "answer": "short sub-answer or null",
  "supporting_hyperedge_ids": [0, 1]
}}

Guidelines:
1. Use ANSWERED only when the hyperedges and passages support a direct answer.
2. The answer must be concise.
3. supporting_hyperedge_ids are 0-based indices from the listed hyperedges.
4. Do not include explanations outside the JSON object.

Sub-query: {sub_query}

Hyperedges:
{hyperedges_block}

Passages:
{passages_block}"""


FINAL_ANSWER_INTEGRATION_PROMPT = """Task:
Given the original question and the sub-question answers produced during multi hop reasoning, combine the intermediate answers into a single final answer.

Guidelines:
1. The final answer must directly answer the original question.
2. Use the available sub-question answers.
3. If one sub-answer is missing, still provide the best short answer supported by the available information.
4. Output only the answer, without reasoning or restating the question.
5. Return only a valid JSON object.

Output format:
{{
  "final_answer": "..."
}}

Original question:
{original_question}

Sub-questions and answers:
{qa_block}"""


QA_READER_SYSTEM_PROMPT = (
    "As an advanced reading comprehension assistant, your task is to analyze text passages and "
    "corresponding questions meticulously. Your response start after \"Thought: \", where you will "
    "methodically break down the reasoning process, illustrating how you arrive at conclusions. "
    "Conclude with exactly one line formatted as \"FINAL_ANSWER: <answer>\". The final answer must "
    "be a concise answer span only, with no explanation, no restated question, and no extra text after it."
)


QA_READER_USER_PROMPT = """{passage_blocks}

Retrieved propositions:
{numbered_propositions}

Retriever hypothesis: {retriever_hypothesis}

Question: {question}
Thought:"""


ANSWER_JUDGE_PROMPT = """Task:
Evaluate whether a model prediction answers the question correctly by comparing it against the reference answer.

Decision rule:
Return [[yes]] if the prediction refers to the same entity, value, or fact as the reference answer. Return [[no]] if it refers to something different or contradicts the reference.

Guidelines:
1. Accept paraphrases, aliases, abbreviations, nicknames, spelling or transliteration variants, and equivalent date, number, or unit formats.
2. Accept a prediction that contains the reference answer, even with additional words or reasoning.
3. Accept a prediction that is a more specific or slightly less specific form of the same answer (for example, a full name versus a surname, a city with or without its state, a full date versus the same month and year), as long as it clearly points to the same thing.
4. Accept numbers that match after reasonable rounding.
5. For a list-valued reference, accept a prediction that gives at least one correct item and no wrong item.
6. Return [[no]] when the prediction names a different entity, value, or time, is too vague to identify the reference, or contradicts it.
7. Return only [[yes]] or [[no]].

Question:
{question}

Reference answer:
{ground_truth}

Model prediction:
{prediction}

Is the model prediction correct?"""


def format_entity_extraction_prompt(passage: str) -> str:
    return ENTITY_EXTRACTION_PROMPT.format(passage=passage)


def format_hyperedge_extraction_prompt(passage: str, entities: list[str]) -> str:
    return HYPEREDGE_EXTRACTION_PROMPT.format(
        passage=passage,
        entities_json_list=json.dumps(entities, ensure_ascii=False),
    )


def format_decomposition_prompt(question: str) -> str:
    return QUESTION_DECOMPOSITION_PROMPT.format(question=question)


def format_subquery_rewrite_prompt(sub_query: str, resolved: list[tuple[str, str]]) -> str:
    qa_block = "\n".join(f"Sub-question: {q}\nAnswer: {a}" for q, a in resolved)
    return SUBQUERY_REWRITE_PROMPT.format(qa_block=qa_block, sub_query=sub_query)


def format_subquery_evaluation_prompt(
    sub_query: str,
    hyperedges: list[str],
    passages: list[dict[str, str]],
) -> str:
    hyperedges_block = "\n".join(f"{idx}. {text}" for idx, text in enumerate(hyperedges)) or "(none)"
    passages_block = "\n---\n".join(
        f"Wikipedia Title: {item.get('title', '')}\n{item.get('text', '')}"
        for item in passages
    ) or "(none)"
    return SUBQUERY_EVALUATION_PROMPT.format(
        sub_query=sub_query,
        hyperedges_block=hyperedges_block,
        passages_block=passages_block,
    )


def format_final_answer_integration_prompt(
    original_question: str,
    sub_answers: list[tuple[str, str | None]],
) -> str:
    qa_block = "\n".join(
        f"Sub-question: {q}\nAnswer: {a or '(missing)'}" for q, a in sub_answers
    )
    return FINAL_ANSWER_INTEGRATION_PROMPT.format(
        original_question=original_question,
        qa_block=qa_block,
    )


def format_qa_reader_prompt(
    question: str,
    passages: list[dict[str, str]],
    hyperedges: list[str],
    retriever_hypothesis: str = "",
) -> tuple[str, str]:
    passage_blocks = "\n\n".join(
        f"Wikipedia Title: {item.get('title', '')}\n{item.get('text', '')}"
        for item in passages
    )
    hyperedges_block = "\n".join(f"{idx}. {text}" for idx, text in enumerate(hyperedges)) or "(none)"
    return QA_READER_SYSTEM_PROMPT, QA_READER_USER_PROMPT.format(
        passage_blocks=passage_blocks,
        numbered_propositions=hyperedges_block,
        retriever_hypothesis=retriever_hypothesis or "(none)",
        question=question,
    )
