"""
Client for cleaning text using a remote LLM API.
"""
from typing import Optional

from virtual_literature_companion.config import TEXT_CLEAN_LLM_CONFIG
from .request import make_llm_request, make_llm_request_async


system_prompt = """You are an expert in cleaning OCR text from books. Your task is to correct errors, remove extraneous elements, and format the text properly, while preserving the original content and structure.

Follow these rules:
1.  **Remove Headers and Footers:** Delete any page numbers, book titles, or chapter titles that appear at the top or bottom of the page.
2.  **Correct OCR Errors:** Fix common OCR mistakes, such as `hew as` to `he was`, `s coundrel` to `scoundrel`, or `oflf` to `off`.
3.  **Preserve Formatting:** Keep original formatting like italics or bold text, often marked with `*` or `_`.
4.  **Do Not Add New Content:** Do not add any words or sentences that were not in the original text. Your job is to clean, not to create.
5. **Do Not Remove Content:** Do not remove any content from the original text. Even if at the end the text cuts off in the middle of a sentence, keep it, as I am giving you a page of text, and the rest is likely on the next page.
6.  **Fix Spacing and Line Breaks:** Correct spacing issues between words and ensure paragraphs are separated by a single newline.

The raw text will be in <raw> tags. Output the cleaned text between <cleaned>...</cleaned> tags.
"""

example_messages = [
    {"role": "user", "content": """<raw>
81 Hamlet ACT 2. SC. 2
FTLN 1046 And leads the will to desperate undertakings
FTLN 1047 As oft as any passions under heaven
FTLN 1048 That does afflict our natures. I am sorry.
</raw>"""},
    {"role": "assistant", "content": """<cleaned>
And leads the will to desperate undertakings
As oft as any passions under heaven
That does afflict our natures. I am sorry.
</cleaned>"""},
    {"role": "user", "content": """<raw>
Why isthe ksy blue? It is not blue, that isjustyour i m a g i n a t i o n.
</raw>"""},
    {"role": "assistant", "content": """<cleaned>
Why is the sky blue? It is not blue, that is just your imagination.
</cleaned>"""},
]


def clean_text(
    raw_text: str,
    max_tokens: Optional[int] = None,
    max_retries: int = 3,
) -> Optional[str]:
    """
    Clean OCR text using a specialized prompt via an LLM API.

    If the input text is empty or consists only of whitespace, it returns
    the original text immediately without calling the LLM.

    Args:
        raw_text: The raw, potentially noisy text to be cleaned.
        max_tokens: The maximum number of tokens for the generated response.
                    If None, it is estimated based on the input text length.

    Returns:
        The cleaned text as a string, or None if the request fails.
    """
    if not raw_text.strip():
        return raw_text

    prompt = f"<raw>\n{raw_text}\n</raw>"

    if max_tokens is None:
        # Estimate max tokens based on input length plus a buffer.
        max_tokens = len(raw_text.split()) + 100

    messages = example_messages + [
        {"role": "user", "content": prompt}
    ]

    response_text = make_llm_request(
        messages=messages,
        max_tokens=max_tokens,
        system_message=system_prompt,
        max_retries=max_retries,
        config=TEXT_CLEAN_LLM_CONFIG
    )

    if response_text:
        # Extract the cleaned text from between <cleaned> tags
        if "<cleaned>" in response_text and "</cleaned>" in response_text:
            start_tag = "<cleaned>"
            end_tag = "</cleaned>"
            start_idx = response_text.find(start_tag) + len(start_tag)
            end_idx = response_text.find(end_tag)
            response_text = response_text[start_idx:end_idx].strip()

        elif "<cleaned>" in response_text:
            start_tag = "<cleaned>"
            start_idx = response_text.find(start_tag) + len(start_tag)
            response_text = response_text[start_idx:].strip()

        return response_text

    return None


async def clean_text_async(
    raw_text: str,
    max_tokens: Optional[int] = None,
    max_retries: int = 3,
) -> Optional[str]:
    """
    Clean OCR text using a specialized prompt via an LLM API.

    If the input text is empty or consists only of whitespace, it returns
    the original text immediately without calling the LLM.

    Args:
        raw_text: The raw, potentially noisy text to be cleaned.
        max_tokens: The maximum number of tokens for the generated response.
                    If None, it is estimated based on the input text length.

    Returns:
        The cleaned text as a string, or None if the request fails.
    """
    if not raw_text.strip():
        return raw_text

    prompt = f"<raw>\n{raw_text}\n</raw>"

    if max_tokens is None:
        # Estimate max tokens based on input length plus a buffer.
        max_tokens = len(raw_text.split()) + 100

    messages = example_messages + [{"role": "user", "content": prompt}]

    response_text = await make_llm_request_async(
        messages=messages,
        max_tokens=max_tokens,
        system_message=system_prompt,
        max_retries=max_retries,
        config=TEXT_CLEAN_LLM_CONFIG
    )

    if response_text:
        # Extract the cleaned text from between <cleaned> tags
        if "<cleaned>" in response_text and "</cleaned>" in response_text:
            start_tag = "<cleaned>"
            end_tag = "</cleaned>"
            start_idx = response_text.find(start_tag) + len(start_tag)
            end_idx = response_text.find(end_tag)
            response_text = response_text[start_idx:end_idx].strip()

        elif "<cleaned>" in response_text:
            start_tag = "<cleaned>"
            start_idx = response_text.find(start_tag) + len(start_tag)
            response_text = response_text[start_idx:].strip()

        return response_text

    return None 