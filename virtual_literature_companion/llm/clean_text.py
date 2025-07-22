"""
Client for cleaning text using a remote LLM API.
"""
from typing import Optional

from virtual_literature_companion.config import TEXT_CLEAN_LLM_CONFIG
from .request import make_llm_request, make_llm_request_async


system_prompt = """You are an expert in cleaning OCR text from books. Your task is to correct errors, remove extraneous elements, and format the text properly, while preserving the original content and structure.

Follow these rules:
- **Remove Headers and Footers:** Delete any page numbers, book titles, or chapter titles that appear at the top or bottom of the page.
- **Correct OCR Errors:** Fix common OCR mistakes, such as `hew as` to `he was`, `s coundrel` to `scoundrel`, or `oflf` to `off`.
- **Preserve Formatting:** Keep original formatting like italics or bold text, often marked with `*` or `_`.
- **Do Not Add New Content:** Do not add any words or sentences that were not in the original text. Your job is to clean, not to create. Content modification should be in the form of corrections, as well as removing headers/footers.
- **Fix Spacing and Line Breaks:** Correct spacing issues between words and ensure paragraphs are separated by a single newline. Do not get rid of newlines.

The raw text will be in `<RAW>` tags. Output the cleaned text between `<CLEANED>` tags. Include closing tag: `</CLEANED>`.
"""

example_messages = [
    {"role": "user", "content": """<RAW>
81 Hamlet ACT 2. SC. 2
FTLN 1046 And leads the will to desperate undertakings
FTLN 1047 As oft as any passions under heaven
FTLN 1048 That does afflict our natures. I am
</RAW>"""},
    {"role": "assistant", "content": """<CLEANED>
And leads the will to desperate undertakings
As oft as any passions under heaven
That does afflict our natures. I am
</CLEANED>"""},
    {"role": "user", "content": """<RAW>
Page 4
The anc ient libr ery contaned th0usands of
bpoks that hadbeen carefullypreserv3d for centuri3s. Many
of the volum3s werehandwritl
en manuscripts frommed ieval tim3s, their pag3s
yellowed withag3 but stilllegible to trained
sch0lars. Theheadlibrarian, Dr.Elisabeth
Hartwell, sp3nther daysc atal0guing these
preciousl3xts andens uringtheir prop3r
stor4ge inclimate-controlledenvironm3nts. She
of tenremarkedthateachbookwas likeatlm3
capsule, off3ringglimps3s intothethougYts and
beliefs ofp30ple froml0ng-forgott3n 3ras. Som3tim3s she
wouldflnd ancl3nt m4nuscripts w1th illuminat3d l3tt3rs that
gl0w3d lik3g3ms inth3
</RAW>"""},
    {"role": "assistant", "content": """<CLEANED>
The ancient library contained thousands of
books that had been carefully preserved for centuries. Many
of the volumes were handwritten
manuscripts from medieval times, their pages
yellowed with age but still legible to trained
scholars. The head librarian, Dr. Elisabeth
Hartwell, spent her days cataloguing these
precious texts and ensuring their proper
storage in climate-controlled environments. She
often remarked that each book was like a time
capsule, offering glimpses into the thoughts and
beliefs of people from long-forgotten eras. Sometimes she
would find ancient manuscripts with illuminated letters that
glowed like gems in the
</CLEANED>"""},
]


def clean_text(
    raw_text: str,
    max_tokens: Optional[int] = None,
    max_retries: int = 3,
    full_response: bool = False,
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

    prompt = f"<RAW>\n{raw_text}\n</RAW>"

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

    if full_response:
        return response_text

    if response_text:
        # Extract the cleaned text from between <cleaned> tags
        if "<CLEANED>" in response_text and "</CLEANED>" in response_text:
            start_tag = "<CLEANED>"
            end_tag = "</CLEANED>"
            start_idx = response_text.find(start_tag) + len(start_tag)
            end_idx = response_text.find(end_tag)
            response_text = response_text[start_idx:end_idx].strip()

        else:
            raise ValueError(f"Both <CLEANED> and </CLEANED> tags must be present in the response.")

        return response_text

    return None


async def clean_text_async(
    raw_text: str,
    max_tokens: Optional[int] = None,
    max_retries: int = 3,
    full_response: bool = False,
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

    prompt = f"<RAW>\n{raw_text}\n</RAW>"

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

    if full_response:
        return response_text

    if response_text:
        # Extract the cleaned text from between <cleaned> tags
        if "<CLEANED>" in response_text and "</CLEANED>" in response_text:
            start_tag = "<CLEANED>"
            end_tag = "</CLEANED>"
            start_idx = response_text.find(start_tag) + len(start_tag)
            end_idx = response_text.find(end_tag)
            response_text = response_text[start_idx:end_idx].strip()

        else:
            raise ValueError(f"Both <CLEANED> and </CLEANED> tags must be present in the response.")

        return response_text

    return None 