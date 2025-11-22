import streamlit as st
import os
from dotenv import load_dotenv
import google.generativeai as genai

from pygments.lexers import guess_lexer
from pygments.util import ClassNotFound
import re
import html  # used for safe escaping if needed
from streamlit_ace import st_ace

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


# -------------------------------------------------------
# 🌟 SUPPORTED LANGUAGES
# -------------------------------------------------------

SUPPORTED_LANGUAGES = {
    "c": "C", "c++": "C++", "c#": "C#",  "python": "Python", "java": "Java", "javascript": "JavaScript",
    "typescript": "TypeScript", "php": "PHP", "go": "Go", "ruby": "Ruby", "swift": "Swift",
    "kotlin": "Kotlin", "r": "R", "sql": "SQL", "html": "HTML", "css": "CSS", "dart": "Dart",
    "rust": "Rust", "perl": "Perl", "scala": "Scala", "lua": "Lua", "matlab": "MATLAB",
    "assembly": "Assembly",
}

# Human‑friendly Gemini model names mapped to actual API model IDs.
GEMINI_MODELS = {
    "Gemini 2.0 Flash": "gemini-2.0-flash",
    "Gemini 2.0 Flash-Lite": "gemini-2.0-flash-lite",
    "Gemini 2.5 Pro": "gemini-2.5-pro",
    "Gemini 2.5 Flash-Lite": "gemini-2.5-flash-lite",
}


# Normalization map so every detector (rules, Pygments, Gemini, dropdown)
# is converted to a single canonical key from SUPPORTED_LANGUAGES.
LANG_ALIASES = {
    "c": "c", "c language": "c",
    "cpp": "c++", "c++": "c++","c plus plus": "c++",
    "csharp": "c#", "c#": "c#", "c sharp": "c#",
    "py": "python", "python": "python",
    "java": "java",
    "js": "javascript", "javascript": "javascript", "node": "javascript",
    "ts": "typescript", "typescript": "typescript",
    "php": "php",
    "go": "go", "golang": "go",
    "rb": "ruby", "ruby": "ruby",
    "swift": "swift",
    "kt": "kotlin", "kotlin": "kotlin",
    "r": "r", "r language": "r", "s": "r",  # Pygments may label R as "S"
    "sql": "sql",
    "html": "html", "htm": "html",
    "css": "css",
    "dart": "dart",
    "rust": "rust", "rs": "rust",
    "perl": "perl", "pl": "perl",
    "scala": "scala",
    "lua": "lua",
    "matlab": "matlab", "m file": "matlab",
    "assembly": "assembly",
}


def normalize_language_name(name: str) -> str:
    """
    Convert any detector / dropdown language label into a canonical key
    from SUPPORTED_LANGUAGES (e.g. 'C++', 'cpp' → 'c++').
    """
    if not name:
        return ""

    name = name.strip().lower()

    # direct alias match first
    if name in LANG_ALIASES:
        return LANG_ALIASES[name]

    # match against both dict keys and human‑readable values
    for key, display in SUPPORTED_LANGUAGES.items():
        if name == key or name == display.lower():
            return key

    return name


# -------------------------------------------------------
# 🌟 SAFE GEMINI CALL (Prevents Crashes on API 429)
# -------------------------------------------------------

def safe_gemini_call(prompt, model_name="gemini-2.0-flash"):
    try:
        model = genai.GenerativeModel(model_name)
        resp = model.generate_content(prompt)
        return resp.text.strip()

    except Exception as e:
        return f"__API_ERROR__::{str(e)}"


# -------------------------------------------------------
# 🌟 ADVANCED LANGUAGE DETECTOR
# -------------------------------------------------------


def detect_language(code_text: str) -> str:
    """
    Try multiple strategies (rules → Pygments → Gemini) and always return
    a canonical key from SUPPORTED_LANGUAGES, or 'unknown' if not sure.
    """
    text_lower = code_text.lower()

    # --- Strong early detection for R --------------------------------------
    # Many R scripts are mis‑detected as Python by lexers, so we look for
    # very characteristic R syntax first. We intentionally AVOID using the
    # generic "<-" token here because Scala and some other languages also
    # use it (e.g. for‑comprehensions), which previously caused Scala to be
    # labelled as R.
    r_sigs = [
        "library(",
        "ggplot(",
        "data.frame(",
        "dplyr::",
        "tidyverse",
        "read.csv(",
        "set.seed(",
        "matrix(",
        "cbind(",
        "rbind(",
    ]
    if any(sig in text_lower for sig in r_sigs):
        return "r"

    # Secondary but safe R detection using the '<-' assignment operator.
    # Many R scripts (especially simple ones) use '<-' without importing
    # common libraries. We still avoid mis-detecting Scala for-comprehensions
    # like: for (x <- xs) yield ...
    if "<-" in text_lower:
        looks_scala = any(
            sig in text_lower
            for sig in [
                "for (",          # Scala for-comprehensions often use this with '<-'
                " yield",         # 'yield' keyword strongly hints Scala
                "case class ",
                "extends app",
                "object ",
            ]
        )
        if not looks_scala:
            return "r"

    # --- Strong early detection for Kotlin ---------------------------------
    # Keep this conservative: only Kotlin‑specific constructs, to avoid
    # mis‑detecting Java / Go / Swift / JS that also use "var", "println", etc.
    kotlin_sigs = [
        "fun main(",
        "fun main ()",
        "data class ",
        "companion object",
        "when (",
        "?.",
        " !!",
        "intarrayof(",
    ]
    if any(sig in text_lower for sig in kotlin_sigs):
        return "kotlin"

    # Heuristic signatures for fast, local detection for all languages.
    # Order matters: more specific languages (like C++ / TypeScript) are checked
    # before more generic ones (like Java / JavaScript) to reduce overlap.
    rules = [
        (
            "C++",
            [
                "#include <iostream>",
                "#include <vector>",
                "std::",
                "cout <<",
                "cin >>",
                "vector<",
                "using namespace std",
            ],
        ),
        ("C", ["#include <stdio.h>", "printf(", "scanf("]),
        ("C#", ["using system", "console.writeline"]),
        (
            "Python",
            [
                # Deliberately exclude extremely generic tokens like "def "
                # and "print(" because they appear in many languages
                # (Ruby, Lua, MATLAB, etc.). We lean on imports and "self"
                # which are more Python‑specific.
                "self",
                "import os",
                "import sys",
                "import numpy",
                "import pandas",
                "def __init__(",
            ],
        ),
        (
            "Java",
            [
                "public static void main",
                "public class ",
                "system.out.println",
                "import java.",
                " class ",  # keep generic but Java is checked after C/C#/C++
            ],
        ),
        # TypeScript: checked BEFORE JavaScript with TS‑specific syntax only,
        # so that plain JS isn't mis‑classified. We avoid the bare token
        # "type " because it appears in HTML attributes (type="text") and
        # previously caused HTML to be flagged as TypeScript.
        (
            "TypeScript",
            [
                "interface ",
                "enum ",
                ": string",
                ": number",
                ": boolean",
                ": any",
                ": unknown",
                ": void",
                ": never",
                "implements ",
                "readonly ",
                "record<",
                "partial<",
                "pick<",
                "omit<",
                "number[]",
                "string[]",
                "boolean[]",
                "array<",
                ": promise<",
                " as const",
                " satisfies ",
            ],
        ),
        # JavaScript: plain JS constructs (TS is checked above). We exclude
        # the very generic "function " token, which appears in many languages
        # (PHP, MATLAB, JS in HTML, etc.) and was causing heavy
        # mis‑classification.
        (
            "JavaScript",
            [
                "console.log",
                "=>",
                "module.exports",
                "require(",
                "document.getelementbyid(",
                "export default ",
            ],
        ),
        ("PHP", ["<?php", "echo ", "$_POST"]),
        ("CSS", ["color:", "background:", "font-size:", "margin:", "padding:"]),
        ("Go", ["package main", "fmt.println", "func main"]),
        ("Rust", ["fn main", "let mut", "println!"]),
        ("SQL", ["select ", "insert ", "update ", "from ", "where "]),
        ("HTML", ["<html", "<div", "<body"]),
        (
            "Ruby",
            [
                "puts ",
                "require '",
                "require_relative",
                " => ",  # hash‑rocket
                "elsif",
            ],
        ),
        (
            "Swift",
            [
                # Restrict to Swift‑specific imports / keywords so that
                # Scala (which also uses "var") is not mis‑detected as Swift.
                "import swiftui",
                "import uikit",
                "func ",
            ],
        ),
        (
            "Lua",
            [
                "local ",
                "then",
                "elseif",
                "--",
            ],
        ),
        (
            "MATLAB",
            [
                "end;",
                "end %",
                "%",  # line comment
                "plot(",
                "function ",
            ],
        ),
        ("Assembly", ["mov ", "add ", "jmp", "cmp"]),
        (
            "R",
            [
                # See R early‑detection notes above: avoid bare "<-" to stop
                # colliding with Scala for‑comprehensions.
                "library(",
                "ggplot(",
                "data.frame(",
            ],
        ),
        (
            "Dart",
            [
                "import 'dart:",
                "import 'package:",
                "void main()",
                "future<",
                "@override",
                " extends statelesswidget",
                " extends statefulwidget",
            ],
        ),
        ("Perl", ["use strict", "my $", "print \""]),
        (
            "Scala",
            [
                "object ",
                "extends app",
                "case class ",
                "Option[",
                "List[",
                "scala.io.stdin",
                "implicit ",
            ],
        ),
    ]

    for lang, sigs in rules:
        if any(sig in text_lower for sig in sigs):
            return normalize_language_name(lang)

    # Pygments lexer as a second opinion
    try:
        lexer = guess_lexer(code_text)
        detected = lexer.name.lower()
        normalized = normalize_language_name(detected)
        if normalized in SUPPORTED_LANGUAGES:
            return normalized
    except Exception:
        pass

    # Gemini fallback (safe). We still normalize its answer.
    ai_resp = safe_gemini_call(
        f"Detect only the programming language name of this code. "
        f"Reply with just the language name, nothing else.\n\n{code_text}"
    )

    if "__API_ERROR__" in ai_resp:
        return "unknown"

    ai_resp_lower = ai_resp.lower()

    for key, display in SUPPORTED_LANGUAGES.items():
        if key in ai_resp_lower or display.lower() in ai_resp_lower:
            return key

    # last resort: try normalizing the raw response
    normalized = normalize_language_name(ai_resp_lower)
    if normalized in SUPPORTED_LANGUAGES:
        return normalized

    return "unknown"


# -------------------------------------------------------
# 🌟 STRUCTURAL CHECKER (local)
# -------------------------------------------------------

def structural_check_and_fix(code):
    """
    Scan code for unbalanced tokens: {}, (), [], quotes, and block comments.
    Returns (errors_list, corrected_code).
    errors_list: list of tuples (line_number, message)
    corrected_code: original code with added closing tokens appended if needed
    """
    stack = []  # items: (char, line_number)
    errors = []
    in_single_quote = False
    in_double_quote = False
    in_block_comment = False
    lines = code.splitlines()
    total_lines = len(lines)

    pairs = {"{": "}", "(": ")", "[": "]"}
    opening = set(pairs.keys())
    closing = {v: k for k, v in pairs.items()}

    for lineno, line in enumerate(lines, start=1):
        j = 0
        while j < len(line):
            ch = line[j]

            # detect start/end of block comment (/* ... */)
            if not in_single_quote and not in_double_quote:
                if not in_block_comment and ch == "/" and j + 1 < len(line) and line[j+1] == "*":
                    in_block_comment = True
                    j += 2
                    continue
                if in_block_comment and ch == "*" and j + 1 < len(line) and line[j+1] == "/":
                    in_block_comment = False
                    j += 2
                    continue
            if in_block_comment:
                j += 1
                continue

            # detect line comment // -> skip rest of line
            if not in_single_quote and not in_double_quote and ch == "/" and j + 1 < len(line) and line[j+1] == "/":
                break

            # quotes handling (don't treat braces inside strings)
            if not in_block_comment:
                if ch == "'" and not in_double_quote:
                    escaped = (j > 0 and line[j-1] == "\\")
                    if not escaped:
                        in_single_quote = not in_single_quote
                elif ch == '"' and not in_single_quote:
                    escaped = (j > 0 and line[j-1] == "\\")
                    if not escaped:
                        in_double_quote = not in_double_quote

            # if inside any quote, skip token logic
            if in_single_quote or in_double_quote or in_block_comment:
                j += 1
                continue

            # push opening tokens
            if ch in opening:
                stack.append((ch, lineno))
            elif ch in closing:
                if stack and stack[-1][0] == closing[ch]:
                    stack.pop()
                else:
                    # unmatched closing token
                    msg = f"unmatched '{ch}' at line {lineno}"
                    errors.append((lineno, msg))
            j += 1

    # after scanning lines, check for unclosed quotes or block comment
    if in_single_quote:
        errors.append((total_lines, "unclosed single quote (')"))
    if in_double_quote:
        errors.append((total_lines, 'unclosed double quote (")'))
    if in_block_comment:
        errors.append((total_lines, "unclosed block comment (/* */)"))

    # any remaining openings in stack are missing corresponding closing tokens
    if stack:
        for open_tok, open_line in reversed(stack):
            needed = pairs.get(open_tok, None)
            if needed:
                errors.append((open_line, f"missing closing '{needed}' for '{open_tok}' opened here"))

    # build corrected code by appending needed closers in correct order
    corrected = code
    if stack:
        if not corrected.endswith("\n"):
            corrected += "\n"
        for open_tok, _ in reversed(stack):
            needed = pairs.get(open_tok, "")
            corrected += needed
        corrected += "\n"

    return errors, corrected


# -------------------------------------------------------
# 🌟 AI ERROR DETECTION (updated to use structural checker)
# -------------------------------------------------------

def detect_errors(code, language, model_name="gemini-2.0-flash"):
    """
    Analyze the entire code snippet and generate a single error report that
    includes only **meaningful errors**:
    - Syntax / parsing errors
    - Type / compilation errors (anything that would stop the compiler)
    - Runtime exceptions (clearly impossible or very likely in normal use,
      for example guaranteed division by zero, always‑out‑of‑bounds index,
      dereferencing a null pointer, etc.)
    - Semantic / logical errors where the code definitely produces the wrong
      result (wrong variable, off‑by‑one in a simple loop, obviously wrong
      condition, etc.).

    Pure style issues (like missing final newlines, choice of variable names,
    or formatting), micro‑optimizations, and speculative logic concerns that
    might still be correct SHOULD NOT be reported as errors.

    We run a fast local structural check (braces, parens, brackets, quotes,
    block comments) and pass those structural issues as hints to Gemini, but
    the final error list + corrected code always comes from a single Gemini
    call so it can fix *all* issues in one corrected version.
    """
    # 1) local structural check over the whole code (used as hints only)
    struct_errors, _ = structural_check_and_fix(code)
    struct_hint = ""
    if struct_errors:
        hint_lines = []
        for line_no, msg in struct_errors:
            hint_lines.append(f"Line {line_no}: {msg}")
        struct_hint = (
            "The following structural issues (unbalanced braces, quotes, "
            "or comments) were detected by a local parser. "
            "Make sure these are ALSO included in your error list:\n"
            + "\n".join(hint_lines)
            + "\n\n"
        )

    # 2) always ask Gemini once over the whole code
    lang_key = normalize_language_name(str(language))
    lang_label = "x86 Assembly" if lang_key == "assembly" else language

    asm_extra = ""
    if lang_key == "assembly":
        asm_extra = (
            "The code is low-level x86 Assembly. Treat it as a standalone snippet: "
            "do NOT require extra sections, startup boilerplate, or OS-specific "
            "conventions to consider it syntactically valid. Only report REAL "
            "assembler-level syntax errors that would cause an assembler/linker to "
            "fail (for example: missing commas where required, invalid/misspelled "
            "mnemonics, invalid register names, wrong operand count or types for an "
            "instruction, or references to undefined labels). Do NOT treat potential "
            "logic issues, style issues, or non-idiomatic patterns as errors. If you "
            "are not sure whether a line is truly invalid for the assembler, assume "
            "it is valid and say 'No errors found'. After listing all syntax errors, "
            "provide a fully corrected version of the same Assembly code that would "
            "assemble successfully.\n\n"
        )

    # Build an explicitly numbered version of the code so the model’s
    # line numbers always match what the user sees in the editor,
    # including blank lines.
    numbered_lines = []
    for idx, line in enumerate(code.splitlines(), start=1):
        numbered_lines.append(f"{idx}: {line}")
    # handle empty snippet gracefully
    numbered_code = "\n".join(numbered_lines) if numbered_lines else "1:"

    prompt = f"""
You are a coding assistant. Carefully analyze the entire {lang_label} code below
and find **all REAL errors** in the whole snippet that make the program
incorrect or impossible to compile/run.

ONLY treat the following as errors:
- Syntax or parsing errors
- Type / compilation errors
- Clearly impossible runtime errors (for example, guaranteed division by zero)
- CLEAR logical errors where you can be highly confident the code is wrong,
  such as:
  - Using the wrong variable in an expression, comparison or assignment
  - A loop that obviously never executes when it should
  - Off‑by‑one mistakes in simple index loops (e.g. iterating 0..n but accessing a[n])
  - A condition that is always false or always true given the nearby code

DO NOT treat the following as errors:
- Style issues or code formatting (including missing newline characters
  at the end of strings or at the end of the file)
- Performance / optimization suggestions
- Possible or hypothetical logic bugs where the program could still be correct
- Missing comments, edge‑case checks, or input validation

If you are not **certain** that something is truly an error of these kinds,
ASSUME THE CODE IS CORRECT and do not report it.

{asm_extra}{struct_hint}If ANY valid errors exist → ALWAYS use THIS FORMAT (each error on its own lines exactly as below),
and include **every** error you can find in this single report:
Error in Line <number>:
<describe error>

Error in Line <number>:
<describe error>

(Corrected code must come AFTER all error lines)

Corrected Code:
<corrected full version of the code with ALL errors fixed>

If there are absolutely NO errors in the code, reply with EXACTLY this sentence
and nothing else:
No errors found

Here is the code with line numbers on the left. These line numbers are the ONLY
line numbers you should use. Do NOT renumber the code, do NOT skip blank lines,
and do NOT invent your own numbering. When you say "Error in Line N", N MUST be
one of the numbers shown on the left below:

{numbered_code}
"""

    resp = safe_gemini_call(prompt, model_name=model_name)
    return resp



# -------------------------------------------------------
# 🌟 EXPLAIN CODE (ONLY IF NO ERRORS)
# -------------------------------------------------------

def explain_code(code_content, language, model_name="gemini-2.0-flash"):
    lang_key = normalize_language_name(str(language))
    lang_label = "x86 Assembly" if lang_key == "assembly" else language

    asm_extra = ""
    if lang_key == "assembly":
        asm_extra = (
            "The code is low-level Assembly. Explain it line by line. Preserve the exact "
            "original line text in each 'Line:' field (including labels and comments), "
            "and then give a very simple English explanation in the 'Explanation:' field.\n\n"
        )

    prompt = f"""
Explain the following {lang_label} code line by line.

Format:
Line: <code>
Explanation: <simple words>

NO markdown. NO JSON.

{asm_extra}

Code:
{code_content}
"""

    return safe_gemini_call(prompt, model_name=model_name)


# -------------------------------------------------------
# 🌟 FOOTER
# -------------------------------------------------------


def render_footer():
    """Render a footer at the bottom of the page (always visible)."""
    st.markdown(
        """
        <div class="app-footer">
            <p class="footer-text">
                Copyright © by
                <span class="footer-name">Binod Kapadi</span>
            </p>
            <p class="footer-text">All Rights Reserved — 2025</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -------------------------------------------------------
# 🌐 STREAMLIT UI
# -------------------------------------------------------

def main():

    st.set_page_config(page_title="💻 Code Explainer", layout="wide")
    st.title("💻 Line-by-Line Code Explainer")

    if os.path.exists("code.css"):
        with open("code.css") as f:
            st.markdown("<style>" + f.read() + "</style>", unsafe_allow_html=True)

    # Render footer once per run (it's fixed-position, so DOM position
    # doesn't matter, and this ensures it never disappears on button clicks)
    render_footer()

    st.markdown("**📝 Paste your code here:**")

    # Desktop-friendly editor with line numbers (Ace).
    ace_code = st_ace(
        placeholder="Write or paste your code here...",
        language="text",
        theme="tomorrow_night_bright",
        key="code_editor",
        height=300,
        show_gutter=True,
        show_print_margin=False,
        wrap=True,
        auto_update=True,
    ) or ""

    # Mobile-friendly plain text area that uses the browser's native
    # paste / select-all behavior (more reliable on phones).
    simple_code = st.text_area(
        "📱 Or paste here (better on mobile):",
        value="",
        height=220,
        key="code_editor_mobile",
    )

    # Prefer whatever the user actually used: if the mobile box has
    # content, use that; otherwise fall back to the Ace editor.
    code = (simple_code or "").strip("\n") or ace_code

    # Side‑by‑side selectors: language (left) and Gemini model (right).
    col_lang, col_model = st.columns(2)
    with col_lang:
        selected_lang = st.selectbox(
            "🖥️ Select programming language:",
            list(SUPPORTED_LANGUAGES.values()),
        )
    with col_model:
        selected_model_label = st.selectbox(
            "🤖 Choose Gemini model:",
            list(GEMINI_MODELS.keys()),
            index=0,
        )
    selected_model_name = GEMINI_MODELS[selected_model_label]

    if st.button("🚀 Explain Code"):

        if not code.strip():
            st.warning("⚠️ Please paste some code.")
            return

        detected_key = detect_language(code)
        selected_key = normalize_language_name(selected_lang)

        # If we are reasonably sure about the pasted language and it does not
        # match the user selection, block further processing.
        if detected_key != "unknown" and selected_key and detected_key != selected_key:
            detected_label = SUPPORTED_LANGUAGES.get(detected_key, detected_key).upper()
            st.error(
                f"❌ You pasted **{detected_label}** code but selected **{selected_lang}** language."
                "\nPlease select the correct language to proceed further."
            )
            return

        # ---------------------------------------------------
        # STEP 1 — CHECK ERRORS
        # ---------------------------------------------------

        st.info("🔍 Checking for errors…")

        error_data = detect_errors(code, selected_lang, model_name=selected_model_name)

        if "__API_ERROR__" in error_data:
            st.error("🚨 Gemini API limit exceeded. Please try again later.")
            st.code(error_data)
            return

        lower_err = error_data.lower()
        no_error_pattern = bool(re.search(r"\bno[_ ]?error(s)?\b", lower_err) or
                                "no errors found" in lower_err or
                                "no error found" in lower_err)

        if not no_error_pattern:
            st.error("❌ Errors found in your code:")

            cleaned = error_data
            cleaned = cleaned.replace("```</div>", "")
            cleaned = cleaned.replace("</div>", "")
            cleaned = re.sub(r"<div[^>]*>", "", cleaned)
            cleaned = re.sub(r"`{3,}\s*$", "", cleaned, flags=re.MULTILINE)

            parts = re.split(r"Corrected Code:\s*", cleaned, maxsplit=1, flags=re.IGNORECASE)
            errors_only = parts[0].strip()
            corrected_block = parts[1].strip() if len(parts) > 1 else ""

            # highlight only the "Error in Line N:" tokens in the error text
            errors_only = re.sub(
                r"(Error in Line \d+:)",
                r"<span class='error-highlight'>\1</span>",
                errors_only,
            )

            st.markdown(
                f"<div class='error-box'>{errors_only}</div>", unsafe_allow_html=True
            )

            # Show corrected code below the error box (separate dark code block)
            if corrected_block:
                st.markdown("**Corrected Code:**")
                st.code(corrected_block)

            return

        # ---------------------------------------------------
        # STEP 2 — EXPLAIN ONLY IF NO ERRORS
        # ---------------------------------------------------

        st.success("✅ No errors found! Generating explanation…")

        explanation = explain_code(code, selected_lang, model_name=selected_model_name)

        if "__API_ERROR__" in explanation:
            st.error("🚨 Gemini API limit exceeded. Try again later.")
            return

        pattern = r"Line:\s*(.*?)\s*Explanation:\s*(.*?)(?=Line:|$)"
        matches = re.findall(pattern, explanation, re.DOTALL)

        if not matches:
            st.code(explanation)
            return

        # show parsed line-by-line explanation
        for line, exp in matches:
            st.markdown(f"<div class='code-line'>{line.strip()}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='explanation'>💡 {exp.strip()}</div>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()