def dedent(text: str):
    """
    Remove leading and trailing whitespace for each line
    For example:
        ```
        Line 1 has no leading space
            Line 2 has two leading spaces
        ```
        The output will be :
        ```
        Line 1 has no leading space
        Line 2 has two leading spaces
        ```
    This allows writing cleaner multiline prompts in the code.
    """
    return "\n".join([line.strip() for line in text.split("\n")])