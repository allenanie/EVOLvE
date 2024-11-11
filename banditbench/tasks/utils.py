def dedent(text: str):
    # remove leading and trailing whitespace for each line
    return "\n".join([line.strip() for line in text.split("\n")])