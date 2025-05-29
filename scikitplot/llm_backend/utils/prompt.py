"""prompt."""

# llm_backend/utils/prompt.py

from jinja2 import Environment, StrictUndefined, TemplateError


def render_prompt(
    template_str: str, variables: "dict[str, object]", strict: bool = False
) -> str:
    """
    Render a prompt template string with provided variables using Jinja2.

    Args:
        template_str (str): The Jinja2 template string.
        variables (Dict[str, object]): A dictionary of variables to fill in the template.
        strict (bool): If True, undefined variables will raise errors (default: False).

    Returns
    -------
    str :
        The rendered prompt.

    Raises
    ------
    TemplateError :
        If rendering fails due to template syntax
        or undefined variables in strict mode.
    """
    # template = Template(template_str)
    env = Environment(undefined=StrictUndefined if strict else None, autoescape=True)
    template = env.from_string(template_str)
    return template.render(**variables)


# Sample template string
SAMPLE_TEMP = """
You are an AI assistant. Your task is to help with {{ task }}.

User Query: {{ query }}

Please provide a detailed response.
"""

# Sample usage
if __name__ == "__main__":
    variables = {
        "task": "summarization",
        "query": "Explain the importance of data normalization in machine learning.",
    }

    try:
        rendered_prompt = render_prompt(SAMPLE_TEMP, variables, strict=True)
        print(rendered_prompt)  # noqa: T201
    except TemplateError as e:
        print(f"Template rendering error: {e}")  # noqa: T201
