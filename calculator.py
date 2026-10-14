from langchain.tools import tool


@tool
def calculator(expression: str) -> str:
    """
       Use this tool to solve mathematical expressions.

    Input: a valid math expression (e.g., 2+2, (5*3)/2)
    Output: numeric result

    Use ONLY for:
    - arithmetic
    - equations
    - conversions
    - formulas

    Do NOT use for general reasoning or facts.
    """
    print(">>> CALCULATOR TOOL CALLED")

    try:

        result = eval(expression, {"__builtins__": {}}, {})

        return str(result)

    except Exception as e:

        return f"Error: {str(e)}"
