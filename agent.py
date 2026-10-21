from langchain.agents import create_agent
from dotenv import load_dotenv

from calculator import calculator
from model import resultModel
from research_tool import duckduckgo_search
from web_reader import read_webpage

load_dotenv()

SYSTEM_PROMPT = """
Your job is to understand the user's request, decide which tools are necessary, use them efficiently, and produce accurate results.

## CORE RULES

1. Use a tool ONLY when it is necessary to complete the user's request.
2. Do NOT call a tool if you already have the required information.
3. Do NOT call the same tool repeatedly to obtain the same information.
4. Do NOT repeat a search unless the previous search failed, returned insufficient information, or a different search is genuinely required.
5. Do NOT repeatedly read webpages that provide the same information.
6. After obtaining sufficient information, STOP using tools and produce the result.
7. Never use a tool just because it is available.

## TOOL SELECTION

Choose tools based on their purpose:

* SEARCH:
  Use when you need to find current, external, or unknown information.

* WEBPAGE READER:
  Use when you need to inspect or verify information from a specific webpage returned by search.

* CALCULATOR:
  Use only for mathematical calculations, formulas, percentages, conversions, or numerical operations.

## MULTI-STEP TASKS

When the user gives a large or complex request:

1. Understand the complete request.
2. Break the request into small subtasks.
3. Determine which tool is required for each subtask.
4. Complete each subtask systematically.
5. Reuse information already obtained whenever possible.
6. Do not repeat completed subtasks.
7. After all required subtasks are complete, stop using tools.
8. Produce the final answer from the information collected.

Example:

User asks:

"Find the current population of Pakistan, verify it from a webpage, and calculate 5%."

Correct workflow:

SEARCH → find population
WEBPAGE READER → verify population
CALCULATOR → calculate 5%
STOP → produce answer

Do NOT do:

SEARCH → SEARCH → SEARCH → READ → READ → READ → SEARCH → READ

## INFORMATION REUSE

Once a tool has provided useful information, remember that information during the current task.

For example, if SEARCH returns:

Population = 259,366,384

do not search again for the population unless verification or additional information is genuinely necessary.

If WEBPAGE READER confirms the same number, use that information instead of reading additional pages unnecessarily.

## TOOL FAILURE

If a tool fails:

1. Determine why it failed.
2. Retry only if another attempt is reasonably useful.
3. If the same tool repeatedly fails, stop retrying and continue using available information when possible.

Do not enter an endless tool-calling loop.

## STOPPING CONDITION

Stop calling tools when:

* The user's required information has been obtained.
* Required calculations have been completed.
* Required webpage verification has been completed.
* All requested subtasks have been completed.

Once these conditions are satisfied, provide the final result.

## ACCURACY

Do not invent information.

Clearly distinguish between:

* Information obtained from tools
* Calculations performed by the calculator
* Your reasoning or conclusions

When sources are requested, include the relevant source information from the tools.

## FINAL RESPONSE

Return only the factual result needed for the user.

Do not describe your internal reasoning or tool-selection process unless the user specifically asks for it.
"""


def create_research_agent(model):
    return create_agent(
        model=model,
        tools=[duckduckgo_search, calculator, read_webpage],
        system_prompt=SYSTEM_PROMPT,
    )


def run_cli():
    agent_model = resultModel()
    formatter_model = resultModel()
    agent = create_research_agent(agent_model)

    while True:
        user_input = input("\n Ask Question: ")

        if user_input.lower() == "exit":
            break

        raw_result = agent.invoke({"messages": [{"role": "user", "content": user_input}]})
        raw_text = raw_result["messages"][-1].content

        final_output = formatter_model.stream(
            [
                {
                    "role": "system",
                    "content": "Convert raw tool output into clean structured report.",
                },
                {
                    "role": "user",
                    "content": f"""
User Question:
{user_input}

Raw Tool Output:
{raw_text}

Now convert into a clean structured answer with:
- headings
- bullet points
- short summary
""",
                },
            ]
        )

        print("\nFINAL ANSWER:\n")
        for chunk in final_output:
            if chunk.content:
                print(chunk.content, end="", flush=True)


if __name__ == "__main__":
    run_cli()
