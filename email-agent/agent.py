import os
import asyncio
from fsspec import asyn
from numpy import roll
import weave
from typing import List
from openai import OpenAI
from utils import (
    call_tool,
    SYSTEM_PROMPT,
    parse_from_response,
    REWARD_MODEL_SYSTEM_PROMPT,
)
from rich import print
from dotenv import load_dotenv
import json
from project_types import RollOut
from tqdm.asyncio import tqdm_asyncio

load_dotenv()


DATA_JSONL = [
    {
        "question": "What is the purpose of the Chinese Wall policy?",
        "answer": "The Chinese Wall policy serves to reduce the risk of insider trading liability by keeping material non-public information walled off from our equity trading personnel.",
        "realistic_score": "0.9",
        "email_date": "1980-01-01 00:00:00",
    },
    {
        "question": "Who should I contact if I have questions about the Chinese Wall policy?",
        "answer": "You can call Lance Schuler at 35419, Bob Bruce at 57780, or Janette Elbertson at 36544 if you have any questions concerning the policy or the role of the Resource Group.",
        "realistic_score": "0.8",
        "email_date": "1980-01-01 00:00:00",
    },
    {
        "question": "When and where is the Chinese Wall training scheduled?",
        "answer": "The Chinese Wall training is scheduled on Monday, March 5, 2001, at various times, and will be held at the downtown Hyatt Regency Hotel in Sandalwood Rooms A & B.",
        "realistic_score": "0.9",
        "email_date": "1980-01-01 00:00:00",
    },
    {
        "question": "How should I register for the Chinese Wall training?",
        "answer": "Please confirm your attendance at one of the sessions with Brenda Whitehead by emailing her at brenda.whitehead@enron.com or calling her at extension 3-5438.",
        "realistic_score": "0.8",
        "email_date": "1980-01-01 00:00:00",
    },
    {
        "question": "What is the proposal for legal staffing on the Internet Project?",
        "answer": "The proposal is to use Clifford Chance as project manager and to use normal outside Continental counsel for both the internet issues and the trading/commodity issues.",
        "realistic_score": "0.7",
        "email_date": "1999-05-06 09:23:00",
    },
]


@weave.op
async def agent_loop(
    inbox: str, question: str, MAX_TURNS: int = 10, MAX_RETRIES: int = 2
) -> List[dict]:

    DE_client = OpenAI(
        base_url=os.getenv("DEEPINFRA_API_LINK"), api_key=os.getenv("DEEPINFRA_API_KEY")
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    final_answer = ""

    for _ in range(MAX_TURNS):
        for _ in range(MAX_RETRIES):
            try:
                response = DE_client.chat.completions.create(
                    model="mistralai/Mistral-Small-3.2-24B-Instruct-2506",  # moonshotai/Kimi-K2-Instruct, deepseek-ai/DeepSeek-V3-0324
                    messages=messages,
                )

                response = response.choices[0].message.content

                # print(f"Response: {response}")

                thinking = parse_from_response(response, "think")  # type: ignore
                tool_call = parse_from_response(response, "tool")  # type: ignore

                if tool_call:
                    try:
                        tool_call = json.loads(tool_call)

                    except Exception as e:
                        print(f"Tool Call Error: {e}")
                        tool_call = None

                answer = parse_from_response(response, "answer")  # type: ignore

                if thinking or tool_call or answer:
                    break

            except Exception as e:
                print(f"Error: {e}")

        if thinking:  # type: ignore
            thinking = thinking.strip()

        if tool_call:  # type: ignore

            tool_result = await call_tool(inbox, tool_call)

            assistant_content = (
                f"<think>\n{thinking}\n</think>\n" if thinking else ""
            ) + f"<tool>{json.dumps(tool_call)}</tool>"

            # print("TOOL CALL")
            # print({"role": "assistant", "content": assistant_content})
            # print({"role": "user", "content": tool_result})

            messages.extend(
                [
                    {"role": "assistant", "content": assistant_content},
                    {"role": "user", "content": tool_result},
                ]
            )

        elif answer:  # type: ignore
            assistant_content = (
                f"<think>\n{thinking}\n</think>\n" if thinking else ""
            ) + f"<answer>{answer}</answer>"

            # print("ANSWER")
            # print(assistant_content)

            final_answer = answer
            messages.append({"role": "assistant", "content": assistant_content})

            break

        else:
            print("Error: No tool call or answer found")
            break

    return final_answer


async def reward_agent_response(rollout: RollOut, MAX_RETRIES: int = 3) -> int:
    DE_client = OpenAI(
        base_url=os.getenv("DEEPINFRA_API_LINK"), api_key=os.getenv("DEEPINFRA_API_KEY")
    )

    scenario = f""" 
        Question: {rollout.question}
        Answer: {rollout.agent_answer}
        Golden Answer: {rollout.golden_answer}
    """

    print(f"SCENARIO: \n {scenario}")

    messages = [
        {"role": "system", "content": REWARD_MODEL_SYSTEM_PROMPT},
        {"role": "user", "content": scenario},
    ]

    final_reward = -1

    for _ in range(MAX_RETRIES):
        try:
            response = DE_client.chat.completions.create(
                model="mistralai/Mistral-Small-3.2-24B-Instruct-2506",  # moonshotai/Kimi-K2-Instruct, deepseek-ai/DeepSeek-V3-0324
                messages=messages,
            )

            response = response.choices[0].message.content
            reward = parse_from_response(response, "reward")  # type: ignore

            try:
                reward = float(reward)

            except Exception as e:
                reward = -1
                print("Error: The Judge Model return non binary value")

            if reward:
                final_reward = reward
                break

        except Exception as e:
            print(f"Error: {e}")

    return final_reward


