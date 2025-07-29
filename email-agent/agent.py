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
from project_types import EvaluationRollOut, Input
from tqdm.asyncio import tqdm_asyncio

load_dotenv()



@weave.op
async def agent_loop(
    input: Input, MAX_TURNS: int = 10, MAX_RETRIES: int = 2, 
    model_name: str = "Qwen/Qwen3-32B"
) -> List[dict]:

    if "gpt" in model_name:
        client = OpenAI(
            base_url=os.getenv("OPENAI_API_LINK"), api_key=os.getenv("OPENAI_API_KEY")
        )
    else:
        client = OpenAI(
            base_url=os.getenv("DEEPINFRA_API_LINK"), api_key=os.getenv("DEEPINFRA_API_KEY")
        )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": input.question},
    ]

    final_answer = None

    for _ in range(MAX_TURNS):
        for _ in range(MAX_RETRIES):
            try:
                response = client.chat.completions.create(
                    model=model_name,  # moonshotai/Kimi-K2-Instruct, deepseek-ai/DeepSeek-V3-0324
                    messages=messages,
                )

                response = response.choices[0].message.content

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

            tool_result = await call_tool(input.inbox_address, tool_call)

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

    print("--------------------------------")
    print("TRACE")
    print(messages[1:])
    print("--------------------------------")
    
    return final_answer if final_answer is not None else "Error: No answer found"


async def reward_agent_response(rollout: EvaluationRollOut, MAX_RETRIES: int = 3, reward_model_name: str = "deepseek-ai/DeepSeek-V3-0324") -> int:

    if "gpt" in reward_model_name:
        client = OpenAI(
            base_url=os.getenv("OPENAI_API_LINK"), api_key=os.getenv("OPENAI_API_KEY")
        )
    else:
        client = OpenAI(
            base_url=os.getenv("DEEPINFRA_API_LINK"), api_key=os.getenv("DEEPINFRA_API_KEY")
        )

    scenario = f"Question: {rollout.question} \n Answer: {rollout.agent_answer} \n Golden Answer: {rollout.golden_answer}"

    print("--------------------------------")
    print("SCENARIO")
    print(scenario)
    print("--------------------------------")

    messages = [
        {"role": "system", "content": REWARD_MODEL_SYSTEM_PROMPT},
        {"role": "user", "content": scenario},
    ]

    final_reward = 0.0

    for _ in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=reward_model_name,  # moonshotai/Kimi-K2-Instruct, deepseek-ai/DeepSeek-V3-0324
                messages=messages,
            )

            response = response.choices[0].message.content
            reward = parse_from_response(response, "reward")  # type: ignore

            try:
                reward = float(reward)
                final_reward = reward
                break

            except Exception as e:
                print(f"Error: The judge model returned non binary value - {e}")

        except Exception as e:
            print(f"Error: API Error with Judge Model - {e}")

    return final_reward


