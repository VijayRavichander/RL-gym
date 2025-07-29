
from typing import List
from tqdm.asyncio import tqdm_asyncio
import asyncio
from utils import SYSTEM_PROMPT
from agent import agent_loop, reward_agent_response
from project_types import EvaluationRollOut, Input
import time
from datasets import load_dataset

async def run(batch: List[dict], model_name: str = "Qwen/Qwen3-32B", reward_model_name: str = "gpt-4.1-nano"):

    input = Input(
        question=batch["question"],
        inbox_address=batch["from_address"],
    )

    agent_answer = await agent_loop(input, model_name = model_name)

    rollout = EvaluationRollOut(
        question=batch["question"],
        agent_answer=agent_answer,
        golden_answer=batch["golden_answer"],
    )

    reward = await reward_agent_response(rollout, reward_model_name = reward_model_name)

    return reward

if __name__ == "__main__":

    async def main():
        
        for model in ["gpt-4.1-nano", "deepseek-ai/DeepSeek-V3-0324"]:
                start = time.time()

                dataset = load_dataset("vijay-ravichander/art-e-sample", split = "test")

                dataset = dataset.select(range(10))

                rewards = await tqdm_asyncio.gather(*[run(batch, model_name = model) for batch in dataset], 
                                                desc="Processing batches",
                                                total=len(dataset))
                
                print(f"Model: {model}")
                print(f"Accuracy: {sum(rewards) / len(rewards)}")
                print(f"Rewards: {rewards}")
                print(f"Time Taken: {time.time() - start} seconds")

    asyncio.run(main())