
from typing import List
from tqdm.asyncio import tqdm_asyncio
import asyncio
from utils import SYSTEM_PROMPT
from agent import agent_loop, reward_agent_response
from project_types import RollOut
import time

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



async def run(batch: List[dict]):
    inbox = "louise.kitchen@enron.com"

    agent_answer = await agent_loop(inbox, batch["question"])

    rollout = RollOut(
        question=batch["question"],
        agent_answer=agent_answer,
        golden_answer=batch["golden_answer"],
    )

    reward = await reward_agent_response(rollout)

    return reward

if __name__ == "__main__":

    async def main():
        
        start = time.time()

        batchs = [
            {
                "question": data["question"],
                "golden_answer": data["answer"],
                "email_date": data["email_date"],
            }
            for data in DATA_JSONL
        ]

        # Use tqdm_asyncio for better async support
        rewards = await tqdm_asyncio.gather(*[run(batch) for batch in batchs], 
                                        desc="Processing batches",
                                        total=len(batchs))
        
        print(f"Accuracy: {sum(rewards) / len(rewards)}")
        print(f"Rewards: {rewards}")

        print(f"Time Taken: {time.time() - start} seconds")

    asyncio.run(main())