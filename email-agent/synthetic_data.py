import sqlite3
import os
import json
from typing import List
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from openai import OpenAI

from project_types import Email
from utils import parse_from_response
from datasets import load_dataset, Dataset

import asyncio
from train_and_test_inbox import train_inboxes, test_inboxes
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio
from rich import print

load_dotenv()

QA_SYS_PROMPT = """ 
You are an assistant that creates realistic question–answer pairs a human might ask about their e-mails.
Every answer MUST be fully contained in the provided e-mails. Do NOT hallucinate.
* Enclose the JSON in <data> </data> tags.
* The questions should be based on the email body.
* The questions should resemble what a user might ask their email agent.
* Generate one JSON object for each email and not more than that. 
* Question's cannot be like what is the subject of the email or things like that which cannot be searched through an inbox.
* I want trivia kinda questions based on the email's body
* You will be given 5 emails, you need to generate 5 questions and answers in total.

Respond with a JSON object with the following structure for each email:
<data>
[
  {
    "question": "Your Question goes here",
    "answer": "Your Answer goes here",
    "realistic_score": "How realistic is this score between 0 to 1"
  }
]
</data>
"""


async def load_emails(inbox_address: str = "phillip.allen@enron.com", limit: int = 5) -> List[Email]:
    emails_for_qa: List[Email] = []

    base_query_ids = """ 
    SELECT DISTINCT e.id
    FROM emails e
    LEFT JOIN recipients r ON r.email_id = e.id
    WHERE LOWER(e.from_address) = ? OR LOWER(r.recipient_address) = ?
    ORDER BY e.date ASC
    """

    conn = sqlite3.connect("db/enron_emails.db")
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute(base_query_ids, (inbox_address, inbox_address))
    all_email_ids = [row[0] for row in cursor.fetchall()][:limit]

    if not all_email_ids:
        conn.close()
        return []

    placeholders = ",".join(["?"] * len(all_email_ids))
    email_data_query = f"""
    SELECT id, message_id, subject, from_address, date, body, file_name
    FROM emails
    WHERE id IN ({placeholders})
    ORDER BY date ASC
    """
    email_rows = cursor.execute(email_data_query, all_email_ids).fetchall()

    for row in email_rows:
        rec_cursor = conn.execute(
            "SELECT recipient_address, recipient_type FROM recipients WHERE email_id = ?",
            (row["id"],),
        )

        to_list, cc_list, bcc_list = [], [], []
        for rec in rec_cursor.fetchall():
            if rec["recipient_type"] == "to":
                to_list.append(rec["recipient_address"])
            elif rec["recipient_type"] == "cc":
                cc_list.append(rec["recipient_address"])
            elif rec["recipient_type"] == "bcc":
                bcc_list.append(rec["recipient_address"])

        email_obj = Email(
            message_id=row["message_id"],
            date=row["date"],
            subject=row["subject"],
            from_address=row["from_address"],
            to_addresses=to_list,
            cc_addresses=cc_list,
            bcc_addresses=bcc_list,
            body=row["body"],
            file_name=row["file_name"],
        )

        emails_for_qa.append(email_obj)

    conn.close()
    return emails_for_qa


async def generate_qa_from_llm(emails: List[Email]) -> List[dict]:
    DI_client = OpenAI(
        base_url=os.getenv("DEEPINFRA_API_LINK"),
        api_key=os.getenv("DEEPINFRA_API_KEY")
    )

    input_text = "\n\n".join(json.dumps(email.model_dump()) for email in emails)

    
    response = DI_client.chat.completions.create(
        model="deepseek-ai/DeepSeek-V3-0324",
        messages=[
            {"role": "system", "content": QA_SYS_PROMPT},
            {"role": "user", "content": input_text},
        ]
    )

    content = response.choices[0].message.content
    parsed = parse_from_response(content, "data")
    parsed_list = json.loads(parsed) if parsed else []

    try:
        return [
            {
                "question": data["question"],
                "golden_answer": data["answer"],
                "realistic_score": data["realistic_score"],
                "query_date": emails[idx].date,
                "inbox_address": emails[idx].from_address,
            }
            for idx, data in enumerate(parsed_list)
    ]

    except json.JSONDecodeError:
        print("Failed to parse JSON from model response.")
        return []


if __name__ == "__main__":

    async def main(limit: int = 5):
        all_train, all_test = [], []

        for split, inbox_list in [("train", train_inboxes), ("test", test_inboxes)]:

            for inbox_address in tqdm(inbox_list, desc=f"{split} inboxes"):

                emails = await load_emails(inbox_address, limit=limit)

                print(f"Retrieved {len(emails)} emails from the database for the inbox: {inbox_address}")

                qa_pairs = await generate_qa_from_llm(emails[:limit])

                print(f"Generated {len(qa_pairs)} pairs from {inbox_address} inbox address | split: {split}")
                print(f"Sample from Generated Data: {json.dumps(qa_pairs[0], indent=2)}")

                all_train.extend(qa_pairs) if split == "train" else all_test.extend(qa_pairs)

                # Write to appropriate split file
                output_path = f"output/qa_{split}.jsonl"
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                with open(output_path, "a", encoding="utf-8") as f:
                    for item in qa_pairs:
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")

        # Build combined dataset with split
        # dataset = Dataset.from_list(all_train + all_test)

        # Push full dataset to Hugging Face Hub
        # dataset.push_to_hub("art-e-sample", split=None)

        # Optional: push train and test separately
        Dataset.from_list(all_train).push_to_hub("art-e-sample", split="train")
        Dataset.from_list(all_test).push_to_hub("art-e-sample", split="test")

    asyncio.run(main())
