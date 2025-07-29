import verifiers as vf
from verifiers.rubrics.judge_rubric import JudgeRubric
from tools import read_email_tool, search_emails_tool
from datasets import load_dataset
from utils import SYSTEM_PROMPT
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

dataset = load_dataset("vijay-ravichander/art-e-sample", split = "test")
tools = [read_email_tool, search_emails_tool]

client = OpenAI(
    base_url=os.getenv("DEEPINFRA_API_LINK"), api_key=os.getenv("DEEPINFRA_API_KEY")
)

dataset = dataset.select(range(10))

dataset = dataset.map(lambda x: {"question": x["question"] + "\n The Inbox Address is: " + x["from_address"] + "\n", "answer": x["golden_answer"]})

# Judge Config
judge_model = "deepseek-ai/DeepSeek-V3-0324"
judge_model_parser = vf.XMLParser(['reward'], answer_field="reward")

if "gpt" in judge_model:
    judge_client = OpenAI(
        base_url=os.getenv("OPENAI_API_LINK"), api_key=os.getenv("OPENAI_API_KEY")
    )
else:
    judge_client = OpenAI(
        base_url=os.getenv("DEEPINFRA_API_LINK"), api_key=os.getenv("DEEPINFRA_API_KEY")
    )

judge_rubric = JudgeRubric(
    judge_client=judge_client, judge_model=judge_model, parser=judge_model_parser
)

# Test Rubric
def reward_func(completion, answer, **kwargs) -> float:
    """
    Check if the completion is sorted    
    """
    return 1.0 

parser = vf.XMLParser(['think', 'tool', 'answer'], answer_field="answer")

# Tool Config
vf_env = vf.ToolEnv(
    dataset=dataset, system_prompt=SYSTEM_PROMPT, tools=tools, max_turns=5, parser=parser, format_prompt=False
)

vf_env.rubric = vf.RubricGroup(rubrics=[judge_rubric])

results = vf_env.evaluate(client, model="deepseek-ai/DeepSeek-V3-0324", num_samples = -1, max_concurrent = 128)

print(results)