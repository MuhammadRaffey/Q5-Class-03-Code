import asyncio
from openai import AsyncOpenAI
from agents import Agent, OpenAIChatCompletionsModel, Runner, set_tracing_disabled
from dotenv import load_dotenv,find_dotenv
import os

_:bool=load_dotenv(find_dotenv())
print("Loaded environment variables:", _)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if OPENROUTER_API_KEY is None:
    raise ValueError("OPENROUTER_API_KEY not found in environment variables.")

BASE_URL = "https://openrouter.ai/api/v1"
MODEL = "meta-llama/llama-4-maverick:free"

client = AsyncOpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url=BASE_URL
)

set_tracing_disabled(disabled=True)

async def main():
    agent = Agent(
        name="Teaching Assistant",
        instructions="You are a teaching assistant. You are given a question and you need to answer it in a way that is easy to understand.",
        model=OpenAIChatCompletionsModel(model=MODEL, openai_client=client),
    )
    msg=[]

    while True:
        user_input = input("You: ")
        msg.append({"role": "user", "content": user_input})
        if user_input.lower() == "exit":
            break
        result = await Runner.run(
            agent,
            msg,
        )
        msg.append({"role": "assistant", "content": result.final_output})
        print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())