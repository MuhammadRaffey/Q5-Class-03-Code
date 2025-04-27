import asyncio
from openai import AsyncOpenAI
from agents import Agent, OpenAIChatCompletionsModel, Runner, set_tracing_disabled
from dotenv import load_dotenv,find_dotenv
import os
from openai.types.responses import ResponseTextDeltaEvent
import chainlit as cl

_:bool=load_dotenv(find_dotenv())

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

teacher = Agent(
        name="Teaching Assistant",
        instructions="You are a teaching assistant. You are given a question and you need to answer it in a way that is easy to understand.",
        model=OpenAIChatCompletionsModel(model=MODEL, openai_client=client),
    )

@cl.on_chat_start
async def start_chat():
    cl.user_session.set("messages", [])
    await cl.Message(content="Welcome to the Teaching Assistant!").send()

@cl.on_message
async def handle_message(message: cl.Message):
    messages = cl.user_session.get("messages", [])
    messages.append({"role": "user", "content": message.content})
    msg=cl.Message(content="")

    result = Runner.run_streamed(
        teacher,
        messages,
    )
    async for e in result.stream_events():
        if e.type=="raw_response_event" and isinstance(e.data, ResponseTextDeltaEvent):
            token=e.data.delta
            await msg.stream_token(token)
    messages.append({"role": "assistant", "content": result.final_output})
    cl.user_session.set("messages", messages)
