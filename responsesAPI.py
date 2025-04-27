import asyncio
from openai import AsyncOpenAI
from openai.types.responses import ResponseTextDeltaEvent
from agents import Agent, OpenAIResponsesModel, Runner, set_tracing_disabled,RunConfig,WebSearchTool
from dotenv import load_dotenv,find_dotenv
import os
import chainlit as cl

_:bool=load_dotenv(find_dotenv())

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY is None:
    raise ValueError("OPENAI_API_KEY not found in environment variables.")

BASE_URL = "https://api.openai.com/v1"
MODEL = "gpt-4.1-mini"

client = AsyncOpenAI(
    api_key=OPENAI_API_KEY,
    base_url=BASE_URL
)

model=OpenAIResponsesModel(
    model=MODEL,
    openai_client=client)

config=RunConfig(
    model_provider=client,
    model=model
)

teacher = Agent(
        name="Teaching Assistant",
        instructions="You are a teaching assistant. You are given a question and you need to answer it in a way that is easy to understand.",
        tools=[
            WebSearchTool(),
        ]
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
        run_config=config
    )
    async for e in result.stream_events():
        if e.type=="raw_response_event" and isinstance(e.data, ResponseTextDeltaEvent):
            token=e.data.delta
            await msg.stream_token(token)
    messages.append({"role": "assistant", "content": result.final_output})
    cl.user_session.set("messages", messages)
