from langchain_openai import OpenAI
import asyncio

# Create the async function
async def generate_song():
    llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0, max_tokens=512)
    async for chunk in llm.astream("Write me a 1 verse song about sparkling water."):
        print(chunk, end="", flush=True)

# Run the async function
if __name__ == "__main__":
    asyncio.run(generate_song())

