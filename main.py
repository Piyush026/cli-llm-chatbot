import asyncio
import click
from dotenv import load_dotenv
from llm_client import get_response

load_dotenv()

@click.command()
@click.option("--role", "-r", default="system", type=click.Choice(["friendly","formal"]), help="Role persona")
def cli(role):
    """Start the async CLI chatbot."""
    asyncio.run(chat_loop(role))

async def chat_loop(role):
    print(f"Starting CLI chatbot with role: {role}\nType 'exit' to quit.\n")
    # basic initial system message (persona)
    context = [{"role": "system", "content": f"You are a {role} assistant. Keep answers short and helpful."}]
    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("Bye!")
            break

        context.append({"role": "user", "content": user_input})
        # call the LLM client (async)
        resp = await get_response(user_input,role)
        # get_response returns a dict with key "content" (or a string fallback)
        assistant_msg = resp.get("content") if isinstance(resp, dict) else str(resp)
        print(f"\nBot: {assistant_msg}\n")
        context.append({"role": "assistant", "content": assistant_msg})

if __name__ == "__main__":
    cli()
