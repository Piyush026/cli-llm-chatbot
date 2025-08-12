import asyncio
import click
from dotenv import load_dotenv
from llm_client import get_response
from colorama import Fore, Style, init
init(autoreset=True)
load_dotenv()


PERSONAS = {
    "friendly": "You are a friendly assistant who uses casual language and emojis.",
    "formal": "You are a professional assistant. Always be concise and polite.",
    "sarcastic": "You are a sarcastic assistant. Respond with dry humor.",
    "mentor": "You are a wise mentor. Give advice and explain reasoning."
}

@click.command()
@click.option("--role", "-r", default="formal",
              type=click.Choice(PERSONAS.keys()),
              help="Choose chatbot persona")

@click.option(
    "--memory", "-m",
    default=5,
    help="Number of past messages to keep in memory"
)
def cli(role, memory):
    """Start the async CLI chatbot."""
    asyncio.run(chat_loop(role,memory))

async def chat_loop(role,memory):
    print(f"Starting CLI chatbot with role: {role}\nType 'exit' to quit.\n")
    # basic initial system message (persona)
    context = [{"role": "system", "content": f"You are a {role} assistant. Keep answers short and helpful."}]
    while True:
        user_input = input(Fore.BLUE + "You: " + Style.RESET_ALL).strip()
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("Bye!")
            break

        context.append({"role": "user", "content": user_input})
        # last N in memory
        context = [context[0]] + context[-(memory * 2):]
        resp = await get_response(PERSONAS[role],context)
        # get_response returns a dict with key "content" (or a string fallback)
        assistant_msg = resp.get("content") if isinstance(resp, dict) else str(resp)
        print(Fore.CYAN + "Bot: " + Style.RESET_ALL + Fore.LIGHTRED_EX + assistant_msg + "\n")
        context.append({"role": "assistant", "content": assistant_msg})

if __name__ == "__main__":
    cli()
