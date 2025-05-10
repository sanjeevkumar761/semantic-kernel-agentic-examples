# Imports
import asyncio
import os
import json
from typing import Any
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.ai.inference.aio import ChatCompletionsClient
from semantic_kernel.connectors.ai.azure_ai_inference import AzureAIInferenceChatCompletion
from semantic_kernel.agents import ChatCompletionAgent, ChatHistoryAgentThread

# Load environment variables
load_dotenv()

# Azure OpenAI configuration
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")
utility_deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
aoai_api_key = os.getenv("AZURE_OPENAI_API_KEY")


class MessageHandler:
    """Handles message lifecycle: loading, updating, and saving."""

    def __init__(self, file_path: str):
        self.file_path = file_path

    def load_message(self) -> dict:
        """Load a message from the JSON file."""
        with open(self.file_path, "r") as file:
            return json.load(file)

    def save_message(self, message: dict) -> None:
        """Save a message to the JSON file."""
        with open(self.file_path, "w") as file:
            json.dump(message, file, indent=4)
        print(f"File '{self.file_path}' updated successfully.")

    def update_message(self, key: str, increment: int) -> dict:
        """Update the value of a key in the message."""
        message = self.load_message()
        message[key] = message.get(key, 0) + increment
        self.save_message(message)
        return message


async def simulate_interaction(agent: ChatCompletionAgent, thread: ChatHistoryAgentThread, handler: MessageHandler):
    """Simulate an interaction with the agent."""

    # Load the initial message
    state = handler.load_message()

    # Send the initial message to the agent
    print(f"Setting state for agent: {json.dumps(state)}")
    response = await agent.get_response(messages=json.dumps(state), thread=thread)
    thread = response.thread
    print(f"# {response.name}: {response}")

    # Update the message
    updated_state = handler.update_message(key="value", increment=1)

    # Send the updated message to the agent
    response = await agent.get_response(messages=json.dumps(updated_state), thread=thread)
    thread = response.thread
    print(f"# {response.name}: {response}")

    # Query the updated value of "foo"
    response = await agent.get_response(messages="What is the value of foo?", thread=thread)
    thread = response.thread
    print(f"# {response.name}: {response}")

    return thread


async def main():
    """Main function to set up the agent and simulate a conversation."""
    file_path = "./state.json"
    handler = MessageHandler(file_path)

    # Create the agent
    agent = ChatCompletionAgent(
        service=AzureAIInferenceChatCompletion(
            ai_model_id="utility",
            service_id="utility",
            client=ChatCompletionsClient(
                endpoint=f"{str(endpoint).strip('/')}/openai/deployments/{utility_deployment_name}",
                credential=AzureKeyCredential(aoai_api_key),
                api_version=api_version,
            )
        ),
        name="Assistant",
        instructions="Just repeat the given key value pair.",
    )

    # Simulate the conversation
    thread = None
    thread = await simulate_interaction(agent, thread, handler)

    # Cleanup: Clear the thread
    if thread:
        await thread.delete()


if __name__ == "__main__":
    asyncio.run(main())