# Agent State Management Example

This project demonstrates how to manage agent state using the Semantic Kernel framework and Azure OpenAI services. The example showcases how to set up an AI-powered conversational agent, simulate interactions, and handle state persistence using a custom `StateHandler` class.

## Features

- **Azure OpenAI Integration**: Utilizes Azure OpenAI's Chat Completions API for conversational AI.
- **State Management**: Manages agent state using a JSON file, allowing for persistent and dynamic state updates.
- **Semantic Kernel Agent Framework**: Leverages Semantic Kernel's `ChatCompletionAgent` and `ChatHistoryAgentThread` for conversational interactions.
- **Asynchronous Programming**: Implements asynchronous functions for efficient API calls and state handling.

## Key Components

1. **Environment Configuration**:
   - Environment variables are loaded using `dotenv` to configure Azure OpenAI credentials and endpoints.
   - Required variables:
     - `AZURE_OPENAI_ENDPOINT`
     - `AZURE_OPENAI_API_VERSION`
     - `AZURE_OPENAI_DEPLOYMENT_NAME`
     - `AZURE_OPENAI_API_KEY`

2. **StateHandler**:
   - A custom class (`StateHandler`) is used to load, update, and persist state in a JSON file.

3. **Agent Setup**:
   - The `ChatCompletionAgent` is initialized with Azure OpenAI's `ChatCompletionsClient`.
   - The agent is configured with instructions to process key-value pairs.

4. **Simulated Interaction**:
   - The `simulate_interaction` function demonstrates:
     - Loading the initial state from a JSON file.
     - Sending the state to the agent and receiving responses.
     - Updating the state dynamically and querying specific values.

5. **Cleanup**:
   - After the interaction, the agent's thread is cleared to ensure proper resource management.

## How to Run

1. **Set Up Environment**:
   - Create a `.env` file with the required Azure OpenAI credentials and configuration.

2. **Install Dependencies**:
   - Install the required Python packages:
     ```bash
     pip install -r requirements.txt
     ```

3. **Run the Script**:
   - Execute the script using Python:
     ```bash
     python agent_state_management.py
     ```

4. **State File**:
   - Ensure a `state.json` file exists in the project directory for state persistence.

## Example Use Case

This example can be adapted for scenarios where conversational agents need to maintain and update state dynamically, such as customer support bots, virtual assistants, or interactive simulations.