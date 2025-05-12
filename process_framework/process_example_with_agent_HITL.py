import asyncio
import os
from dotenv import load_dotenv
from typing import ClassVar

from pydantic import BaseModel, Field

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.chat_completion_client_base import ChatCompletionClientBase
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.contents import ChatHistory
from semantic_kernel.functions import kernel_function
from semantic_kernel.processes import ProcessBuilder
from semantic_kernel.processes.kernel_process import KernelProcessStep, KernelProcessStepContext, KernelProcessStepState
from semantic_kernel.processes.kernel_process import (
    KernelProcess,
    KernelProcessEvent,
    KernelProcessEventVisibility,
    KernelProcessStep,
    KernelProcessStepContext,
    KernelProcessStepState,
)

from semantic_kernel.processes.local_runtime.local_kernel_process import start
from semantic_kernel.connectors.ai.azure_ai_inference import AzureAIInferenceChatCompletion
from azure.core.credentials import AzureKeyCredential
from azure.ai.inference.aio import ChatCompletionsClient
from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.open_ai_prompt_execution_settings import (
    OpenAIChatPromptExecutionSettings,
    OpenAITextPromptExecutionSettings,
    PromptExecutionSettings
)

load_dotenv()

# Azure OpenAI configuration
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")
utility_deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
aoai_api_key = os.getenv("AZURE_OPENAI_API_KEY")

# A process step to gather information about a product
class GatherProductInfoStep(KernelProcessStep):
    @kernel_function
    def gather_product_information(self, product_name: str) -> str:
        print(f"{GatherProductInfoStep.__name__}\n\t Gathering product information for Product Name: {product_name}")

        return """
Product Description:

GlowBrew is a revolutionary AI driven coffee machine with industry leading number of LEDs and 
programmable light shows. The machine is also capable of brewing coffee and has a built in grinder.

Product Features:
1. **Luminous Brew Technology**: Customize your morning ambiance with programmable LED lights that sync 
    with your brewing process.
2. **AI Taste Assistant**: Learns your taste preferences over time and suggests new brew combinations 
    to explore.
3. **Gourmet Aroma Diffusion**: Built-in aroma diffusers enhance your coffee's scent profile, energizing 
    your senses before the first sip.

Troubleshooting:
- **Issue**: LED Lights Malfunctioning
    - **Solution**: Reset the lighting settings via the app. Ensure the LED connections inside the 
        GlowBrew are secure. Perform a factory reset if necessary.
        """


# A sample step state model for the GenerateDocumentationStep
class GeneratedDocumentationState(BaseModel):
    """State for the GenerateDocumentationStep."""

    chat_history: ChatHistory | None = None


# Updated process step to generate and edit documentation for a product
class GenerateDocumentationStep(KernelProcessStep[GeneratedDocumentationState]):
    state: GeneratedDocumentationState = Field(default_factory=GeneratedDocumentationState)

    system_prompt: ClassVar[str] = """
Your job is to write high quality and engaging customer facing documentation for a new product from Contoso. You will 
be provided with information about the product in the form of internal documentation, specs, and troubleshooting guides 
and you must use this information and nothing else to generate the documentation. If suggestions are provided on the 
documentation you create, take the suggestions into account and rewrite the documentation. Make sure the product 
sounds amazing.
"""

    async def activate(self, state: KernelProcessStepState[GeneratedDocumentationState]):
        self.state = state.state
        if self.state.chat_history is None:
            self.state.chat_history = ChatHistory(system_message=self.system_prompt)
        self.state.chat_history

    @kernel_function
    async def generate_documentation(
        self, context: KernelProcessStepContext, product_info: str, kernel: Kernel
    ) -> None:
        print(f"{GenerateDocumentationStep.__name__}\n\t Generating documentation for provided product_info...")

        self.state.chat_history.add_user_message(f"Product Information:\n{product_info}")

        chat_service, settings = kernel.select_ai_service(type=ChatCompletionClientBase)
        assert isinstance(chat_service, ChatCompletionClientBase)  # nosec

        response = await chat_service.get_chat_message_content(chat_history=self.state.chat_history, settings=settings)

        await context.emit_event(process_event="documentation_generated", data=str(response))

    @kernel_function
    async def apply_suggestions(self, suggestions: str, context: KernelProcessStepContext, kernel: Kernel) -> None:
        print(f"{GenerateDocumentationStep.__name__}\n\t Rewriting documentation with provided suggestions...")

        self.state.chat_history.add_user_message(
            f"Rewrite the documentation with the following suggestions:\n\n{suggestions}"
        )

        chat_service, settings = kernel.select_ai_service(type=ChatCompletionClientBase)
        assert isinstance(chat_service, ChatCompletionClientBase)  # nosec

        generated_documentation_response = await chat_service.get_chat_message_content(
            chat_history=self.state.chat_history, settings=settings
        )

        await context.emit_event(process_event="documentation_generated", data=str(generated_documentation_response))

# A process step to publish documentation
class PublishDocumentationStep(KernelProcessStep):
    @kernel_function
    async def publish_documentation(self, docs: str) -> None:
        print(f"{PublishDocumentationStep.__name__}\n\t Publishing product documentation:\n\n{docs}")

# A sample response model for the ProofreadingStep structured output
class ProofreadingResponse(BaseModel):
    """A class to represent the response from the proofreading step."""

    meets_expectations: bool = Field(description="Specifies if the proposed docs meets the standards for publishing.")
    explanation: str = Field(description="An explanation of why the documentation does or does not meet expectations.")
    suggestions: list[str] = Field(description="List of suggestions, empty if there are no suggestions for improvement.")

# A process step to proofread documentation
class ProofreadStep(KernelProcessStep):
    @kernel_function
    async def proofread_documentation(self, docs: str, context: KernelProcessStepContext, kernel: Kernel) -> None:
        print(f"{ProofreadStep.__name__}\n\t Proofreading product documentation...")

        system_prompt = """
        Your job is to proofread customer facing documentation for a new product from Contoso. You will be provided with 
        proposed documentation for a product and you must do the following things:

        1. Determine if the documentation passes the following criteria:
            1. Documentation must use a professional tone.
            1. Documentation should be free of spelling or grammar mistakes.
            1. Documentation should be free of any offensive or inappropriate language.
            1. Documentation should be technically accurate.
        2. If the documentation does not pass 1, you must write detailed feedback of the changes that are needed to 
            improve the documentation. 
        """

        chat_history = ChatHistory(system_message=system_prompt)
        chat_history.add_user_message(docs)

        # Use structured output to ensure the response format is easily parsable
        chat_service, settings = kernel.select_ai_service(type=ChatCompletionClientBase)
        assert isinstance(chat_service, ChatCompletionClientBase)  # nosec
        assert isinstance(settings, PromptExecutionSettings)  # nosec

        settings.response_format = ProofreadingResponse

        response = await chat_service.get_chat_message_content(chat_history=chat_history, settings=settings)

        formatted_response: ProofreadingResponse = ProofreadingResponse.model_validate_json(response.content)

        suggestions_text = "\n\t\t".join(formatted_response.suggestions)
        print(
            f"\n\tGrade: {'Pass' if formatted_response.meets_expectations else 'Fail'}\n\t"
            f"Explanation: {formatted_response.explanation}\n\t"
            f"Suggestions: {suggestions_text}"
        )

        if formatted_response.meets_expectations:
            await context.emit_event(process_event="documentation_approved", data=docs)
        else:
            await context.emit_event(
                process_event="documentation_rejected",
                data={"explanation": formatted_response.explanation, "suggestions": formatted_response.suggestions},
            )
# Create the process builder
# Create the process builder
process_builder = ProcessBuilder(name="DocumentationGeneration")

# Add the steps
info_gathering_step = process_builder.add_step(GatherProductInfoStep)
docs_generation_step = process_builder.add_step(GenerateDocumentationStep)
docs_proofread_step = process_builder.add_step(ProofreadStep)  # Add new step here
docs_publish_step = process_builder.add_step(PublishDocumentationStep)

# Orchestrate the events
process_builder.on_input_event("Start").send_event_to(target=info_gathering_step)

info_gathering_step.on_function_result(function_name="gather_product_information").send_event_to(
    target=docs_generation_step, function_name="generate_documentation", parameter_name="product_info"
)

docs_generation_step.on_event("documentation_generated").send_event_to(
    target=docs_proofread_step, parameter_name="docs"
)

docs_proofread_step.on_event("documentation_rejected").send_event_to(
    target=docs_generation_step,
    function_name="apply_suggestions",
    parameter_name="suggestions",
)

docs_proofread_step.on_event("documentation_approved").send_event_to(target=docs_publish_step)

async def main():
    # Configure the kernel with an AI Service and connection details, if necessary
    kernel = Kernel()
    kernel.add_service(
        AzureAIInferenceChatCompletion(
            ai_model_id="utility",
            service_id="utility",
            client=ChatCompletionsClient(
                endpoint=f"{str(endpoint).strip('/')}/openai/deployments/{utility_deployment_name}",
                credential=AzureKeyCredential(aoai_api_key),
                api_version=api_version,
            )
        )
    )

    # Build the process
    kernel_process = process_builder.build()

    # Start the process
    async with await start(
        process=kernel_process,
        kernel=kernel,
        initial_event=KernelProcessEvent(id="Start", data="Contoso GlowBrew"),
    ) as process_context:
        _ = await process_context.get_state()

# Run the async main function
if __name__ == "__main__":
    asyncio.run(main())