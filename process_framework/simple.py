import asyncio
from typing import ClassVar

from pydantic import BaseModel, Field

from semantic_kernel import Kernel
from semantic_kernel.functions import kernel_function
from semantic_kernel.processes import ProcessBuilder
from semantic_kernel.processes.kernel_process import KernelProcessStep, KernelProcessStepContext, KernelProcessStepState
from semantic_kernel.processes.local_runtime.local_kernel_process import start
from semantic_kernel.processes.kernel_process import (
    KernelProcess,
    KernelProcessEvent,
    KernelProcessEventVisibility,
    KernelProcessStep,
    KernelProcessStepContext,
    KernelProcessStepState,
)

# Step 1: Gather user input
class GatherInputStepState(BaseModel):
    """State for the GatherInputStep."""
    user_input: str | None = None


class GatherInputStep(KernelProcessStep[GatherInputStepState]):
    state: GatherInputStepState = Field(default_factory=GatherInputStepState)

    @kernel_function
    async def gather_input(self, context: KernelProcessStepContext) -> None:
        print(f"{GatherInputStep.__name__}\n\t Gathering user input...")
        # Simulate gathering input
        self.state.user_input = "Hello, Semantic Kernel!"
        print(f"User input gathered: {self.state.user_input}")
        await context.emit_event(process_event="input_gathered", data=self.state.user_input)


# Step 2: Process the input
class ProcessInputStepState(BaseModel):
    """State for the ProcessInputStep."""
    processed_output: str | None = None


class ProcessInputStep(KernelProcessStep[ProcessInputStepState]):
    state: ProcessInputStepState = Field(default_factory=ProcessInputStepState)

    @kernel_function
    async def process_input(self, context: KernelProcessStepContext, user_input: str) -> None:
        print(f"{ProcessInputStep.__name__}\n\t Processing user input: {user_input}")
        # Simulate processing input
        self.state.processed_output = user_input.upper()
        print(f"Processed output: {self.state.processed_output}")
        await context.emit_event(process_event="input_processed", data=self.state.processed_output)


# Create the process builder
process_builder = ProcessBuilder(name="SimpleTwoStepProcess")

# Add the steps
input_gathering_step = process_builder.add_step(GatherInputStep)
input_processing_step = process_builder.add_step(ProcessInputStep)

# Orchestrate the events
process_builder.on_input_event("Start").send_event_to(target=input_gathering_step)

input_gathering_step.on_event("input_gathered").send_event_to(
    target=input_processing_step, function_name="process_input", parameter_name="user_input"
)


async def main():
    # Configure the kernel
    kernel = Kernel()

    # Build the process
    kernel_process = process_builder.build()

    # Start the process
    async with await start(
        process=kernel_process,
        kernel=kernel,
        initial_event=KernelProcessEvent(id="Start", data=None),
    ) as process_context:
        _ = await process_context.get_state()


# Run the async main function
if __name__ == "__main__":
    asyncio.run(main())