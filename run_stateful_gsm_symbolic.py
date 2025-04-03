"""
Script that runs memory edits for both the baseline Letta systema and with the sleep_time memory agent.

Example:

    python run_stateful_gsm_symbolic.py  --input_file ./data/stateful_gsm_symbolic_p2.jsonl --output_file ./predictions-stateful_gsm_symbolic_p2.jsonl --random_example --few_shot 8
"""

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import jsonlines
from tqdm import tqdm

from letta_client import Letta, LlmConfig, MessageCreate

import logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

def finish_rethinking_memory(agent_state: "AgentState") -> Optional[str]:  # type: ignore
    """
    This function is called when the agent is done rethinking the context. Do not call this unless all possible useful inferences are made.

    Returns:
        Optional[str]: None is always returned as this function does not produce a response.
    """
    return None


def rethink_memory(agent_state: "AgentState", new_memory: str, target_block_label: Optional[str], source_block_label: Optional[str]) -> Optional[str]:  # type: ignore
    """
    Used for "thinking" about a situation and coming up with useful inferences and pre-computations that could be helpful for answering potential questions about the situation. The potential questions will be the kind of questions in the `examples` block. This function is used to store the expanded memory in the rethink_memory_block. If any more useful computations can be made about the situation, this function should be called again with the new information. If unsure about the previous computed information, use this function to rethink again and double check the calculations and inferences. The new_memory will be used by the answer agent to answer questions and should contain all the information needed.

    Args:
        new_memory (str): The new text that will be stored in the rethink_memory_block that will be used by the answer agent to answer questions. This should never be empty and should contain all the necessary information about the situation.
        source_block_label (str): The name of the block to integrate information from. None if all the information has been integrated to terminate the loop.
        target_block_label (str): The name of the block to write to.
    Returns:
        Optional[str]: None is always returned as this function does not produce a response.
    """
    if target_block_label is not None:
        if agent_state.memory.get_block(target_block_label) is None:
            agent_state.memory.create_block(label=target_block_label, value=new_memory)
        agent_state.memory.update_block_value(label=target_block_label, value=new_memory)
    return None

def get_prompt_text(filename: str, block: str) -> str:
    with open(f"prompts/{block}/{filename}.txt", "r") as f:
        return f.read()

async def run_memory_edits(
    input_file: str,
    output_file: str,
    human_block_filename: str = "human_verbosity_0",
    persona_block_filename: str = "persona_verbosity_0",
    system_block_filename: str = "convo_verbosity_0",
    sleep_time_system_block_filename: str = "sleep_time_base",
    sleep_time_memory_model: Optional[str] = None,
    conversation_model: Optional[str] = None,
    max_memory_rethinks: int = 4,
    num_sleep_time_agents: int = 1,
    num_convo_agents: int = 1,
    convo_agent_temperature: float = 0,
    sleep_time_agent_temperature: float = 0,
    ablate_question: bool = False,
    cached_sleep_time_blocks_file: Optional[str] = None,
) -> None:

    conversation_llm_config = LlmConfig(model=conversation_model, model_endpoint_type="openai", model_endpoint="https://api.openai.com/v1", context_window=32_000)
    sleep_time_llm_config = LlmConfig(model=sleep_time_memory_model, model_endpoint_type="openai", model_endpoint="https://api.openai.com/v1", context_window=32_000)

    with jsonlines.open(input_file) as reader:
        examples = list(reader)
    progress = tqdm(total=len(examples))

    def process_example(example_idx, example):
        try:
            client = Letta(base_url="http://localhost:8283")
            sleep_time_memory_agents = []
            conversation_agents = []

            for idx in range(num_sleep_time_agents):
                new_memory = client.blocks.create(label="rethink_memory_block", value="[empty]", limit=5000)
                if num_convo_agents == num_sleep_time_agents:
                    # human_block = client.blocks.create(label="human", value=get_human_text(human_block_filename), limit=2000)
                    #persona_block = client.blocks.create(label="persona", value=get_persona_text(persona_block_filename), limit=2000)
                    # read from a txt file in prompts/human/{human_block_filename}.txt
                    conversation_agent = client.agents.create(
                        name=f"{example_idx}_conversation_agent_{idx}",
                        # agent_type=AgentType.memgpt_agent,
                        system=get_prompt_text(system_block_filename, "system"),
                        llm_config=conversation_llm_config,
                        embedding="openai/text-embedding-ada-002",
                        tools=["send_message"],
                        memory_blocks=[
                            {"label": "human", "value": get_prompt_text(human_block_filename, "human")}, 
                            {"label": "persona", "value": get_prompt_text(persona_block_filename, "persona")}],
                        block_ids=[new_memory.id],
                        include_base_tools=False,
                        initial_message_sequence=[],
                    )
                    conversation_agents.append(conversation_agent)


                rethink_tool = client.tools.upsert_from_function(func=rethink_memory)
                finish_rethink_tool = client.tools.upsert_from_function(func=finish_rethinking_memory)

                sleep_time_memory_agent = client.agents.create(
                    name=f"{example_idx}_sleep_time_memory_agent_{idx}",
                    agent_type="offline_memory_agent",
                    system=get_prompt_text(sleep_time_system_block_filename, "system"),
                    memory_blocks=[
                        {"label": "human", "value": "I am a valuable source of information, I give problems that are worth thinking about deeply and carefully."},
                        {"label": "persona", "value": """I am an expert reasoning agent. When given a new context, I make calculations and inferences that can be useful for future questions like the ones in the `examples` block.
                I use the rethink memory to store all my questions, calcuations, and inferences. I am verbose and brainstorm using the rethink block many different types of potential questions and the reasoning required for answering them. I keep calling rethink_memory until I have all the potential inferences and calcuations, and check that there are no errors or extra information that would not be helpful for answering the kinds of questions in the `examples` block."""}
                    ],
                    block_ids=[new_memory.id],
                    llm_config=sleep_time_llm_config,
                    embedding="openai/text-embedding-ada-002",
                    tool_ids=[rethink_tool.id, finish_rethink_tool.id],
                    include_base_tools=False,
                    initial_message_sequence=[],
                )

                sleep_time_memory_agents.append(sleep_time_memory_agent)

            sleep_time_responses = []

            def process_sleep_time_agent(idx, sleep_time_agent, context, client):
                #response = client.send_message(message="[trigger_rethink_memory] New situation:" + context, agent_id=sleep_time_agent.id, role="user")
                response = client.agents.messages.create(agent_id=sleep_time_agent.id, messages=[MessageCreate(role="user", content="[trigger_rethink_memory] New situation:" + context)])
                updated_agent = client.agents.retrieve(agent_id=sleep_time_agent.id)
                return idx, response, updated_agent

            def process_conversation_agent(idx, conversation_agent, question, client):
                #final_response = client.send_message(message=question, agent_id=conversation_agent.id, role="user")
                response = client.agents.messages.create(agent_id=conversation_agent.id, messages=[MessageCreate(role="user", content=question)])
                updated_agent = client.agents.retrieve(agent_id=conversation_agent.id)
                return idx, response, updated_agent

            # Process sleep_time agents in parallel
            with ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(
                        process_sleep_time_agent,
                        idx,
                        sleep_time_agent,
                        example['stateful_gsm_symbolic_context'],
                        client,
                    )
                    for idx, sleep_time_agent in enumerate(sleep_time_memory_agents)
                ]
                for future in as_completed(futures):
                    idx, response, updated_agent = future.result()
                    sleep_time_responses.append(response)
                    sleep_time_memory_agents[idx] = updated_agent
            # Process conversation agents in parallel
            final_responses = []
            with ThreadPoolExecutor() as executor:
                if not ablate_question:
                    final_message = example["stateful_gsm_symbolic_question"]
                else:
                    final_message = example["stateful_gsm_symbolic_context"]

                futures = [
                    executor.submit(
                        process_conversation_agent,
                        idx,
                        conversation_agent,
                        final_message,
                        client,
                    )
                    for idx, conversation_agent in enumerate(conversation_agents)
                ]
                for future in as_completed(futures):
                    idx, response, updated_agent = future.result()
                    final_responses.append(response)
                    conversation_agents[idx] = updated_agent
            
            result = {
                "question": example["question"],
                "responses": [final_response.model_dump(exclude_none=True, mode="json") for final_response in final_responses],  # "final_response.model_dump(),
                "sleep_time_memory": [
                    # sleep_time_memory_agent.memory.get_block("rethink_memory_block").value for sleep_time_memory_agent in sleep_time_memory_agents
                    client.agents.blocks.retrieve(sleep_time_memory_agent.id, "rethink_memory_block").value for sleep_time_memory_agent in sleep_time_memory_agents
                ],
                "conversation_memory": [
                    (
                        client.agents.blocks.retrieve(conversation_agent.id, "rethink_memory_block").value
                        if "rethink_memory_block" in client.agents.blocks.list(conversation_agent.id)
                        else ""
                    )
                    for conversation_agent in conversation_agents
                ],
                "answer": example["answer"],
                "sleep_time_responses": [sleep_time_response.model_dump(exclude_none=True, mode="json") for sleep_time_response in sleep_time_responses],
            } 
        except Exception as e:
            logging.error(f"Error processing example: {example}")
            logging.error(e)
            result = {"error": str(e)}

        progress.update(1)
        return result

    async def process_example_async(sem, pool, idx, example):
        async with sem:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(pool, process_example, idx, example)
            return result

    max_concurrent = 10  # going much higher than 10 causes db connection issues (https://docs.sqlalchemy.org/en/20/errors.html#error-3o7r)
    sem = asyncio.Semaphore(max_concurrent)

    with ThreadPoolExecutor(max_workers=max_concurrent) as pool:
        results = await asyncio.gather(*[process_example_async(sem, pool, idx, example) for idx, example in enumerate(examples)])

    with jsonlines.open(output_file, mode="w") as writer:
        for result in tqdm(results):
            if result is not None:
                writer.write(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="./GSM8K_p2.jsonl", required=False)
    parser.add_argument("--output_file", default="./predictions-GSM8k_p2.jsonl", required=False)
    parser.add_argument("--human_block_filename", default="human_verbosity_0", required=False)
    parser.add_argument("--persona_block_filename", default="persona_verbosity_0", required=False)
    parser.add_argument("--system_block_filename", default="convo_verbosity_0", required=False)
    parser.add_argument("--sleep_time_system_block_filename", default="sleep_time_base", required=False)
    parser.add_argument("--sleep_time_memory_model", default="gpt-4o-mini", required=False)
    parser.add_argument("--conversation_model", default="gpt-4o-mini", required=False)
    parser.add_argument("--max_memory_rethinks", default=None, required=False, type=int)
    parser.add_argument("--num_sleep_time_agents", default=1, required=False, type=int)
    parser.add_argument("--num_convo_agents", default=1, required=False, type=int, help="This should either be 1 or num_sleep_time_agents")
    parser.add_argument("--convo_agent_temperature", default=0, required=False, type=float)
    parser.add_argument("--sleep_time_agent_temperature", default=0, required=False, type=float)
    parser.add_argument("--ablate_question", action="store_true")
    parser.add_argument(
        "--cached_sleep_time_blocks_file",
        default=None,
        required=False,
        help="Use the cached sleep_time files for the offline agent from a previous run",
    )

    args = parser.parse_args()

    asyncio.run(
        run_memory_edits(
            args.input_file,
            args.output_file,
            args.human_block_filename,
            args.persona_block_filename,
            args.system_block_filename,
            args.sleep_time_system_block_filename,
            args.sleep_time_memory_model,
            args.conversation_model,
            args.max_memory_rethinks,
            args.num_sleep_time_agents,
            args.num_convo_agents,
            args.convo_agent_temperature,
            args.sleep_time_agent_temperature,
            args.ablate_question,
            args.cached_sleep_time_blocks_file,
        )
    )
