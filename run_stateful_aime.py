"""
Script that runs memory edits for both the baseline Letta systema and with the sleep_time memory agent.

Example:

    python run_stateful_gsm_symbolic.py  --input_file ./data/stateful_gsm_symbolic_p2.jsonl --output_file ./predictions-stateful_gsm_symbolic_p2.jsonl --random_example --few_shot 8
"""

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
import uuid
import base64
import time

import jsonlines
from tqdm import tqdm
import datasets

from letta_client import Letta, LlmConfig, MessageCreate

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
    output_file: str,
    human_block_filename: str = "human_verbosity_0",
    persona_block_filename: str = "persona_verbosity_0",
    system_block_filename: str = "convo_verbosity_0",
    sleep_time_system_block_filename: str = "sleep_time_base_aime",
    model: str = "o3-mini",
    num_sleep_time_agents: int = 1,
    num_convo_agents: int = 1,
) -> None:
    if "claude" in model:
        LLM_CONFIG = LlmConfig(
            model=model,
            model_endpoint_type="anthropic",
            model_endpoint="https://api.anthropic.com/v1",
            context_window=200_000,
            enable_reasoner=True,
            max_reasoning_tokens=20_000,
            max_tokens=30_000,
            put_inner_thoughts_in_kwargs=False,
        )
    elif model == 'o3-mini' or model == 'o1':
        LLM_CONFIG = LlmConfig(
            model=model,
            model_endpoint_type="openai",
            model_endpoint="https://api.openai.com/v1",
            context_window=200_000,
            temperature=1.0,
        )

    client = Letta()
    examples = list(datasets.load_dataset("letta-ai/stateful-aime-2024")['train'])
    progress = tqdm(total=len(examples))

    uuid_bytes = uuid.uuid4().bytes
    run_uuid = base64.urlsafe_b64encode(uuid_bytes).rstrip(b'=').decode('ascii')

    def process_example(example_idx, example):
        retry_id = 0
        while True:
            try:
                client = Letta(base_url="http://localhost:8283")
                sleep_time_memory_agents = []
                conversation_agents = []

                for idx in range(num_sleep_time_agents):
                    new_memory = client.blocks.create(label="rethink_memory_block", value="[empty]", limit=5000)
                    if num_convo_agents == num_sleep_time_agents:
                        human_block = client.blocks.create(label="human", value="", limit=2000)
                        persona_block = client.blocks.create(label="persona", value="", limit=2000)

                        conversation_agent = client.agents.create(
                            name=f"{example_idx}_conversation_agent_{idx}_{run_uuid}_{retry_id}",
                            system=get_prompt_text(system_block_filename, "system"),
                            llm_config=LLM_CONFIG,
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
                    time.sleep(1) # Postgres may generate duplicate message IDs when we create agents too fast

                    sleep_time_human_block = client.blocks.create(
                        label="human",
                        value="I am a valuable source of information, I give problems that are worth thinking about deeply and carefully.",
                        limit=2000,
                    )
                    sleep_time_persona_block = client.blocks.create(
                        label="persona",
                        value="""I am an expert reasoning agent. When given a new context, I make calculations and inferences that can be useful for future questions like the ones in the `examples` block.
                    I use the rethink memory to store all my questions, calcuations, and inferences. I am verbose and brainstorm using the rethink block many different types of potential questions and the reasoning required for answering them. I keep calling rethink_memory until I have all the potential inferences and calcuations, and check that there are no errors or extra information that would not be helpful for answering the kinds of questions in the `examples` block.
                    """,
                        limit=2000,
                    )

                    rethink_tool = client.tools.upsert_from_function(func=rethink_memory)
                    finish_rethink_tool = client.tools.upsert_from_function(func=finish_rethinking_memory)

                    sleep_time_memory_agent = client.agents.create(
                        name=f"{example_idx}_sleep_time_memory_agent_{idx}_{run_uuid}",
                        agent_type="sleeptime_agent",
                        system=get_prompt_text(sleep_time_system_block_filename, "system"),
                        memory_blocks=[
                            {"label": "human", "value": "I am a valuable source of information, I give problems that are worth thinking about deeply and carefully."},
                            {"label": "persona", "value": """I am an expert reasoning agent. When given a new context, I make calculations and inferences that can be useful for future questions like the ones in the `examples` block.
                    I use the rethink memory to store all my questions, calcuations, and inferences. I am verbose and brainstorm using the rethink block many different types of potential questions and the reasoning required for answering them. I keep calling rethink_memory until I have all the potential inferences and calcuations, and check that there are no errors or extra information that would not be helpful for answering the kinds of questions in the `examples` block."""}
                        ],
                        block_ids=[new_memory.id],
                        llm_config=LLM_CONFIG,
                        embedding="openai/text-embedding-ada-002",
                        tool_ids=[rethink_tool.id, finish_rethink_tool.id],
                        include_base_tools=False,
                        initial_message_sequence=[],
                    )

                    sleep_time_memory_agents.append(sleep_time_memory_agent)

                sleep_time_responses = []

                def process_sleep_time_agent(idx, sleep_time_agent, context, client):
                    response = client.agents.messages.create_stream(agent_id=sleep_time_agent.id, messages=[MessageCreate(role="user", content="[trigger_rethink_memory] New situation:" + context)], stream_tokens=True)
                    updated_agent = client.agents.retrieve(agent_id=sleep_time_agent.id)
                    return idx, response, updated_agent

                def process_conversation_agent(idx, conversation_agent, question, client):
                    response = client.agents.messages.create_stream(agent_id=conversation_agent.id, messages=[MessageCreate(role="user", content=question)], stream_tokens=True)
                    updated_agent = client.agents.retrieve(agent_id=conversation_agent.id)
                    return idx, response, updated_agent

                # Process sleep_time agents in parallel
                with ThreadPoolExecutor() as executor:
                    futures = [
                        executor.submit(
                            process_sleep_time_agent,
                            idx,
                            sleep_time_agent,
                            example['stateful_aime_context'],
                            client,
                        )
                        for idx, sleep_time_agent in enumerate(sleep_time_memory_agents)
                    ]
                    for future in tqdm(as_completed(futures), total=len(sleep_time_memory_agents)):
                        idx, response, updated_agent = future.result()
                        print("RESPONSE", response)
                        sleep_time_responses.append(response)
                        sleep_time_memory_agents[idx] = updated_agent
                # Process conversation agents in parallel
                final_responses = []
                with ThreadPoolExecutor() as executor:
                    final_message = example["stateful_aime_context"] + " " + example["stateful_aime_question"]

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
                    for future in tqdm(as_completed(futures), total=len(conversation_agents)):
                        idx, response, updated_agent = future.result()
                        final_responses.append(response)
                        conversation_agents[idx] = updated_agent
                result = {
                    "question": example["stateful_aime_question"],
                    "responses": [final_response.model_dump(exclude_none=True, mode="json") for final_response in final_responses],  # "final_response.model_dump(),
                    "sleep_time_memory": [
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
                break
            except Exception as e:
                print(f"Error processing example: {example}")
                print(e)
                result = None
                retry_id += 1
                raise e # TODO: remove this line to allow retries

        progress.update(1)
        return result

    async def process_example_async(sem, pool, idx, example):
        async with sem:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(pool, process_example, idx, example)
            return result

    max_concurrent = 1  # going much higher than 10 causes db connection issues (https://docs.sqlalchemy.org/en/20/errors.html#error-3o7r)
    sem = asyncio.Semaphore(max_concurrent)

    with ThreadPoolExecutor(max_workers=max_concurrent) as pool:
        results = await asyncio.gather(*[process_example_async(sem, pool, idx, example) for idx, example in enumerate(examples)])

    with jsonlines.open(output_file, mode="w") as writer:
        for result in tqdm(results):
            if result is not None:
                writer.write(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file", default="./predictions_aime.jsonl", required=False)
    parser.add_argument("--human_block_filename", default="human_verbosity_0", required=False)
    parser.add_argument("--persona_block_filename", default="persona_verbosity_0", required=False)
    parser.add_argument("--system_block_filename", default="convo_verbosity_0", required=False)
    parser.add_argument("--sleep_time_system_block_filename", default="sleep_time_base_aime", required=False)
    parser.add_argument("--model", default="o3-mini", required=False)
    parser.add_argument("--num_sleep_time_agents", default=1, required=False, type=int)
    parser.add_argument("--num_convo_agents", default=1, required=False, type=int, help="This should either be 1 or num_sleep_time_agents")

    args = parser.parse_args()

    asyncio.run(
        run_memory_edits(
            args.output_file,
            args.human_block_filename,
            args.persona_block_filename,
            args.system_block_filename,
            args.sleep_time_system_block_filename,
            args.model,
            args.num_sleep_time_agents,
            args.num_convo_agents,
        )
    )
