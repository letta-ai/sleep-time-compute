"""
Evaluate the results for the stateful GSM-Symbolic dataset

Usage:

    python evaluate_gsm_symbolic.py --input_file <path_to_input_file>
"""

import argparse
import json
import re
import statistics
import jsonlines

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def standardize_answer(answer: str) -> str:
    return answer.replace("$", "").replace(",", "").strip(".")


def evaluate(input_file: str):
    correct = 0
    total = 0
    example_num_rethinks = []

    with jsonlines.open(input_file) as reader:
        responses = []
        for obj in reader:
            response = obj["responses"][0]
            responses.append(response)
            answer = standardize_answer(re.sub(
                r"(?s).*#### ", "", obj["answer"]))
            final_answer = ""
            for message in response["messages"]:
                if message["message_type"] == "assistant_message":
                    final_answer = message["content"]
                    regex_str = "(-?[$0-9.,]{2,})|(-?[0-9]+)"
                    matches = re.findall(regex_str, final_answer)
                    if matches == []:
                        continue
                    final_answer = "".join(matches[-1])
                    final_answer = standardize_answer(final_answer)

            if final_answer == answer:
                correct += 1
            total += 1
            num_rethinks = 0
            if "offline_responses" in obj and len(obj["offline_responses"]) > 0:
                for message in obj["offline_responses"][0]["messages"]:
                    if message["message_type"] == "tool_call_message":
                        if message["tool_call"]["name"] == "rethink_memory":
                            num_rethinks += 1
                example_num_rethinks.append(num_rethinks)
        results = {
            "accuracy": correct / total,
            "avg_completion_tokens": statistics.mean([r["usage"]["completion_tokens"] for r in responses]),
        }
        logger.info(f"Results: {results}")
        with open(input_file.replace(".jsonl", "_results.json"), "w") as f:
            json.dump(results, f, indent=4)
        logger.info(f"Wrote results to {input_file.replace('.jsonl', '_results.json')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    args = parser.parse_args()

    evaluate(args.input_file)
