"""
Evaluate the results for the GSM8K dataset

Usage:

    python evaluate_gsm_symbolic.py --input_file <path_to_input_file>
"""

import argparse
import json
import re
import statistics
import jsonlines


def standardize_answer(answer: str):
    answer = answer.replace("$", "")
    answer = answer.replace(",", "")
    answer = answer.strip(".")
    return answer


def evaluate(input_file: str):
    correct = 0
    total = 0
    example_num_rethinks = []

    with jsonlines.open(input_file) as reader:
        responses = []
        for obj in reader:
            response = obj["responses"][0]
            responses.append(response)
            ignore_regex = "(?s).*#### "
            answer = obj["answer"]
            answer = re.sub(ignore_regex, "", answer)
            answer = standardize_answer(answer)

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

            if final_answer == answer:  # TODO: should we convert to float
                correct += 1
            else:
                print(f"Final answer: {final_answer}")
                print(f"Answer: {answer}")
            total += 1

            num_rethinks = 0
            if "offline_responses" in obj and len(obj["offline_responses"]) > 0:
                for message in obj["offline_responses"][0]["messages"]:
                    if message["message_type"] == "tool_call_message":
                        if message["tool_call"]["name"] == "rethink_memory":
                            num_rethinks += 1
                example_num_rethinks.append(num_rethinks)
        import pdb; pdb.set_trace() 
        print([r["usage"]["completion_tokens"] for r in responses])
        results = {
            "accuracy": correct / total,
            "avg_completion_tokens": statistics.mean([r["usage"]["completion_tokens"] for r in responses]),
        }
        print(results)
        with open(input_file.replace(".jsonl", "_results.json"), "w") as f:
            json.dump(results, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    args = parser.parse_args()

    evaluate(args.input_file)
