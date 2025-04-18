import argparse
import json
import re
import jsonlines

def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx == None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval

def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[: len(left)] == left
        assert s[-1] == "}"
        return s[len(left) : -1]
    except:
        return None

def parse_answer(response):
    try:
        return int(remove_boxed(last_boxed_only_string(response)))
    except:
        regex_str = r"([1-9][0-9][0-9]|[1-9][0-9]|[1-9])"
        matches = re.findall(regex_str, response)
        if matches == []:
            return None
        return int(matches[-1])

def grade_answer(predicted_answer, actual_answer):
    if predicted_answer is None:
        return False
    return predicted_answer == actual_answer

def get_data_for_run(data, idx=-1):
    grades, sleep_tokens, chat_tokens = [], [], []
    for item in data:
        try:
            pred_ans = parse_answer(item["responses"][0][idx]["content"])
        except:
            print("failed to parse answer")
            pred_ans = ""
        gt_ans = item["answer"]
        grade = grade_answer(pred_ans, gt_ans)
        grades.append(grade)
        if len(item["responses"]) > 0:
            chat_tokens.append(item["conversation_usage_list"][0]["completion_tokens"])
            if "sleep_time_responses" in item:
                sleep_tokens.append(item["sleep_time_usage_list"][0]["completion_tokens"])
            else:
                sleep_tokens.append(0)
        else:
            chat_tokens.append(0)
            sleep_tokens.append(0)
    return {
        "accuracy": sum(grades) / len(grades),
        "test_time_avg_tokens": sum(chat_tokens) / len(chat_tokens),
        "sleep_time_avg_tokens": sum(sleep_tokens) / len(sleep_tokens),
        "num_examples": len(data),
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_file", default="./predictions_aime.jsonl", required=False)

    args = parser.parse_args()

    with jsonlines.open(args.results_file, mode="r") as reader:
        data = [line for line in reader]

    print(get_data_for_run(data))
