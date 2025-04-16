"""
Script to generate the data for stateful GSM-Symbolic


Example usage:
    python create_stateful_gsm_symbolic.py --input_file data/GSM_p2.jsonl --output_file data/stateful_gsm_symbolic_p2.jsonl 
    python create_stateful_gsm_symbolic.py --input_file data/GSM_p1.jsonl --output_file data/stateful_gsm_symbolic_p1.jsonl 

"""
import jsonlines
import argparse
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_jsonl(input_file: str, output_file: str) -> None:
    """
    Process each line by, by splitting the original question into a "stateful_gsm_symbolic_context"
    and a "stateful_gsm_symbolic_question".

    Args:
        input_file (str): Path to input JSONL file
        output_file (str): Path to output JSONL file
    """
    total_num_lines = 0
    with jsonlines.open(input_file, 'r') as infile, \
         jsonlines.open(output_file, 'w') as outfile:
        for example in infile:
            sentences = list(map(lambda x: x.strip(), filter(lambda x: x.strip() != "", example["question"].split("."))))
            ends_with_period = sentences[-1] == ""
            context = ". ".join(sentences[:-1]).strip() + "."
            question = sentences[-1] + ("." if ends_with_period else "")
            example['stateful_gsm_symbolic_context'] = context
            example['stateful_gsm_symbolic_question'] = question
            outfile.write(example)
            total_num_lines += 1
    logger.info(f"Processed {total_num_lines} lines.")
         
def main():
    parser = argparse.ArgumentParser(description='Process JSONL file to keep only question and answer fields')
    parser.add_argument('--input_file', help='Path to input JSONL file')
    parser.add_argument('--output_file', help='Path to output JSONL file')
    args = parser.parse_args()
    logger.info(f"Processing {args.input_file} to {args.output_file}")
    process_jsonl(args.input_file, args.output_file)

if __name__ == '__main__':
    main()