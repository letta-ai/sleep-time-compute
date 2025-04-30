# Sleep-time Compute (paper code)
Code and data accompanying the paper [Sleep-time Compute: Beyond Inference Scaling at Test-time](https://arxiv.org/abs/2504.13171) from Letta and UC Berkeley.

> [!NOTE]
> This repo contains code to reproduce the empirical AIME/GSM results in the Sleep-time Compute **research paper**. If you are interested in building agents that use sleep-time compute, view the official [sleep-time agents developer docs](https://docs.letta.com/guides/agents/sleep-time-agents).

Useful links:
* [arXiv paper](https://arxiv.org/abs/2504.13171)
* [blog post](https://www.letta.com/blog/sleep-time-compute)
* [sleep-time agent documentation](https://docs.letta.com/guides/agents/sleep-time-agents)

## Data

Stateful AIME 2024: https://huggingface.co/datasets/letta-ai/stateful-aime-2024

Stateful AIME 2025: https://huggingface.co/datasets/letta-ai/stateful-aime-2025

Stateful GSM-Symbolic: https://huggingface.co/datasets/letta-ai/stateful-gsm-symbolic

SWE-Features: https://huggingface.co/datasets/letta-ai/SWE-Features

## Setup
```
conda create -n sleep-time-compute python=3.12 --yes
conda activate sleep-time-compute
pip install -r requirements.txt
```

Run a Letta server, following the instructions here: https://github.com/letta-ai/letta
```
docker run \
  -v ~/.letta/.persist/pgdata:/var/lib/postgresql/data \
  -p 8283:8283 \
  -e OPENAI_API_KEY=${OPENAI_API_KEY} \
  letta/letta:latest
```


## Stateful AIME Experiments  

## Stateful GSM-Symbolic Experiments
Download data
```
cd data
wget https://huggingface.co/datasets/letta-ai/stateful-gsm-symbolic/resolve/main/stateful_gsm_symbolic_p1.jsonl
wget https://huggingface.co/datasets/letta-ai/stateful-gsm-symbolic/resolve/main/stateful_gsm_symbolic_p2.jsonl
cd ..
```

Run the script to generate predictions
```
python run_stateful_gsm_symbolic.py  \
--input_file ./data/stateful_gsm_symbolic_p2.jsonl \
--output_file ./predictions-stateful_gsm_symbolic_p2.jsonl  \
--test_time_persona_block_filename persona_verbosity_2
```

Evaluate the results
```
python evaluate_gsm_symbolic.py  \
  --input_file ./predictions-stateful_gsm_symbolic_p2.jsonl
```

## Reference
If you find this helpful, please consider citing:
```bibtex
@article{lin-snell-etal:2025:arxiv},
  title={Sleep-time Compute: Beyond Inference Scaling at Test-time},
  author={Lin, Kevin and Snell, Charlie, and Wang, Yu and Packer, Charles and Wooders, Sarah and Stoica, Ion, and Gonzalez, Joseph E.},
  journal={arXiv:2504.13171},
  year={2025}
}
```

