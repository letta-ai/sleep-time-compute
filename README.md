# sleep-time-compute
Code and data accompanying the paper "Sleep-time Compute: Beyond Inference Scaling at Test-time" 

## Data

Stateful GSM-Symbolic: https://huggingface.co/datasets/letta-ai/stateful-gsm-symbolic

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

## Stateful GSM-Symbolic Experiments
## Stateful AIME Experiments  
## Reference
If you find this helpful, please consider citing:
```bibtex
@article{lin2025sleep,
  title={Sleep-time Compute: Beyond Inference Scaling at Test-time},
  author={Lin, Kevin and Snell, Charlie, and Wang, Yu and Packer, Charles and Wooders, Sarah and Stoica, Ion, and Gonzalez, Joseph E.},
  journal={arXiv preprint},
  year={2025}
}
```

