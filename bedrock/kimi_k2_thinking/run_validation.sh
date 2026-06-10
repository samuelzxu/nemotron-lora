#!/bin/bash
cd /home/ec2-user/dev/nemotron_training
exec bedrock/.venv/bin/python bedrock/driver.py \
  --model-id moonshot.kimi-k2-thinking \
  --label kimi_k2_thinking \
  --max-tokens 32000 \
  --n-gens 4 \
  --temperature 0.8 \
  --concurrency 4 \
  --limit 4
