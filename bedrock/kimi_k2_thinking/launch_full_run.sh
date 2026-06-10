#!/bin/bash
# Full run launcher for kimi_k2_thinking
# Model: moonshot.kimi-k2-thinking
# Thinking: built-in (reasoningContent returned automatically, no --thinking flag needed)
# Max tokens: 65536 (hard problems use ~25k output tokens for reasoning)

source ~/.claudeinit 2>/dev/null

cd /home/ec2-user/dev/nemotron_training

nohup bedrock/.venv/bin/python bedrock/driver.py \
  --model-id moonshot.kimi-k2-thinking \
  --label kimi_k2_thinking \
  --max-tokens 65536 \
  --n-gens 16 \
  --limit 0 \
  --temperature 0.8 \
  --concurrency 6 \
  >> bedrock/kimi_k2_thinking/run.log 2>&1 &

echo "Full run launched with PID: $!"
