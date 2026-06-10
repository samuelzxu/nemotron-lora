import boto3, sys
from botocore.config import Config
PROBLEM = r"The sum of all positive integers $m$ such that $\frac{13!}{m}$ is a perfect square can be written as $2^a3^b5^c7^d11^e13^f,$ where $a,b,c,d,e,$ and $f$ are positive integers. Find $a+b+c+d+e+f.$"
SUFFIX = "\nPlease put your final answer inside \\boxed{}. For example: \\boxed{your answer}"
cfg = Config(retries={"max_attempts": 3, "mode": "adaptive"}, read_timeout=300, connect_timeout=15)
br = boto3.client("bedrock-runtime", region_name="us-east-1", config=cfg)
for mid in ["us.anthropic.claude-sonnet-4-6", "anthropic.claude-sonnet-4-6"]:
    try:
        r = br.converse(
            modelId=mid,
            messages=[{"role": "user", "content": [{"text": PROBLEM + SUFFIX}]}],
            inferenceConfig={"maxTokens": 16000, "temperature": 1.0},
            additionalModelRequestFields={"reasoning_config": {"type": "enabled", "budget_tokens": 8000}},
        )
        content = r["output"]["message"]["content"]
        rsn = "".join(c["reasoningContent"]["reasoningText"]["text"] for c in content if "reasoningContent" in c)
        txt = "".join(c["text"] for c in content if "text" in c)
        print(f"=== MODEL {mid} OK ===")
        print("reasoning_chars:", len(rsn), "| text_chars:", len(txt), "| usage:", r.get("usage"))
        print("\n--- REASONING (first 1200 chars) ---\n", rsn[:1200])
        print("\n--- ANSWER TEXT ---\n", txt[:600])
        break
    except Exception as e:
        print(f"=== MODEL {mid} FAILED: {type(e).__name__}: {str(e)[:240]} ===")
