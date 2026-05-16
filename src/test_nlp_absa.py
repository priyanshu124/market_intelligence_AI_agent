# test_absa_model.py
# =============================================================================
# Quick test to verify DeBERTa ABSA model loads and scores correctly.
# Run this BEFORE nlp_absa.py to confirm everything works.
#
# USAGE:
#   python test_absa_model.py
#
# EXPECTED OUTPUT (if working correctly):
#   "reporting is fantastic" + "financial reporting" → POSITIVE ~0.85+
#   "support takes weeks"    + "customer support"    → NEGATIVE ~0.80+
#   "easy to navigate"       + "ease of use"         → POSITIVE ~0.85+
#   "implementation nightmare" + "implementation"    → NEGATIVE ~0.75+
# =============================================================================

import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import torch.nn.functional as F

MODEL = "yangheng/deberta-v3-base-absa-v1.1"

print("=" * 60)
print("  DeBERTa ABSA MODEL TEST")
print("=" * 60)
print(f"\nLoading model: {MODEL}")
print("(~180MB download on first run)\n")

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model     = AutoModelForSequenceClassification.from_pretrained(MODEL)
model.eval()

id2label = {k: v.lower() for k, v in model.config.id2label.items()}
print(f"Label mapping: {id2label}")
print(f"Parameters:    {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M\n")

# Test pairs — known expected sentiment
TEST_PAIRS = [
    # (text, aspect, expected_label)
    ("The financial reporting is absolutely fantastic and saves us hours",
     "financial reporting",        "positive"),
    ("Customer support takes weeks to respond to any ticket",
     "customer support",           "negative"),
    ("Very easy to navigate and the interface is intuitive",
     "ease of use",                "positive"),
    ("Implementation was a nightmare, took 14 months and cost 3x budget",
     "implementation",             "negative"),
    ("The pricing went up 40% with no added value",
     "pricing",                    "negative"),
    ("Bank reconciliation works perfectly and saves hours each month",
     "bank reconciliation",        "positive"),
    ("Reporting is limited and very difficult to customize",
     "reporting",                  "negative"),
    ("The inventory management module is comprehensive and accurate",
     "inventory management",       "positive"),
]

print(f"{'Text':<50} {'Aspect':<25} {'Expected':>10} {'Got':>10} {'Pos':>6} {'Neg':>6} {'OK':>4}")
print("-" * 115)

passed = 0
for text, aspect, expected in TEST_PAIRS:
    # Format exactly as the pipeline does
    pair   = f"{text} [SEP] {aspect}"
    inputs = tokenizer(pair, return_tensors="pt", truncation=True, max_length=128)

    with torch.no_grad():
        outputs = model(**inputs)
        probs   = F.softmax(outputs.logits, dim=-1)[0].numpy()

    scores = {id2label[i]: float(probs[i]) for i in range(len(probs))}
    label  = max(scores, key=scores.get)
    ok     = "✅" if label == expected else "❌"
    if label == expected:
        passed += 1

    print(f"  {text[:48]:<50} {aspect:<25} {expected:>10} {label:>10} "
          f"{scores.get('positive',0):>6.3f} {scores.get('negative',0):>6.3f} {ok}")

print(f"\nResult: {passed}/{len(TEST_PAIRS)} tests passed")

if passed == len(TEST_PAIRS):
    print("\n✅ Model working correctly — safe to run nlp_absa.py")
elif passed >= len(TEST_PAIRS) * 0.75:
    print(f"\n⚠️  Model mostly working ({passed}/{len(TEST_PAIRS)}) — proceed with caution")
else:
    print(f"\n❌ Model not working ({passed}/{len(TEST_PAIRS)}) — check model and input format")
    print("   Try: pip install transformers==4.35.0")