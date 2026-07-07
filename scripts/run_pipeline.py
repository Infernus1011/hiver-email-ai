#!/usr/bin/env python3
"""
End-to-end pipeline: generate dataset -> generate responses -> evaluate.
"""
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.generate_dataset import generate_email_pair, EmailPair
from generator.response_generator import ResponseGenerator, EmailContext
from evaluator.metrics import EmailEvaluator
from dataclasses import asdict
import random
import time

def main():
    print("=" * 60)
    print("HIVER EMAIL AI CHALLENGE - FULL PIPELINE")
    print("=" * 60)
    
    # Configuration
    num_pairs = 50
    output_base = "output"
    os.makedirs(output_base, exist_ok=True)
    
    # Step 1: Generate Dataset
    print("\n[1/3] Generating synthetic email dataset...")
    pairs = [generate_email_pair(i) for i in range(num_pairs)]
    
    # Save dataset
    dataset_path = os.path.join(output_base, "email_dataset.jsonl")
    with open(dataset_path, "w") as f:
        for pair in pairs:
            f.write(json.dumps(asdict(pair)) + "\n")
    print(f"  Generated {len(pairs)} email pairs -> {dataset_path}")
    
    # Step 2: Generate AI Responses
    print("\n[2/3] Generating AI responses...")
    generator = ResponseGenerator(preferred_provider="mock")
    
    contexts = [EmailContext(
        incoming=p.incoming,
        category=p.category,
        urgency=p.urgency,
        customer_tier=p.customer_tier
    ) for p in pairs]
    
    responses = generator.batch_generate(contexts)
    
    generated_path = os.path.join(output_base, "generated_responses.jsonl")
    with open(generated_path, "w") as f:
        for i, resp in enumerate(responses):
            f.write(json.dumps({
                "id": f"gen_{i:05d}",
                "response": resp.response,
                "model": resp.model,
                "tokens_used": resp.tokens_used,
                "latency_ms": resp.latency_ms,
                "confidence": resp.confidence
            }) + "\n")
    print(f"  Generated {len(responses)} responses -> {generated_path}")
    
    # Step 3: Evaluate
    print("\n[3/3] Evaluating response quality...")
    evaluator = EmailEvaluator()
    
    per_response = []
    metric_sums = {}
    metric_counts = {}
    
    for i, (pair, resp) in enumerate(zip(pairs, responses)):
        result = evaluator.evaluate(
            generated=resp.response,
            expected=pair.expected_reply,
            context={"category": pair.category, "urgency": pair.urgency, "tier": pair.customer_tier}
        )
        
        per_response.append({
            "id": f"eval_{i:05d}",
            "email_id": pair.id,
            "category": pair.category,
            "urgency": pair.urgency,
            "tier": pair.customer_tier,
            "incoming": pair.incoming,
            "expected_reply": pair.expected_reply,
            "generated_reply": resp.response,
            "overall_score": round(result.overall_score, 4),
            "metrics": {k: round(v, 4) for k, v in result.metrics.items()}
        })
        
        for metric, value in result.metrics.items():
            metric_sums.setdefault(metric, 0.0)
            metric_counts.setdefault(metric, 0)
            metric_sums[metric] += value
            metric_counts[metric] += 1
    
    avg_metrics = {
        m: round(metric_sums[m] / metric_counts[m], 4)
        for m in metric_sums
    }
    
    weighted_avg = sum(avg_metrics.values()) / len(avg_metrics) if avg_metrics else 0.0
    
    evaluation_results = {
        "overall_score": round(weighted_avg, 4),
        "average_metrics": avg_metrics,
        "total_evaluated": len(per_response),
        "per_response": per_response,
        "note": "Includes all evaluated email-response pairs."
    }
    
    eval_path = os.path.join(output_base, "evaluation_results.json")
    with open(eval_path, "w") as f:
        json.dump(evaluation_results, f, indent=2)
    
    print(f"  Evaluation saved -> {eval_path}")
    
    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Overall Score: {weighted_avg:.4f}")
    print(f"  Responses Evaluated: {len(per_response)}")
    print(f"\n  Metric Breakdown:")
    for metric, score in sorted(avg_metrics.items()):
        bar = "█" * int(score * 30) + "░" * (30 - int(score * 30))
        print(f"    {metric:25s} [{bar}] {score:.4f}")
    
    print(f"\n  Output files:")
    print(f"    Dataset:      {dataset_path}")
    print(f"    Generations:  {generated_path}")
    print(f"    Evaluation:   {eval_path}")
    print("\nDone! Pipeline completed successfully.")

if __name__ == "__main__":
    main()
