#!/usr/bin/env python3
"""
Comprehensive email response evaluation metrics.
Computes per-response and overall quality scores.
"""
import re
import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

@dataclass
class EvaluationResult:
    overall_score: float
    metrics: Dict[str, float]
    details: Dict[str, str] = field(default_factory=dict)

class EmailEvaluator:
    """Evaluates generated email responses against expected replies."""
    
    def __init__(self):
        # Weight configuration for composite score
        self.weights = {
            "semantic_similarity": 0.25,
            "rouge_f1": 0.15,
            "response_length_score": 0.10,
            "politeness_score": 0.10,
            "action_item_score": 0.10,
            "specificity_score": 0.10,
            "tone_match_score": 0.10,
            "readability_score": 0.05,
            "bler_score": 0.05,
        }
    
    def evaluate(self, generated: str, expected: str, context: Dict = None) -> EvaluationResult:
        metrics = {}
        details = {}
        
        # Core NLP metrics
        metrics["semantic_similarity"] = self._semantic_similarity(generated, expected)
        metrics["rouge_f1"] = self._rouge_f1(generated, expected)
        metrics["bler_score"] = self._bler_score(generated, expected)
        
        # Quality metrics
        metrics["response_length_score"] = self._response_length_score(generated)
        metrics["politeness_score"] = self._politeness_score(generated)
        metrics["action_item_score"] = self._action_item_score(generated, context)
        metrics["specificity_score"] = self._specificity_score(generated)
        metrics["tone_match_score"] = self._tone_match_score(generated, expected)
        metrics["readability_score"] = self._readability_score(generated)
        
        # Calculate weighted composite score
        overall_score = sum(
            metrics.get(metric, 0) * weight
            for metric, weight in self.weights.items()
        )
        overall_score = max(0.0, min(1.0, overall_score))
        
        return EvaluationResult(
            overall_score=overall_score,
            metrics=metrics,
            details=details
        )
    
    def _semantic_similarity(self, generated: str, expected: str) -> float:
        try:
            from sentence_transformers import SentenceTransformer, util
            model = SentenceTransformer('all-MiniLM-L6-v2')
            emb1 = model.encode(generated, convert_to_tensor=True)
            emb2 = model.encode(expected, convert_to_tensor=True)
            similarity = util.cos_sim(emb1, emb2).item()
            return float(max(0.0, min(1.0, similarity)))
        except ImportError:
            return self._token_overlap_similarity(generated, expected)
    
    def _token_overlap_similarity(self, gen: str, exp: str) -> float:
        """Fallback: compute similarity using token overlap (Jaccard)."""
        import re
        def tokenize(text):
            return set(re.findall(r'\b\w+\b', text.lower()))
        g_tokens = tokenize(gen)
        e_tokens = tokenize(exp)
        if not g_tokens or not e_tokens:
            return 0.0
        intersection = g_tokens & e_tokens
        union = g_tokens | e_tokens
        return len(intersection) / len(union)
    
    def _rouge_f1(self, generated: str, expected: str) -> float:
        """Compute ROUGE-L F1 score."""
        try:
            from rouge_score import rouge_scorer
            scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
            scores = scorer.score(expected, generated)
            return float(scores['rougeL'].fmeasure)
        except ImportError:
            return self._rouge_simple(generated, expected)
    
    def _rouge_simple(self, gen: str, exp: str) -> float:
        """Simple ROUGE-like F1 computation."""
        def lcs(s1, s2):
            m, n = len(s1), len(s2)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if s1[i-1] == s2[j-1]:
                        dp[i][j] = dp[i-1][j-1] + 1
                    else:
                        dp[i][j] = max(dp[i-1][j], dp[i][j-1])
            return dp[m][n]
        
        g_tokens = re.findall(r'\b\w+\b', gen.lower())
        e_tokens = re.findall(r'\b\w+\b', exp.lower())
        
        lcs_len = lcs(g_tokens, e_tokens)
        if len(e_tokens) == 0 or len(g_tokens) == 0:
            return 0.0
        
        precision = lcs_len / len(g_tokens)
        recall = lcs_len / len(e_tokens)
        
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)
    
    def _bler_score(self, generated: str, expected: str) -> float:
        """Bilingual Evaluation Understudy-style n-gram precision."""
        def get_ngrams(text, n=2):
            tokens = re.findall(r'\b\w+\b', text.lower())
            return set(' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1))
        
        g_ngrams = get_ngrams(generated)
        e_ngrams = get_ngrams(expected)
        
        if not g_ngrams or not e_ngrams:
            return 0.0
        
        matches = g_ngrams & e_ngrams
        precision = len(matches) / len(g_ngrams)
        
        # Brevity penalty
        g_len = len(generated.split())
        e_len = len(expected.split())
        if g_len < e_len:
            bp = math.exp(1 - e_len / g_len) if g_len > 0 else 0
        else:
            bp = 1.0
        
        return float(precision * bp)
    
    def _response_length_score(self, response: str) -> float:
        """Score based on appropriate response length (50-200 words is ideal)."""
        word_count = len(response.split())
        if 50 <= word_count <= 200:
            return 1.0
        elif word_count < 20:
            return 0.2
        elif word_count < 50:
            return 0.4 + (word_count / 50) * 0.6
        elif word_count > 300:
            return 0.3
        elif word_count > 200:
            return 1.0 - min(1.0, (word_count - 200) / 200)
        return 0.5
    
    def _politeness_score(self, response: str) -> float:
        """Score based on politeness markers."""
        polite_markers = [
            r'\b(please|thank\s*you|thanks|appreciate|glad|happy\s+to\s+help|sorry|apologize|regret)\b',
            r'\b(we\'?re\s+(here\s+to\s+help|happy\s+to)|let\s+(us|me)\s+know)\b',
            r'\b(have\s+a\s+great|best\s+regards|kind\s+regards|warmly|cheers)\b',
        ]
        
        response_lower = response.lower()
        matches = 0
        for pattern in polite_markers:
            matches += len(re.findall(pattern, response_lower))
        
        # Cap at reasonable politeness
        return min(1.0, matches / 3.0)
    
    def _action_item_score(self, response: str, context: Dict = None) -> float:
        """Score based on inclusion of specific action items."""
        action_markers = [
            r'\b(I\'?ve?\s+\w+(ed|ed)\s+)',
            r'\b(we\s+have\s+\w+(ed)\s+)',
            r'\b(as\s+a\s+one-time\s+courtesy)',
            r'\b(ticket\s*#?\d+)',
            r'\b(in\s+\d+\s+(business\s+)?days?)',
            r'\b(your request has been|l\'ve (processed|submitted|updated|escalated|initiated))',
            r'\b(we\'?ll\s+(update|notify|follow\s+up|process|refund))',
            r'\b(you can expect|within|by\s+\w+)',
        ]
        
        response_lower = response.lower()
        matches = sum(1 for p in action_markers if re.search(p, response_lower))
        return min(1.0, matches / 3.0)
    
    def _specificity_score(self, response: str) -> float:
        """Score based on use of specific details vs. vague language."""
        specific_patterns = [
            r'\b(ticket\s*#?\d+)',
            r'\b\d+\s*(business\s+)?(hours|days|minutes)',
            r'\b(version\s*\d+\.\d+|iOS|Android|Windows|Mac)\b',
            r'\b(Mr\.|Mrs\.|Ms\.)\s\w+',
            r'\$\d+(\.\d{2})?',
            r'\b(update|notification|email|confirmation)\s+(in|within|in about)\s+\d+',
        ]
        
        vague_patterns = [
            r'\b(soon|shortly|asap|as\s+soon\s+as\s+possible|right\s+away)\b',
            r'\b(someone|appropriate\s+team|relevant\s+department)\b',
            r'\b(at\s+your\s+earliest\s+convenience)\b',
        ]
        
        specific = sum(1 for p in specific_patterns if re.search(p, response, re.IGNORECASE))
        vague = sum(1 for p in vague_patterns if re.search(p, response, re.IGNORECASE))
        
        base_score = min(1.0, specific / 2.0)
        penalty = vague * 0.15
        
        return max(0.0, base_score - penalty)
    
    def _tone_match_score(self, generated: str, expected: str) -> float:
        """Score how well the tone matches the expected response."""
        def formality_score(text):
            formal = len(re.findall(r'\b(regarding|approximately|subsequently|however|therefore|please\s+do\s+not\s+hesitate)\b', text.lower()))
            informal = len(re.findall(r'\b(hey|no\s+worries|awesome|great\s+to|sorry\s+about\s+that)\b', text.lower()))
            total = formal + informal
            if total == 0:
                return 0.5
            return formal / total
        
        gen_formal = formality_score(generated)
        exp_formal = formality_score(expected)
        
        return 1.0 - abs(gen_formal - exp_formal)
    
    def _readability_score(self, response: str) -> float:
        """Score readability using Flesch-Kincaid-like metric."""
        sentences = re.split(r'[.!?]+', response)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 0]
        
        if not sentences:
            return 0.0
        
        words = response.split()
        if not words:
            return 0.0
        
        avg_sentence_len = len(words) / len(sentences) if sentences else 0
        syllables = sum(self._count_syllables(w) for w in words)
        avg_syllables = syllables / len(words) if words else 0
        
        # Score: 0.0 if very complex, 1.0 if very readable
        readability = 206.835 - 1.015 * avg_sentence_len - 84.6 * avg_syllables
        
        # Normalize to 0-1
        norm = (readability + 20) / 120
        return max(0.0, min(1.0, norm))
    
    def _count_syllables(self, word: str) -> int:
        word = word.lower()
        count = 0
        vowels = 'aeiou'
        if len(word) <= 3:
            return 1
        if word[0] in vowels:
            count += 1
        for i in range(1, len(word)):
            if word[i] in vowels and word[i-1] not in vowels:
                count += 1
        if word.endswith('e'):
            count -= 1
        if word.endswith('le') and len(word) > 2 and word[-3] not in vowels:
            count += 1
        if count == 0:
            count += 1
        return count

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated", type=str, required=True, help="Generated responses JSONL")
    parser.add_argument("--expected", type=str, required=True, help="Expected responses JSONL")
    parser.add_argument("--output", type=str, default="eval_results.json", help="Output JSON path")
    args = parser.parse_args()
    
    import json
    
    generated = []
    with open(args.generated, "r") as f:
        for line in f:
            generated.append(json.loads(line.strip()))
    
    expected = []
    with open(args.expected, "r") as f:
        for line in f:
            expected.append(json.loads(line.strip()))
    
    evaluator = EmailEvaluator()
    per_response = []
    overall_metrics = {}
    metric_sums = {}
    metric_counts = {}
    
    print(f"Evaluating {len(generated)} generated responses against {len(expected)} expected...")
    
    for g, e in zip(generated, expected):
        g_text = g.get("response", "")
        e_text = e.get("expected_reply", "")
        cat = e.get("category", "unknown")
        urg = e.get("urgency", "unknown")
        tier = e.get("customer_tier", "unknown")
        
        result = evaluator.evaluate(g_text, e_text, {"category": cat, "urgency": urg, "tier": tier})
        
        per_response.append({
            "generated_id": g.get("id", ""),
            "expected_id": e.get("id", ""),
            "category": cat,
            "urgency": urg,
            "tier": tier,
            "overall_score": round(result.overall_score, 4),
            "metrics": {k: round(v, 4) for k, v in result.metrics.items()}
        })
        
        for metric, value in result.metrics.items():
            if metric not in metric_sums:
                metric_sums[metric] = 0.0
                metric_counts[metric] = 0
            metric_sums[metric] += value
            metric_counts[metric] += 1
    
    overall_metrics = {
        metric: round(metric_sums[metric] / metric_counts[metric], 4) if metric_counts[metric] > 0 else 0.0
        for metric in metric_sums
    }
    
    overall_score = sum(overall_metrics.values()) / len(overall_metrics) if overall_metrics else 0.0
    
    output = {
        "overall_score": round(overall_score, 4),
        "average_metrics": overall_metrics,
        "total_evaluated": len(per_response),
        "per_response": per_response
    }
    
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nEvaluation Complete!")
    print(f"Total responses evaluated: {len(per_response)}")
    print(f"Overall Score: {overall_score:.4f}")
    print(f"\nPer-Metric Breakdown:")
    for metric, score in sorted(overall_metrics.items()):
        bar_len = int(score * 50)
        bar = "█" * bar_len + "░" * (50 - bar_len)
        print(f"  {metric:30s} [{bar}] {score:.4f}")
    print(f"\nSaved evaluation results to {args.output}")

if __name__ == "__main__":
    main()
