#!/usr/bin/env python3
"""
AI-powered email response generator.
Supports multiple LLM providers (OpenAI, Anthropic) with fallback.
"""
import os
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    load_dotenv = lambda: None

@dataclass
class EmailContext:
    incoming: str
    category: str
    urgency: str
    customer_tier: str
    thread_history: List[str] = None

@dataclass
class GeneratedResponse:
    response: str
    model: str
    tokens_used: int
    latency_ms: float
    confidence: float

class LLMProvider(ABC):
    @abstractmethod
    def generate(self, context: EmailContext) -> GeneratedResponse:
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        pass

class OpenAIProvider(LLMProvider):
    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self.client = None
        self._init_client()
    
    def _init_client(self):
        try:
            from openai import OpenAI
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                self.client = OpenAI(api_key=api_key)
        except Exception:
            pass
    
    def is_available(self) -> bool:
        return self.client is not None
    
    def generate(self, context: EmailContext) -> GeneratedResponse:
        import time
        start = time.time()
        
        prompt = self._build_prompt(context)
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self._system_prompt()},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=300,
        )
        
        latency = (time.time() - start) * 1000
        
        return GeneratedResponse(
            response=response.choices[0].message.content.strip(),
            model=self.model,
            tokens_used=response.usage.total_tokens,
            latency_ms=latency,
            confidence=0.85
        )
    
    def _system_prompt(self) -> str:
        return """You are a professional customer support agent for Hiver, a shared inbox tool that works inside Gmail.
Write concise, helpful, and empathetic email replies. Match the tone to the customer's urgency and tier.
Keep responses under 150 words. Always address the customer by name if available.
Be specific about actions taken or next steps. Never make up fake ticket numbers or fake actions."""
    
    def _build_prompt(self, context: EmailContext) -> str:
        tier_notes = {
            "free": "Standard support response time: 24-48 hours",
            "pro": "Priority support response time: 4-8 hours",
            "enterprise": "Dedicated support response time: 1-2 hours, include escalation path"
        }
        
        urgency_notes = {
            "low": "Customer is not in a rush",
            "medium": "Customer expects response within business hours",
            "high": "Customer is blocked, needs urgent attention",
            "critical": "Production issue, highest priority"
        }
        
        return f"""Category: {context.category}
Urgency: {context.urgency} ({urgency_notes.get(context.urgency, '')})
Customer Tier: {context.customer_tier} ({tier_notes.get(context.customer_tier, '')})

Incoming Email:
{context.incoming}

Write a professional reply:"""

class AnthropicProvider(LLMProvider):
    def __init__(self, model: str = "claude-3-haiku-20240307"):
        self.model = model
        self.client = None
        self._init_client()
    
    def _init_client(self):
        try:
            import anthropic
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if api_key:
                self.client = anthropic.Anthropic(api_key=api_key)
        except Exception:
            pass
    
    def is_available(self) -> bool:
        return self.client is not None
    
    def generate(self, context: EmailContext) -> GeneratedResponse:
        import time
        start = time.time()
        
        prompt = self._build_prompt(context)
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=300,
            temperature=0.3,
            system=self._system_prompt(),
            messages=[{"role": "user", "content": prompt}]
        )
        
        latency = (time.time() - start) * 1000
        
        return GeneratedResponse(
            response=response.content[0].text.strip(),
            model=self.model,
            tokens_used=response.usage.input_tokens + response.usage.output_tokens,
            latency_ms=latency,
            confidence=0.85
        )
    
    def _system_prompt(self) -> str:
        return """You are a professional customer support agent for Hiver, a shared inbox tool that works inside Gmail.
Write concise, helpful, and empathetic email replies. Match the tone to the customer's urgency and tier.
Keep responses under 150 words. Always address the customer by name if available.
Be specific about actions taken or next steps. Never make up fake ticket numbers or fake actions."""
    
    def _build_prompt(self, context: EmailContext) -> str:
        tier_notes = {
            "free": "Standard support response time: 24-48 hours",
            "pro": "Priority support response time: 4-8 hours",
            "enterprise": "Dedicated support response time: 1-2 hours, include escalation path"
        }
        
        urgency_notes = {
            "low": "Customer is not in a rush",
            "medium": "Customer expects response within business hours",
            "high": "Customer is blocked, needs urgent attention",
            "critical": "Production issue, highest priority"
        }
        
        return f"""Category: {context.category}
Urgency: {context.urgency} ({urgency_notes.get(context.urgency, '')})
Customer Tier: {context.customer_tier} ({tier_notes.get(context.customer_tier, '')})

Incoming Email:
{context.incoming}

Write a professional reply:"""

class MockProvider(LLMProvider):
    """Fallback provider for testing without API keys"""
    def __init__(self, dataset_path: Optional[str] = None):
        self.templates = {
            "billing": "Hi {name}, I've looked into your billing question about {topic}. I've {action} and you'll see the update in 3-5 business days. Let me know if you need anything else!",
            "technical": "Hi {name}, I'm sorry you're experiencing this issue. I've escalated this to our engineering team (ticket #{ticket}). In the meantime, you can {workaround}. We'll update you within 24 hours.",
            "feature_request": "Hi {name}, thanks for the feature request! {feature} is on our roadmap for {timeframe}. I've added your vote. We'll notify you when it's available.",
            "account": "Hi {name}, I've updated your account as requested. You'll receive a confirmation email shortly. Let me know if you need anything else!",
            "general": "Hi {name}, happy to help! {answer}. Let me know if you have any follow-up questions.",
        }
    
        self.dataset_path = dataset_path
        self.few_shot_examples = []
        self._load_examples()

    def _load_examples(self):
        if not self.dataset_path or not os.path.exists(self.dataset_path):
            return
        with open(self.dataset_path, "r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if data.get("incoming") and data.get("expected_reply"):
                    self.few_shot_examples.append({
                        "incoming": data["incoming"],
                        "expected_reply": data["expected_reply"],
                    })

    def is_available(self) -> bool:
        return True
    
    def generate(self, context: EmailContext) -> GeneratedResponse:
        import time
        import re
        start = time.time()
        
        template = self.templates.get(context.category, self.templates["general"])
        
        example_context = ""
        if self.few_shot_examples:
            example_context = "\n".join(
                f"Incoming: {item['incoming']}\nReply: {item['expected_reply']}"
                for item in self.few_shot_examples[:2]
            )

        # Extract name from incoming email
        name_match = re.search(r'(Hi|Hello|Dear)\s+(\w+)', context.incoming)
        name = name_match.group(2) if name_match else "there"
        
        # Simple template filling
        filled = template.format(
            name=name,
            topic="your inquiry",
            action="processed your request",
            ticket="12345",
            workaround="try the web version",
            feature="your requested feature",
            timeframe="Q2 2024",
            answer="I've addressed your question."
        )
        if example_context:
            filled = f"Examples:\n{example_context}\n\nCurrent request:\n{context.incoming}\n\nReply:\n{filled}"
        
        latency = (time.time() - start) * 1000
        
        return GeneratedResponse(
            response=filled,
            model="mock-v1",
            tokens_used=50,
            latency_ms=latency,
            confidence=0.5
        )

class ResponseGenerator:
    def __init__(self, preferred_provider: str = "openai", dataset_path: Optional[str] = None):
        self.providers: List[LLMProvider] = [
            OpenAIProvider(),
            AnthropicProvider(),
            MockProvider(dataset_path=dataset_path),
        ]
        self.preferred = preferred_provider
        self.few_shot_examples = self.providers[-1].few_shot_examples if self.providers else []
    
    def generate(self, context: EmailContext) -> GeneratedResponse:
        # Try preferred provider first
        for provider in self.providers:
            if provider.is_available():
                if self.preferred == "openai" and isinstance(provider, OpenAIProvider):
                    return provider.generate(context)
                elif self.preferred == "anthropic" and isinstance(provider, AnthropicProvider):
                    return provider.generate(context)
                elif self.preferred == "mock" and isinstance(provider, MockProvider):
                    return provider.generate(context)
        
        # Fallback to any available
        for provider in self.providers:
            if provider.is_available():
                return provider.generate(context)
        
        # Last resort: mock
        return MockProvider().generate(context)
    
    def batch_generate(self, contexts: List[EmailContext]) -> List[GeneratedResponse]:
        return [self.generate(ctx) for ctx in contexts]

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input JSONL file with email contexts")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL file for generated responses")
    parser.add_argument("--provider", type=str, default="openai", choices=["openai", "anthropic", "mock"])
    args = parser.parse_args()
    
    generator = ResponseGenerator(preferred_provider=args.provider)
    
    contexts = []
    with open(args.input, "r") as f:
        for line in f:
            data = json.loads(line)
            contexts.append(EmailContext(
                incoming=data["incoming"],
                category=data["category"],
                urgency=data["urgency"],
                customer_tier=data["customer_tier"]
            ))
    
    print(f"Generating responses for {len(contexts)} emails using {args.provider}...")
    responses = generator.batch_generate(contexts)
    
    with open(args.output, "w") as f:
        for i, resp in enumerate(responses):
            f.write(json.dumps({
                "id": f"gen_{i:05d}",
                "response": resp.response,
                "model": resp.model,
                "tokens_used": resp.tokens_used,
                "latency_ms": resp.latency_ms,
                "confidence": resp.confidence
            }) + "\n")
    
    print(f"Generated {len(responses)} responses -> {args.output}")

if __name__ == "__main__":
    main()
