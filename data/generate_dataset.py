#!/usr/bin/env python3
"""
Generate synthetic email dataset for training/evaluation.
Creates realistic customer support email pairs (incoming -> reply).
"""
import json
import random
from dataclasses import dataclass, asdict
from typing import List

FIRST_NAMES = [
    "Alice", "Bob", "Charlie", "Diana", "Edward", "Fiona", "George",
    "Hannah", "Ivan", "Julia", "Kevin", "Laura", "Michael", "Nina",
    "Oliver", "Patricia", "Quinn", "Rachel", "Samuel", "Tara",
    "Uma", "Victor", "Wendy", "Xavier", "Yara", "Zach",
    "Abigail", "Benjamin", "Claire", "Daniel", "Emma", "Frank",
]

LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia",
    "Miller", "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez",
    "Gonzalez", "Wilson", "Anderson", "Thomas", "Taylor", "Moore",
    "Jackson", "Martin", "Lee", "Perez", "Thompson", "White",
]

DOMAINS = ["gmail.com", "outlook.com", "company.com", "hotmail.com", "yahoo.com", "protonmail.com"]

MONTHS = ["January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]

def random_name():
    return random.choice(FIRST_NAMES)

def random_email():
    return f"{random_name().lower()}.{random.choice(LAST_NAMES).lower()}{random.randint(1,99)}@{random.choice(DOMAINS)}"

def random_date():
    month = random.choice(MONTHS)
    day = random.randint(1, 28)
    year = random.randint(2023, 2024)
    return f"{month} {day}, {year}"

@dataclass
class EmailPair:
    id: str
    incoming: str
    expected_reply: str
    category: str
    urgency: str
    customer_tier: str

# Templates for different categories
TEMPLATES = {
    "billing": {
        "incoming": [
            "Hi, I was charged ${amount} on {date} but I don't recognize this charge. Can you help?",
            "My subscription renewed at ${amount} but I wanted to cancel. Can I get a refund?",
            "I see a duplicate charge of ${amount} on my account. Please investigate.",
            "Why was I charged ${amount} for {service}? I thought it was included in my plan.",
            "I need an invoice for the ${amount} charge on {date} for my records.",
        ],
        "replies": [
            "Hi {name}, I've looked into the ${amount} charge from {date}. This was for {service}. I've {action} and you'll see the update in 3-5 business days. Let me know if you need anything else!",
            "Hello {name}, I apologize for the confusion around the ${amount} charge. This was your {plan} subscription renewal. I've {action} as a one-time courtesy. Your subscription is now {status}.",
            "Hi {name}, I found the duplicate ${amount} charge and have {action}. The refund will process in 5-7 business days. You'll receive a confirmation email once complete.",
        ]
    },
    "technical": {
        "incoming": [
            "I can't log into my account. I keep getting '{error}' error.",
            "The {feature} feature isn't working. When I try to {action}, nothing happens.",
            "I'm getting a '{error}' error when trying to {action}. This started {timeframe}.",
            "My {integration} integration stopped syncing. Last successful sync was {date}.",
            "The mobile app crashes when I {action}. I'm on {platform} version {version}.",
        ],
        "replies": [
            "Hi {name}, I'm sorry you're experiencing the '{error}' error. This is a known issue we're actively fixing. As a workaround, you can {workaround}. We'll notify you when it's resolved.",
            "Hello {name}, thanks for reporting the {feature} issue. I've escalated this to our engineering team (ticket #{ticket}). In the meantime, {workaround}. We'll update you within 24 hours.",
            "Hi {name}, I've investigated the '{error}' error when {action}. This appears to be related to {cause}. I've {action_taken} on your account. Please try again and let me know if it persists.",
        ]
    },
    "feature_request": {
        "incoming": [
            "It would be great if you could add {feature}. This would help me {benefit}.",
            "Is there a way to {action}? I need this for {use_case}.",
            "I'd love to see {feature} in a future release. Currently I have to {workaround}.",
            "Can you add support for {integration}? My team uses it extensively.",
            "Feature request: {feature}. This is a blocker for {use_case}.",
        ],
        "replies": [
            "Hi {name}, thanks for the feature request! {feature} is actually on our roadmap for {timeframe}. I've added your vote to the feature request. We'll notify you when it's available.",
            "Hello {name}, great suggestion! While {feature} isn't available yet, you can currently {workaround}. I've logged this request for our product team to review.",
            "Hi {name}, appreciate the feedback on {feature}. This is something we're exploring for {timeframe}. In the meantime, {workaround}. I'll keep you posted on progress.",
        ]
    },
    "account": {
        "incoming": [
            "I need to update my email address from {old_email} to {new_email}.",
            "How do I delete my account? I want to remove all my data.",
            "I forgot my password and the reset email isn't arriving.",
            "Can you merge my two accounts? I have {email1} and {email2}.",
            "I need to change my plan from {old_plan} to {new_plan}.",
        ],
        "replies": [
            "Hi {name}, I've updated your email from {old_email} to {new_email}. You'll receive a confirmation at the new address. Let me know if you need anything else!",
            "Hello {name}, I can help you delete your account. This action is irreversible and will remove all data. Please confirm you want to proceed, and I'll process it immediately.",
            "Hi {name}, I've triggered a password reset to {email}. If you don't see it in 5 minutes, check spam or let me know and I'll send a magic link instead.",
        ]
    },
    "general": {
        "incoming": [
            "Hi, I have a question about {topic}. Can you help?",
            "What's the status of my request #{ticket}?",
            "I'd like to speak to a human agent about {topic}.",
            "Thanks for your help! Just wanted to say great service.",
            "Is there a phone number I can call for support?",
        ],
        "replies": [
            "Hi {name}, happy to help with {topic}! {answer}. Let me know if you have any follow-up questions.",
            "Hello {name}, your request #{ticket} is currently {status}. We expect to resolve it by {timeframe}. I'll update you as soon as there's progress.",
            "Hi {name}, I'm here to help! I can assist with {topic} right here. What specific questions do you have?",
        ]
    }
}

CATEGORIES = list(TEMPLATES.keys())
URGENCIES = ["low", "medium", "high", "critical"]
TIERS = ["free", "pro", "enterprise"]

def generate_email_pair(idx: int) -> EmailPair:
    category = random.choice(CATEGORIES)
    templates = TEMPLATES[category]
    
    incoming_template = random.choice(templates["incoming"])
    reply_template = random.choice(templates["replies"])
    
    # Fill in template variables
    variables = {
        "name": random_name(),
        "amount": f"${random.randint(10, 500)}.{random.randint(10, 99):02d}",
        "date": random_date(),
        "service": random.choice(["Premium Plan", "Add-on Storage", "API Access", "Priority Support", "Custom Domain"]),
        "plan": random.choice(["Monthly Pro", "Annual Business", "Enterprise"]),
        "error": random.choice(["Invalid credentials", "Session expired", "Rate limit exceeded", "Network error", "Permission denied"]),
        "feature": random.choice(["Dark mode", "Bulk export", "Custom workflows", "Advanced analytics", "SSO integration"]),
        "action": random.choice(["export data", "invite team members", "create automation", "generate report", "sync contacts"]),
        "timeframe": random.choice(["last week", "yesterday", "this morning", "2 days ago"]),
        "integration": random.choice(["Slack", "Salesforce", "HubSpot", "Jira", "Asana"]),
        "platform": random.choice(["iOS", "Android"]),
        "version": f"{random.randint(1,5)}.{random.randint(0,9)}.{random.randint(0,9)}",
        "workaround": random.choice(["use the web app", "clear cache and retry", "reinstall the app", "use a different browser"]),
        "cause": random.choice(["a recent update", "server maintenance", "a configuration issue", "an API change"]),
        "action_taken": random.choice(["cleared the cache", "reset your session", "updated your permissions", "refreshed the connection"]),
        "benefit": random.choice(["save time", "automate workflows", "improve team collaboration", "get better insights"]),
        "use_case": random.choice(["client onboarding", "weekly reporting", "team collaboration", "data analysis"]),
        "old_email": random_email(),
        "new_email": random_email(),
        "email1": random_email(),
        "email2": random_email(),
        "old_plan": random.choice(["Free", "Pro"]),
        "new_plan": random.choice(["Pro", "Business", "Enterprise"]),
        "topic": random.choice(["pricing", "features", "integrations", "security", "compliance"]),
        "ticket": f"#{random.randint(10000, 99999)}",
        "status": random.choice(["in progress", "under review", "waiting for your reply", "escalated"]),
        "answer": random.choice([
            "Our pricing starts at $15/month for the Pro plan.",
            "Yes, we support that integration via our API.",
            "All data is encrypted at rest and in transit.",
            "We're SOC2 Type II certified and GDPR compliant."
        ]),
    }
    
    incoming = incoming_template.format(**variables)
    expected_reply = reply_template.format(**variables)
    
    return EmailPair(
        id=f"email_{idx:05d}",
        incoming=incoming,
        expected_reply=expected_reply,
        category=category,
        urgency=random.choice(URGENCIES),
        customer_tier=random.choice(TIERS)
    )

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=500, help="Number of email pairs to generate")
    parser.add_argument("--output", type=str, default="data/email_dataset.jsonl", help="Output file path")
    parser.add_argument("--split", type=float, default=0.8, help="Train/test split ratio")
    args = parser.parse_args()
    
    print(f"Generating {args.count} email pairs...")
    pairs = [generate_email_pair(i) for i in range(args.count)]
    
    # Split train/test
    split_idx = int(args.count * args.split)
    train_pairs = pairs[:split_idx]
    test_pairs = pairs[split_idx:]
    
    # Write train
    with open(args.output.replace(".jsonl", "_train.jsonl"), "w") as f:
        for pair in train_pairs:
            f.write(json.dumps(asdict(pair)) + "\n")
    
    # Write test
    with open(args.output.replace(".jsonl", "_test.jsonl"), "w") as f:
        for pair in test_pairs:
            f.write(json.dumps(asdict(pair)) + "\n")
    
    print(f"Generated {len(train_pairs)} train + {len(test_pairs)} test pairs")
    print(f"Train: {args.output.replace('.jsonl', '_train.jsonl')}")
    print(f"Test: {args.output.replace('.jsonl', '_test.jsonl')}")

if __name__ == "__main__":
    main()
