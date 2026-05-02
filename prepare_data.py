from __future__ import annotations
import argparse
import json
import random
import re
import uuid
from abc import ABC, abstractmethod
from typing import Any
import yaml
from faker import Faker
import sys
import time
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from settings import settings
from RAG.ingestion import load_corpus, build_index, load_existing_index
from RAG.retriever import EnterpriseRetriever
from RAG.vector_store import get_chroma_client

INJECTION_METHODS = ("append", "middle", "html_comment")

_SLACK_DEPARTMENTS = [
    "Engineering", "Platform", "Security", "Data Science",
    "DevOps", "Product", "SRE", "Infrastructure", "QA", "ML",
]

_EMAIL_DEPARTMENTS = [
    "Engineering", "Platform", "Security", "Data Science",
    "DevOps", "Product", "SRE", "HR", "Finance", "Legal",
]

_INTERNAL_DOCS_DEPARTMENTS = [
    "Engineering", "Platform", "Security", "Data Science",
    "DevOps", "SRE", "Infrastructure", "Architecture",
]

class DataSource(ABC):

    @abstractmethod
    def generate(self, count):
        ...

    @property
    @abstractmethod
    def source_type(self):
        ...

class TemplateDataSource(DataSource):
    template_filename: str = ""

    def __init__(self, seed = 42):
        self._seed = seed
        self._fake = Faker()
        Faker.seed(seed)
        random.seed(seed)
        self._templates = self._load_templates()

    def generate(self, count):
        return [self._generate_one() for _ in range(count)]

    def _load_templates(self):
        path = settings.data_seed_dir / self.template_filename
        with open(path) as f:
            return json.load(f)

    def _pick_access_level(self):
        return random.choice(self._templates["access_levels"])

    def _pick_trust_score(self):
        lo, hi = self._templates["trust_score_range"]
        return round(random.uniform(lo, hi), 3)

    @abstractmethod
    def _generate_one(self):
        ...

class ConfluenceGenerator(TemplateDataSource):
    template_filename = "confluence_pages.json"

    @property
    def source_type(self):
        return "confluence"

    def _fill(self, text, department, author):
        replacements = {
            "{{department}}": department,
            "{{author}}": author,
            "{{team_name}}": department,
            "{{service_name}}": self._fake.bs().split()[0].title() + "Service",
            "{{tool_name}}": random.choice(["Terraform", "Kubernetes", "Docker", "Vault", "Prometheus"]),
            "{{version}}": f"{random.randint(1,3)}.{random.randint(0,9)}",
            "{{owner_name}}": author,
            "{{date}}": self._fake.date_this_year().isoformat(),
            "{{step_description}}": self._fake.sentence(),
            "{{overview}}": self._fake.paragraph(nb_sentences=4),
            "{{details}}": self._fake.paragraph(nb_sentences=5),
            "{{guideline}}": self._fake.sentence(),
            "{{policy_text}}": self._fake.paragraph(nb_sentences=3),
        }
        for key, value in replacements.items():
            text = text.replace(key, value)
        return re.sub(r"\{\{[^}]+\}\}", lambda _: self._fake.sentence(), text)

    def _generate_one(self):
        department = random.choice(self._templates["departments"])
        topic = random.choice(self._templates["topics"])
        author = self._fake.name()
        template = random.choice(self._templates["templates"])
        title = template["title_template"].replace("{{topic}}", topic).replace("{{department}}", department)
        content_template = random.choice(template["content_templates"])
        content = self._fill(content_template, department, author)
        return {
            "id": f"conf-{uuid.uuid4().hex[:12]}",
            "title": title,
            "content": f"# {title}\n\n{content}",
            "source_type": self.source_type,
            "author": author,
            "department": department,
            "access_level": self._pick_access_level(),
            "trust_score": self._pick_trust_score(),
            "created_at": self._fake.iso8601(),
            "metadata": {
                "space": f"{department.lower().replace(' ', '-')}-space",
                "url": f"https://acme.atlassian.net/wiki/spaces/{department[:3].upper()}/pages/{random.randint(100000, 999999)}",
                "labels": [topic.lower().replace(" ", "-"), department.lower()],
                "version": random.randint(1, 15),
            },
        }

class SlackGenerator(TemplateDataSource):
    template_filename = "slack_messages.json"
    @property
    def source_type(self):
        return "slack"

    def _fill(self, text, author):
        replacements = {
            "{{author}}": author,
            "{{user_name}}": self._fake.first_name(),
            "{{tool_name}}": random.choice(["Terraform", "Kubernetes", "Vault", "Datadog", "PagerDuty"]),
            "{{service_name}}": self._fake.bs().split()[0].title() + "Service",
            "{{environment}}": random.choice(["prod", "staging", "dev", "canary"]),
            "{{pr_number}}": str(random.randint(1000, 9999)),
            "{{issue_number}}": f"JIRA-{random.randint(1000, 9999)}",
            "{{meeting_type}}": random.choice(["standup", "sprint retro", "arch review", "1-on-1"]),
            "{{date}}": self._fake.date_this_month().isoformat(),
            "{{time}}": self._fake.time(),
            "{{message}}": self._fake.sentence(),
            "{{update}}": self._fake.sentence(),
            "{{topic}}": self._fake.bs(),
        }
        for key, value in replacements.items():
            text = text.replace(key, value)
        return re.sub(r"\{\{[^}]+\}\}", lambda _: self._fake.word(), text)

    def _generate_one(self):
        channel = random.choice(self._templates["channels"])
        author = self._fake.name()
        department = random.choice(_SLACK_DEPARTMENTS)
        template_group = random.choice(self._templates["templates"])
        message_template = random.choice(template_group["message_templates"])
        content = self._fill(message_template, author)
        if random.random() < 0.3:
            reply_author = self._fake.first_name()
            reply = random.choice([
                f"  > {reply_author}: +1, same here",
                f"  > {reply_author}: thanks, that fixed it!",
                f"  > {reply_author}: {self._fake.sentence()}",
                f"  > {reply_author}: linking the ticket — JIRA-{random.randint(1000, 9999)}",
            ])
            content = f"{content}\n{reply}"
        return {
            "id": f"slack-{uuid.uuid4().hex[:12]}",
            "title": content.split("\n")[0][:80],
            "content": content,
            "source_type": self.source_type,
            "author": author,
            "department": department,
            "access_level": self._pick_access_level(),
            "trust_score": self._pick_trust_score(),
            "created_at": self._fake.iso8601(),
            "metadata": {
                "channel": f"#{channel}",
                "thread_ts": f"{random.randint(1_700_000_000, 1_710_000_000)}.{random.randint(100000, 999999)}",
                "reactions": random.randint(0, 12),
                "reply_count": random.randint(0, 25),
            },
        }

class EmailGenerator(TemplateDataSource):
    template_filename = "emails.json"

    @property
    def source_type(self):
        return "email"

    def _fill(self, text, sender, recipient, department):
        replacements = {
            "{{sender_name}}": sender,
            "{{recipient_name}}": recipient,
            "{{department}}": department,
            "{{team_name}}": department,
            "{{quarter}}": random.choice(["Q1 2025", "Q2 2025", "Q3 2025", "Q4 2024"]),
            "{{meeting_topic}}": self._fake.bs(),
            "{{meeting_type}}": random.choice(["all-hands", "sprint review", "1-on-1", "QBR"]),
            "{{date}}": self._fake.date_this_year().isoformat(),
            "{{time}}": self._fake.time(),
            "{{project_name}}": self._fake.bs().title(),
            "{{feature_name}}": self._fake.bs().title(),
            "{{service_name}}": self._fake.bs().split()[0].title() + "Service",
            "{{policy_name}}": self._fake.bs().title() + " Policy",
            "{{deadline}}": self._fake.date_between("+1d", "+30d").isoformat(),
            "{{action_item}}": self._fake.sentence(),
            "{{owner}}": self._fake.first_name(),
            "{{metric}}": f"{random.uniform(95.0, 99.99):.2f}%",
            "{{budget_amount}}": f"${random.randint(10, 500)}k",
            "{{paragraph}}": self._fake.paragraph(nb_sentences=4),
            "{{sentence}}": self._fake.sentence(),
            "{{update}}": self._fake.paragraph(nb_sentences=3),
        }
        for key, value in replacements.items():
            text = text.replace(key, value)
        return re.sub(r"\{\{[^}]+\}\}", lambda _: self._fake.sentence(), text)

    def _generate_one(self):
        template = random.choice(self._templates["templates"])
        sender = self._fake.name()
        recipient = self._fake.name()
        department = random.choice(_EMAIL_DEPARTMENTS)
        subject = re.sub(
            r"\{\{[^}]+\}\}",
            lambda _: self._fake.bs().title(),
            template["subject_template"],
        )
        body = self._fill(random.choice(template["body_templates"]), sender, recipient, department)
        return {
            "id": f"email-{uuid.uuid4().hex[:12]}",
            "title": subject,
            "content": f"Subject: {subject}\n\n{body}",
            "source_type": self.source_type,
            "author": sender,
            "department": department,
            "access_level": self._pick_access_level(),
            "trust_score": self._pick_trust_score(),
            "created_at": self._fake.iso8601(),
            "metadata": {
                "from": f"{sender.lower().replace(' ', '.')}@acme.com",
                "to": [f"{recipient.lower().replace(' ', '.')}@acme.com"],
                "cc": (
                    [f"{self._fake.name().lower().replace(' ', '.')}@acme.com"]
                    if random.random() < 0.4
                    else []
                ),
                "has_attachment": random.random() < 0.2,
                "thread_id": f"thread-{uuid.uuid4().hex[:8]}",
            },
        }

class InternalDocsGenerator(TemplateDataSource):
    template_filename = "internal_docs.json"

    @property
    def source_type(self):
        return "internal_docs"

    def _fill(self, text, author, department):
        replacements = {
            "{{owner_name}}": author,
            "{{author}}": author,
            "{{department}}": department,
            "{{team_name}}": department,
            "{{service_name}}": self._fake.bs().split()[0].title() + "Service",
            "{{version}}": f"{random.randint(1,3)}.{random.randint(0,9)}",
            "{{environment}}": random.choice(["staging", "production", "canary", "dev"]),
            "{{date}}": self._fake.date_this_year().isoformat(),
            "{{updated_at}}": self._fake.date_this_year().isoformat(),
            "{{status}}": random.choice(["Approved", "In Review", "Draft", "Deprecated"]),
            "{{rfc_status}}": random.choice(["proposed", "accepted", "rejected", "implemented"]),
            "{{endpoint}}": f"https://api.acme.com/v{random.randint(1,3)}/{self._fake.slug()}",
            "{{config_key}}": random.choice(["LOG_LEVEL", "MAX_RETRIES", "TIMEOUT_MS", "CACHE_TTL"]),
            "{{config_value}}": random.choice(["info", "3", "5000", "300"]),
            "{{paragraph}}": self._fake.paragraph(nb_sentences=5),
            "{{technical_paragraph}}": self._fake.paragraph(nb_sentences=6),
            "{{step}}": self._fake.sentence(),
            "{{risk}}": self._fake.sentence(),
            "{{metric}}": f"P99 latency: {random.randint(50, 500)}ms",
            "{{employee_name}}": self._fake.name(),
            "{{salary}}": f"${random.randint(80, 250)}k",
            "{{ssn}}": f"{random.randint(100,999)}-{random.randint(10,99)}-{random.randint(1000,9999)}",
        }
        for key, value in replacements.items():
            text = text.replace(key, value)
        return re.sub(r"\{\{[^}]+\}\}", lambda _: self._fake.sentence(), text)

    def _generate_one(self):
        template = random.choice(self._templates["templates"])
        author = self._fake.name()
        department = random.choice(_INTERNAL_DOCS_DEPARTMENTS)
        doc_type = template.get("doc_type", random.choice(self._templates["doc_types"]))
        title = re.sub(
            r"\{\{[^}]+\}\}",
            lambda _: self._fake.bs().title(),
            template["title_template"],
        )
        content = self._fill(random.choice(template["content_templates"]), author, department)
        return {
            "id": f"doc-{uuid.uuid4().hex[:12]}",
            "title": title,
            "content": content,
            "source_type": self.source_type,
            "author": author,
            "department": department,
            "access_level": self._pick_access_level(),
            "trust_score": self._pick_trust_score(),
            "created_at": self._fake.iso8601(),
            "metadata": {
                "doc_type": doc_type,
                "version": f"v{random.randint(1, 5)}.{random.randint(0, 9)}",
                "review_status": random.choice(["approved", "in_review", "draft"]),
                "page_count": random.randint(2, 20),
                "tags": [doc_type.lower().replace(" ", "-"), department.lower()],
            },
        }

class PayloadInjector:
    def __init__(self, attacks_dir = None, num_poisoned_per_type = None, seed = 42):
        self._attacks_dir = attacks_dir or settings.data_attacks_dir
        self._num_per_type = num_poisoned_per_type or settings.num_poisoned_per_type
        self._seed = seed
        random.seed(seed)
        self._attack_defs = self._load_attack_definitions()

    def inject(self, docs):
        manifest = []
        modified = list(docs)
        for attack_def in self._attack_defs:
            attack_type = attack_def["attack_type"]
            variants = attack_def["variants"]
            target_sources = {
                source
                for variant in variants
                for source in variant.get("target_sources", [])
            }
            eligible_indices = [
                index
                for index, doc in enumerate(modified)
                if doc["source_type"] in target_sources
                and not any(entry["doc_id"] == doc["id"] for entry in manifest)
            ]
            if not eligible_indices:
                continue
            chosen_indices = random.sample(
                eligible_indices,
                min(self._num_per_type, len(eligible_indices)),
            )
            for index in chosen_indices:
                doc = dict(modified[index])
                variant = random.choice(variants)
                method = random.choice(INJECTION_METHODS)
                doc["content"] = self._inject_payload(doc["content"], variant["payload"], method)
                doc["id"] = f"{doc['id']}-poisoned"
                modified[index] = doc
                manifest.append({
                    "doc_id": doc["id"],
                    "attack_type": attack_type,
                    "variant_id": variant["id"],
                    "variant_name": variant["name"],
                    "injection_method": method,
                    "original_source_type": doc["source_type"],
                    "success_patterns": variant.get("success_patterns", []),
                })
        return modified, manifest

    @property
    def attack_types(self):
        return [attack_def["attack_type"] for attack_def in self._attack_defs]

    def _load_attack_definitions(self):
        attack_defs = []
        for yaml_file in sorted(Path(self._attacks_dir).glob("*.yaml")):
            with open(yaml_file) as f:
                data = yaml.safe_load(f)
            if data and "variants" in data:
                attack_defs.append(data)
        return attack_defs

    @staticmethod
    def _inject_payload(content, payload, method):
        payload = payload.strip()
        if method == "append":
            return f"{content}\n\n{payload}"
        if method == "html_comment":
            return f"{content}\n\n<!-- {payload} -->"
        if method == "middle":
            lines = content.split("\n")
            lines.insert(len(lines) // 2, f"\n{payload}\n")
            return "\n".join(lines)
        return f"{content}\n\n{payload}"

def parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic enterprise data")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--num-docs", type=int, default=None, help="Total clean docs to generate (default: from settings)")
    parser.add_argument("--corpus", type=Path, default=None, help="Path to corpus.json (default: data/generated/corpus.json)")
    parser.add_argument("--test-query", type=str, default="What is the company vacation policy?", help="Test query to run after ingestion.")
    return parser.parse_args()

def main():
    args = parse_args()
    seed = args.seed
    total = args.num_docs or settings.num_clean_docs
    n_confluence = int(total * 0.30)
    n_slack = int(total * 0.30)
    n_email = int(total * 0.20)
    n_internal = total - n_confluence - n_slack - n_email  # remainder
    print(f"Generating {total} clean documents (seed={seed}) ...")
    print(f"\t- Confluence: {n_confluence}")
    print(f"\t- Slack: {n_slack}")
    print(f"\t- Email: {n_email}")
    print(f"\t- Internal: {n_internal}")
    confluence_gen = ConfluenceGenerator(seed=seed)
    slack_gen = SlackGenerator(seed=seed + 1)
    email_gen = EmailGenerator(seed=seed + 2)
    internal_gen = InternalDocsGenerator(seed=seed + 3)
    docs = []
    docs.extend(confluence_gen.generate(n_confluence))
    docs.extend(slack_gen.generate(n_slack))
    docs.extend(email_gen.generate(n_email))
    docs.extend(internal_gen.generate(n_internal))
    print(f"\nGenerated {len(docs)} clean documents.")
    injector = PayloadInjector(seed=seed + 100)
    print(f"\nLoaded attack types: {injector.attack_types}")
    print(f"Poisoning {settings.num_poisoned_per_type} docs per attack type ...")
    corpus, manifest = injector.inject(docs)
    poisoned_count = len(manifest)
    print(f"\nPoisoned {poisoned_count} documents total.")
    attack_counts = Counter(m["attack_type"] for m in manifest)
    for atype, cnt in sorted(attack_counts.items()):
        print(f"{atype}: {cnt}")
    method_counts = Counter(m["injection_method"] for m in manifest)
    print("\nInjection methods used:")
    for method, cnt in sorted(method_counts.items()):
        print(f"{method}: {cnt}")
    source_counts = Counter(m["original_source_type"] for m in manifest)
    print("\nPoisoned source types:")
    for src, cnt in sorted(source_counts.items()):
        print(f"{src}: {cnt}")
    output_dir = Path(settings.data_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = output_dir / "corpus.json"
    manifest_path = output_dir / "manifest.json"
    with open(corpus_path, "w") as f:
        json.dump(corpus, f, indent=2, default=str)
    print(f"\nCorpus saved to {corpus_path}  ({len(corpus)} documents)")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest saved to {manifest_path}  ({len(manifest)} entries)")
    total_sources = Counter(d["source_type"] for d in corpus)
    total_access = Counter(d["access_level"] for d in corpus)
    print("\n--- Corpus Summary ---")
    print(f"Total documents: {len(corpus)}")
    print(f"  Clean: {len(corpus) - poisoned_count}")
    print(f"  Poisoned: {poisoned_count}")
    print("\nBy source type:")
    for src, cnt in sorted(total_sources.items()):
        print(f"{src}: {cnt}")
    print("\nBy access level:")
    for level, cnt in sorted(total_access.items()):
        print(f"{level}: {cnt}")
    avg_trust = sum(d["trust_score"] for d in corpus) / len(corpus)
    print(f"\nAverage trust score: {avg_trust:.3f}")
    corpus_path = args.corpus or (settings.data_output_dir / "corpus.json")
    settings.ensure_dirs()
    print(f"Loading corpus from {corpus_path} ...")
    documents = load_corpus(corpus_path)
    print(f"Loaded {len(documents)} documents.")
    print("Building index (chunking, embedding, storing in ChromaDB) ...")
    t0 = time.perf_counter()
    index = build_index(documents=documents)
    elapsed = time.perf_counter() - t0
    print(f"Index built in {elapsed}s.")
    client = get_chroma_client()
    collection = client.get_or_create_collection("enterprise_docs")
    num_chunks = collection.count()
    print(f"ChromaDB collection 'enterprise_docs' contains {num_chunks} chunks.")
    print(f"\nRunning test query: \"{args.test_query}\"")
    retriever = EnterpriseRetriever(index)
    results = retriever.retrieve(args.test_query)
    print(f"Retrieved {len(results)} nodes:\n")
    for i, nws in enumerate(results, start=1):
        meta = nws.node.metadata or {}
        score = f"{nws.score}" if nws.score is not None else "N/A"
        snippet = nws.node.get_content()[:120].replace("\n", " ")
        print(f"[{i}] score={score} | source={meta.get('source_type', '?')} | title={meta.get('title', '?')}")
        print(f"{snippet}...\n")
    print("\nData prepared.")

if __name__ == "__main__":
    main()
