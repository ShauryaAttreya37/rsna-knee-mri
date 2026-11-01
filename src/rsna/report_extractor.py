"""Multilingual clinical radiology report extraction for RSNA Knee findings."""

from __future__ import annotations

import json
import re
from typing import Any

from rsna.submission import TARGET_COLUMNS

EXTRACTION_SYSTEM_PROMPT = """You are an expert musculoskeletal radiologist specializing in knee MRI interpretation.
You will be provided a clinical knee MRI radiology report written in any language (English, Spanish, Turkish, German, French, Croatian, Greek, Bulgarian, Dutch, etc.).

Analyze the report carefully and determine the presence of 12 distinct abnormalities:
1. ACL: Anterior Cruciate Ligament tear or rupture.
2. MCL: Medial Collateral Ligament tear or sprain.
3. Medial Meniscus: Tear or disruption of the medial meniscus.
4. Lateral Meniscus: Tear or disruption of the lateral meniscus.
5. Medial OA: Osteoarthritis, cartilage loss, joint space narrowing in the medial compartment.
6. Lateral OA: Osteoarthritis, cartilage loss, joint space narrowing in the lateral compartment.
7. PF OA: Patellofemoral osteoarthritis, chondromalacia, cartilage wear under the kneecap.
8. Effusion: Joint effusion, excess intra-articular synovial fluid, water on the knee.
9. Synovitis: Synovial thickening, inflammation, or pannus.
10. Baker's: Baker's cyst, popliteal cyst.
11. Contusion: Bone marrow edema, bone bruise, trabecular microfracture.
12. Fracture: Bone fracture, cortical break, fissure.

RULES:
- NEGATION: Normal/intact/unremarkable/uninjured/no tear/no fracture/no effusion -> 0.0.
  (e.g., "intact", "unauffällig", "sin rotura", "bütünlüğü korunmuştur", "intakt", "integro") -> 0.0.
- POSITIVE: Confirmed tear, tear with retraction, grade 3 signal, cartilage denudation, bone marrow lesion -> 1.0.
- UNCERTAIN / MILD: "Suspected", "possible", "mild", "cannot be excluded" -> 0.5 to 0.7.
- NOT MENTIONED: If a condition is not mentioned and the joint is otherwise unremarkable -> 0.0.

OUTPUT FORMAT:
Respond ONLY with a JSON object mapping all 12 condition names exactly to float probabilities between 0.0 and 1.0:
{
  "ACL": 0.0,
  "MCL": 0.0,
  "Medial Meniscus": 0.0,
  "Lateral Meniscus": 0.0,
  "Medial OA": 0.0,
  "Lateral OA": 0.0,
  "PF OA": 0.0,
  "Effusion": 0.0,
  "Synovitis": 0.0,
  "Baker's": 0.0,
  "Contusion": 0.0,
  "Fracture": 0.0
}"""


def extract_json_from_text(raw_text: str) -> dict[str, float]:
    """Parse probability dict from model output."""
    raw_text = raw_text.strip()
    match = re.search(r"\{[^{}]*\}", raw_text, re.DOTALL)
    if not match:
        return {col: 0.0 for col in TARGET_COLUMNS}

    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {col: 0.0 for col in TARGET_COLUMNS}

    result: dict[str, float] = {}
    for col in TARGET_COLUMNS:
        val = data.get(col, 0.0)
        try:
            val_float = float(val)
        except (ValueError, TypeError):
            val_float = 0.0
        result[col] = max(0.0, min(1.0, val_float))
    return result


class ClinicalReportExtractor:
    """Multilingual report pseudo-label extractor using local Qwen2.5."""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-3B-Instruct",
        device: str = "cuda",
        load_in_4bit: bool = True,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        bnb_config = None
        if load_in_4bit and device == "cuda":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto" if device == "cuda" else None,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        )
        self.model.eval()

    def build_prompt(self, report_text: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Radiology Report:\n```text\n{report_text.strip()}\n```\n\nReturn JSON:",
            },
        ]

    def extract_batch(
        self, report_texts: list[str], max_new_tokens: int = 160
    ) -> list[dict[str, float]]:
        import torch

        if not report_texts:
            return []

        prompts = [
            self.tokenizer.apply_chat_template(
                self.build_prompt(text), tokenize=False, add_generation_prompt=True
            )
            for text in report_texts
        ]

        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        prompt_len = inputs["input_ids"].shape[1]
        results: list[dict[str, float]] = []
        for i in range(len(report_texts)):
            generated_ids = outputs[i][prompt_len:]
            response_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            results.append(extract_json_from_text(response_text))
        return results

    def extract_one(self, report_text: str) -> dict[str, float]:
        return self.extract_batch([report_text])[0]
