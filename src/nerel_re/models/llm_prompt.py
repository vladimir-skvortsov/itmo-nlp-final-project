"""Few-shot relation extraction using a local LLM via MLX or subprocess."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass

from nerel_re.data.dataset import NO_RELATION, RelationExample

_SYSTEM_PROMPT = """\
You are a named entity relation classifier for Russian text.
Given a sentence with two highlighted entities, output ONLY the relation type
from the allowed list, or "no_relation" if no relation holds.
Respond with a single JSON object: {"relation": "<type>"}.
"""

_FEW_SHOT_TEMPLATE = """\
Sentence: {sentence}
Subject entity ({subj_type}): {subj_text}
Object entity ({obj_type}): {obj_text}
Answer:"""


@dataclass
class LLMConfig:
    """Configuration for LLM-based relation extraction.

    Attributes:
        model_path: Local MLX model path or HuggingFace repo ID.
        allowed_relations: List of valid relation type strings.
        n_few_shot: Number of few-shot examples to prepend.
        max_new_tokens: Maximum tokens to generate per prediction.
        temperature: Sampling temperature (lower = more deterministic).
    """

    model_path: str
    allowed_relations: list[str]
    n_few_shot: int = 5
    max_new_tokens: int = 64
    temperature: float = 0.0


class LLMRelationExtractor:
    """Few-shot relation extractor using a local LLM (MLX backend).

    Args:
        config: :class:`LLMConfig` with model and inference settings.
        few_shot_examples: Examples used to build the few-shot prompt.
            Should be drawn from the training split.
    """

    def __init__(
        self,
        config: LLMConfig,
        few_shot_examples: list[RelationExample] | None = None,
    ) -> None:
        """Initialise with config and optional few-shot examples."""
        self.config = config
        self.few_shot_examples = few_shot_examples or []

    def predict(self, example: RelationExample) -> str:
        """Predict the relation type for a single example.

        Args:
            example: The relation extraction example to classify.

        Returns:
            Predicted relation type string, or ``NO_RELATION`` on parse failure.
        """
        prompt = self._build_prompt(example)
        raw = self._call_mlx(prompt)
        return self._parse_response(raw)

    def predict_batch(self, examples: list[RelationExample]) -> list[str]:
        """Predict relation types for a list of examples.

        Args:
            examples: List of examples to classify.

        Returns:
            List of predicted relation type strings.
        """
        return [self.predict(ex) for ex in examples]

    def _build_prompt(self, example: RelationExample) -> str:
        """Construct the full few-shot prompt for an example.

        Args:
            example: Target example.

        Returns:
            Full prompt string including system instructions and few-shot demos.
        """
        parts = [_SYSTEM_PROMPT, '\n']
        parts.append(f'Allowed relations: {json.dumps(self.config.allowed_relations)}\n\n')

        for demo in self.few_shot_examples[: self.config.n_few_shot]:
            parts.append(
                _FEW_SHOT_TEMPLATE.format(
                    sentence=demo.sentence,
                    subj_type=demo.entity1.type,
                    subj_text=demo.entity1.text,
                    obj_type=demo.entity2.type,
                    obj_text=demo.entity2.text,
                )
            )
            parts.append(f' {{"relation": "{demo.relation}"}}\n\n')

        parts.append(
            _FEW_SHOT_TEMPLATE.format(
                sentence=example.sentence,
                subj_type=example.entity1.type,
                subj_text=example.entity1.text,
                obj_type=example.entity2.type,
                obj_text=example.entity2.text,
            )
        )
        return ''.join(parts)

    def _call_mlx(self, prompt: str) -> str:
        """Invoke the MLX CLI to generate a response.

        Args:
            prompt: Full prompt string.

        Returns:
            Raw text output from the model.
        """
        cmd = [
            'mlx_lm.generate',
            '--model',
            self.config.model_path,
            '--prompt',
            prompt,
            '--max-tokens',
            str(self.config.max_new_tokens),
            '--temp',
            str(self.config.temperature),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)  # noqa: S603
        return result.stdout.strip()

    @staticmethod
    def _parse_response(raw: str) -> str:
        """Parse a JSON relation prediction from model output.

        Args:
            raw: Raw model output string.

        Returns:
            Relation type string, or ``NO_RELATION`` if parsing fails.
        """
        try:
            # Find the first JSON object in the output.
            start = raw.index('{')
            end = raw.rindex('}') + 1
            data = json.loads(raw[start:end])
            return str(data.get('relation', NO_RELATION))
        except ValueError, KeyError, json.JSONDecodeError:
            return NO_RELATION
