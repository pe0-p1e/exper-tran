from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from primary_ml_cka.models.common.outputs import GenerationOutput
from primary_ml_cka.prompts.chat_templates import (
    classification_messages,
    render_chat_template,
)
from primary_ml_cka.prompts.parser import parse_exact_label


@dataclass(slots=True)
class TransformersTargetGenerator:
    model: object
    processor: object

    def generate_label(self, image_path: Path, prompt: str) -> GenerationOutput:
        return self.generate_labels((image_path,), prompt)[0]

    def generate_labels(
        self, image_paths: tuple[Path, ...] | list[Path], prompt: str
    ) -> tuple[GenerationOutput, ...]:
        if not image_paths:
            return ()
        messages = classification_messages(prompt).prompt_only
        rendered = render_chat_template(
            self.processor, messages, add_generation_prompt=True
        )
        images = []
        for image_path in image_paths:
            with Image.open(image_path) as image:
                images.append(image.convert("RGB"))
        # Gemma 4 interprets a flat image list as one prompt containing many
        # images. Wrap every image in its own per-example list for batched
        # single-image classification; Qwen/InternVL expect a flat batch.
        processor_images = (
            [[image] for image in images]
            if self.processor.__class__.__name__ == "Gemma4Processor"
            else images
        )
        inputs = self.processor(
            text=[rendered] * len(images),
            images=processor_images,
            return_tensors="pt",
            padding=True,
        )
        device = next(self.model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        generated = self.model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=4,
            use_cache=True,
        )
        new_tokens = generated[:, inputs["input_ids"].shape[1] :]
        decoded = self.processor.batch_decode(new_tokens, skip_special_tokens=True)
        outputs = []
        for raw in decoded:
            parsed = parse_exact_label(raw)
            outputs.append(GenerationOutput(raw, parsed.label, parsed.status, "transformers"))
        return tuple(outputs)
