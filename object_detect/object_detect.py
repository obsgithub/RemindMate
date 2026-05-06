import torch
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict, Union, Optional
import requests
from io import BytesIO
import os
from transformers import CLIPTokenizerFast
from transformers import Owlv2Processor, Owlv2ForObjectDetection

class ObjectDetector:
    def __init__(
        self,
        model_path: str = "YOUR_MODEL_PATH_HERE",  # Default set to placeholder, e.g., "google/owlv2-large-patch14-finetuned"
        device: Optional[str] = None,
        box_threshold: float = 0.4,
        text_threshold: float = 0.3  # text_threshold is temporarily unused in OwlV2, retained for compatibility
    ):
        self.device = device or ("cuda:0")
        self.processor = Owlv2Processor.from_pretrained(
            model_path,
            tokenizer_kwargs={"from_slow": True}
        )
        self.model = Owlv2ForObjectDetection.from_pretrained(model_path).to(self.device)
        self.box_threshold = box_threshold

    def _load_image(self, image_input: Union[str, Image.Image]) -> Image.Image:
        """
        Helper method to load an image from a PIL Image, URL, or local path.
        """
        if isinstance(image_input, Image.Image):
            return image_input.convert("RGB")
        if isinstance(image_input, str):
            if image_input.startswith("http://") or image_input.startswith("https://"):
                response = requests.get(image_input)
                return Image.open(BytesIO(response.content)).convert("RGB")
            if os.path.exists(image_input):
                return Image.open(image_input).convert("RGB")
        raise ValueError(f"Cannot load image: {image_input}")

    def detect(
        self,
        image_input: Union[str, Image.Image],
        labels_text: str,
        visualize: bool = False
    ) -> List[Dict]:
        """
        Detect objects in the image based on the provided text labels.
        """
        image = self._load_image(image_input)
        texts = [[label.strip() for label in labels_text.split('.') if label.strip()]]

        inputs = self.processor(text=texts, images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = torch.Tensor([image.size[::-1]]).to(self.device)
        results = self.processor.post_process_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=self.box_threshold
        )
        result = results[0]

        detections = []
        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", size=20)
        except:
            font = ImageFont.load_default()

        for box, score, label_idx in zip(result["boxes"], result["scores"], result["labels"]):
            label = texts[0][label_idx]
            score = round(score.item(), 3)
            box = [round(x, 2) for x in box.tolist()]
            detections.append({
                "label": label,
                "score": score,
                "box": box
            })

            if visualize:
                draw.rectangle(box, outline="red", width=3)
                draw.text((box[0] + 2, box[1] - 25), f"{label}: {score}", fill="red", font=font)

        if visualize:
            image.show(title="Detection Results")

        return detections


if __name__ == "__main__":
    # Example usage for demonstration purposes only.
    # Initialization
    # detector = ObjectDetector()

    # Detect objects in the image
    # text = "black Kettle. cup. scissors. pliers. umbrella. spoon. knife. watch. card. book. Mouse. razor. stapler. pen. remote control. mobile phone. car keys. sunglasses. pills. couch. keyboard. screen. chair. earphones. shaver. medicine"
    # image_path = "/path/to/your/image.jpg"

    # detections = detector.detect(image_path, text, visualize=True)
    
    # Debugging loop
    # while(1):
    #     print(1)
        
    # Output detection results
    # for det in detections:
    #     print(det)
    
    pass