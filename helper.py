import os
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertTokenizer, BertModel
import numpy as np
import tiktoken
from ultralytics import YOLO
from PIL import Image
import cv2
import torch


class AdvancedDynamicMemoryNetwork:
    def __init__(self):
        self.memory = []
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')

    def encode(self, text):
        """Encode text into a vector using BERT."""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        outputs = self.model(**inputs)
        # Use the mean of the last hidden state as the sentence vector
        sentence_vector = outputs.last_hidden_state.mean(dim=1).detach().numpy()
        return sentence_vector.flatten()

    def cosine_similarity(self, vec_a, vec_b):
        """Compute the cosine similarity between two vectors."""
        dot_product = np.dot(vec_a, vec_b)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        return dot_product / (norm_a * norm_b)

    def update_memory(self, objects_list, timestep):
        """Update memory with objects observed at a given timestep."""
        for obj in objects_list:
            self.memory.append({
                'timestep': timestep,
                'object': obj,
                # Store the encoded vector to avoid re-encoding later
                'encoded_vector': self.encode(obj)
            })

    def query_memory(self, target_object, similarity_threshold=0.5):
        """Query memory for objects similar to the current target object."""
        target_vector = self.encode(target_object)
        relevant_entries = []

        for entry in self.memory:
            similarity = self.cosine_similarity(target_vector, entry['encoded_vector'])
            if similarity >= similarity_threshold:
                entry['similarity'] = similarity  # Update entry with similarity
                relevant_entries.append(entry)

        return relevant_entries

    def print_memory(self):
        """Utility function to print the memory contents."""
        for entry in self.memory:
            print(f"Timestep: {entry['timestep']}, Object: {entry['object']}")


def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613"):
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


def adjust_system_prompt(system_prompt, user_prompt, model, max_tokens=4096):
    """
    Adjusts the system prompt by removing strings from the end one by one 
    until the total number of tokens (system prompt + user prompt) is less than max_tokens.
    
    Args:
    - system_prompt (str): The original system prompt.
    - user_prompt (str): The user's input.
    - model (str): The model being used, affecting tokenization.
    - max_tokens (int): The maximum allowed number of tokens.
    
    Returns:
    - list: A list of messages with the adjusted system prompt and the original user prompt.
    """
    # Split the system prompt into separate strings based on newline character
    prompt_parts = system_prompt.split("\n")
    
    while prompt_parts:
        # Join the parts back into a system prompt string
        adjusted_system_prompt = "\n".join(prompt_parts)
        # Simulate messages with the adjusted system prompt and the user prompt
        messages = [
            {"role": "system", "content": "{}".format(adjusted_system_prompt)},
            {"role": "user", "content": "{}".format(user_prompt)}
        ]

        # Assume num_tokens_from_messages is a function that accurately counts tokens
        # For this simulation, we'll assume tokens is just the sum of characters as a placeholder
        tokens = num_tokens_from_messages(messages, model=model)
        
        # Check if the total number of tokens is within the limit
        if tokens < max_tokens:
            break
        else:
            # If not, remove the last part of the system prompt and try again
            prompt_parts.pop()
    
    # Ensure the final message list contains the adjusted system prompt
    messages[0]['content'] = adjusted_system_prompt
    return messages


def gpt_run(sys_prompt, user_prompt, model="gpt-4"):

    key = 'Place key here'
    os.environ["OPENAI_API_KEY"] = key

    from openai import OpenAI

    client = OpenAI()

    messages = adjust_system_prompt(sys_prompt, user_prompt, model, max_tokens=4096)

    response = client.chat.completions.create(
      model=model,
      # response_format={ "type": "json_object" },
      messages=messages,
      temperature=0.5
     )

    response = response.choices[0].message.content

    return response

def yolo_run(image):

    model = YOLO('yolov8n.pt')  # load an official detection model

    results = model.predict(source=image, save=False, save_txt=False)  # save predictions as labels

    class_labels = []
    # Extract class labels
    for box in results[0].boxes:
        class_id = int(box.cls)  # Get class ID
        class_labels.append(results[0].names[class_id])  # Get class label from class ID
        # print(f'Detected class: {class_labels[-1]}')  # Print class label

    finlist = list(set(class_labels))

    if len(finlist) == 0: 
        return ["random objects"]
    else:
        return finlist

def yolo_run_with_boxes(image, putText=True):
    # Load and prepare the model
    model = YOLO('yolov8n.pt').cuda()  # Load your YOLO model onto the GPU

    # Load image as a tensor and move to GPU
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR (OpenCV format) to RGB
    image_tensor = torch.from_numpy(image).float().div(255).permute(2, 0, 1).unsqueeze(0).cuda()  # Convert to tensor and normalize
    
    # Run predictions
    results = model(source=image_tensor, save=False, save_txt=False)

    # Process detections
    class_labels = []
    image_np = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()  # Convert tensor back to numpy array for drawing
    image_np = (image_np * 255).astype(np.uint8)  # Denormalize and convert to uint8

    for det in results.detections:
        box = det.xyxy[0].cpu().numpy()  # Assuming det.xyxy contains the bounding box coordinates
        conf = det.conf[0].cpu().item()  # Assuming det.conf contains the confidence score
        label = results.names[int(det.cls[0].cpu().item())]  # Get class label from class ID
        class_labels.append(label)

        if putText:
            # Draw bounding box and label on the image
            start_point = (int(box[0]), int(box[1]))
            end_point = (int(box[2]), int(box[3]))
            image_np = cv2.rectangle(image_np, start_point, end_point, (0, 255, 0), 2)
            cv2.putText(image_np, f'{label}: {conf:.2f}', (start_point[0], start_point[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    finlist = list(set(class_labels))
    return cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR), finlist if finlist else ["random objects"]  # Convert back to BGR for OpenCV compatibility


def detic_run(image):

    from transformers import AutoImageProcessor, DeformableDetrForObjectDetection
    import torch
    from PIL import Image
    import requests

    processor = AutoImageProcessor.from_pretrained("SenseTime/deformable-detr-single-scale-dc5")
    model = DeformableDetrForObjectDetection.from_pretrained("SenseTime/deformable-detr-single-scale-dc5")

    height, width, _ = image.shape
    target_sizes = torch.tensor([[width, height]])  # Width and height are reversed in cv2

    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    # convert outputs (bounding boxes and class logits) to COCO API
    # let's only keep detections with score > 0.7
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.7)[0]

    objlist = []

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        print(
                f"Detected {model.config.id2label[label.item()]} with confidence "
                f"{round(score.item(), 3)} at location {box}"
        )

        objlist.append(model.config.id2label[label.item()])

    finlist = list(set(objlist))

    if len(finlist) == 0: 
        return ["random objects"]
    else:
        return finlist


def lavis_run():
    import torch
    from lavis.models import load_model_and_preprocess
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # loads BLIP caption base model, with finetuned checkpoints on MSCOCO captioning dataset.
    # this also loads the associated image processors
    model, vis_processors, _ = load_model_and_preprocess(name="blip_caption", model_type="base_coco", is_eval=True, device=device)
    # preprocess the image
    # vis_processors stores image transforms for "train" and "eval" (validation / testing / inference)
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    # generate caption
    model.generate({"image": image})
    # ['a large fountain spewing water into the air']
