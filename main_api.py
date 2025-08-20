from openai import OpenAI
import base64
from io import BytesIO
from PIL import Image
import json
import os
import glob
import argparse
from tqdm import tqdm

# Set OpenAI's API key and API base to use vLLM's API server.
def encode_image(image: Image.Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def build_message_with_image_and_text(text_prompt: str, image_b64: str):
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_b64}"
                    }
                },
                {"type": "text", "text": text_prompt},
            ],
        }
    ]

def main():
    parser = argparse.ArgumentParser(description="Batch process images using OpenAI API.")
    parser.add_argument('--image_dir', type=str, required=True, help="Directory containing images to process.")
    parser.add_argument('--result_dir', type=str, required=True, help="Directory to save JSON results.")
    parser.add_argument('--prompt_path', type=str, required=True, help="Path to the text file containing the prompt.")
    parser.add_argument('--model_name', type=str, default="gemini-2.5-pro-exp-03-25", help="Name of the model to use.")

    args = parser.parse_args()

    openai_api_key = "your-openai-key-here"  # Hardcoded API key
    openai_api_base = "Openai_Website"  # Hardcoded API base URL
    model_name = args.model_name
    image_dir = args.image_dir
    result_dir = args.result_dir
    prompt_path = args.prompt_path

    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    try:
        with open(prompt_path, 'r', encoding='utf-8') as file:
            prompt_content = file.read().strip()
    except FileNotFoundError:
        print(f"Error: Prompt file not found at {prompt_path}")
        return
    except Exception as e:
        print(f"Error reading prompt file: {e}")
        return

    os.makedirs(result_dir, exist_ok=True)

    image_extensions = ["*.png", "*.jpg", "*.jpeg", "*.gif", "*.bmp"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(image_dir, ext)))

    if not image_files:
        print(f"No images found in {image_dir}")
        return

    # image_files = sorted(image_files)[5: 200]
    image_files = sorted(image_files)[800: 1500]
    for image_file_path in tqdm(image_files, desc="Processing images"):
        image_name = os.path.splitext(os.path.basename(image_file_path))[0]
        result_file_path = os.path.join(result_dir, f"{image_name}.json")
        # print(result_file_path)
        if os.path.exists(result_file_path):
            # print(result_file_path, "Already exists")
            continue
        
        try:
            image = Image.open(image_file_path).convert("RGB")
            image_b64 = encode_image(image)
        except Exception as e:
            print(f"Error processing image {image_file_path}: {e}")
            continue

        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
        ] + build_message_with_image_and_text(prompt_content, image_b64)

        try:
            chat_response = client.chat.completions.create(
                model=model_name,
                messages=messages,
            )
            answer_from_api = chat_response.choices[0].message.content

            image_name = os.path.splitext(os.path.basename(image_file_path))[0]
            result_file_path = os.path.join(result_dir, f"{image_name}.json")

            string_to_parse = answer_from_api
            markdown_stripped = False

            if model_name == "gpt-4o-ca":
                # For gpt-4o-ca, attempt direct parsing of the raw response
                pass # string_to_parse is already answer_from_api
            else: # For Gemini and other models, attempt to strip markdown first
                temp_processed_string = answer_from_api.strip()
                if temp_processed_string.startswith("```json") and temp_processed_string.endswith("```"):
                    first_newline = temp_processed_string.find('\n')
                    last_backticks = temp_processed_string.rfind('```')
                    if first_newline != -1 and last_backticks != -1 and first_newline < last_backticks:
                        string_to_parse = temp_processed_string[first_newline + 1:last_backticks].strip()
                        markdown_stripped = True
                elif temp_processed_string.startswith("```") and temp_processed_string.endswith("```"):
                    # Handle cases like ```{...}``` (no language specifier)
                    string_to_parse = temp_processed_string[3:-3].strip()
                    markdown_stripped = True
                else:
                    string_to_parse = temp_processed_string # No markdown detected, use stripped original
            
            try:
                parsed_answer = json.loads(string_to_parse)
                with open(result_file_path, "w", encoding="utf-8") as f:
                    json.dump(parsed_answer, f, ensure_ascii=False, indent=2)
            except json.JSONDecodeError:
                error_detail = "Failed to parse model output as JSON."
                if markdown_stripped:
                    error_detail = "Failed to parse model output as JSON after attempting to strip markdown."
                
                print(f"Warning: Could not parse API response for {image_file_path} as JSON (model: {model_name}). {error_detail} Saving raw. Raw response: <<<{answer_from_api}>>>")
                with open(result_file_path, "w", encoding="utf-8") as f:
                    json.dump({"result": answer_from_api, "error_message": error_detail}, f, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f"Error calling OpenAI API for {image_file_path} (model: {model_name}): {e}")
            image_name = os.path.splitext(os.path.basename(image_file_path))[0]
            result_file_path = os.path.join(result_dir, f"{image_name}_error.json")
            with open(result_file_path, "w", encoding="utf-8") as f:
                json.dump({"error": str(e), "image_file": image_file_path, "model_name": model_name}, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()

