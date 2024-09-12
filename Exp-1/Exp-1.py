import subprocess
import pandas as pd
import os
import openai
import json
import logging
from tqdm import tqdm
import time
from functools import wraps
import anthropic

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("single_policy_dual_analysis.log"), logging.StreamHandler()])

#Set Anthropic and Open_AI clients up with API Keys


gpt_model_name = "ft:gpt-4o-mini-2024-07-18:personal::A5b7jUfX"
claude_model_name = "claude-3-5-sonnet-20240620"

# Define paths
policy_folder = "/home/adarsh/Documents/Experiments/Dataset"
quacky_path = "/home/adarsh/Documents/quacky/src/quacky.py"
working_directory = "/home/adarsh/Documents/quacky/src/"
response_file_path = "/home/adarsh/Documents/quacky/src/response.txt"
response2_file_path = "/home/adarsh/Documents/quacky/src/response2.txt"
result_table_path = "/home/adarsh/Documents/Experiments/Exp-1/single_policy_dual_analysis.csv"
generated_policy_path = "/home/adarsh/Documents/quacky/src/gen_pol.json"
p1_not_p2_models_path = "/home/adarsh/Documents/quacky/src/P1_not_P2.models"
not_p1_p2_models_path = "/home/adarsh/Documents/quacky/src/not_P1_P2.models"
progress_file_path = "/home/adarsh/Documents/Experiments/Exp-1/single_policy_dual_progress.json"

def read_policy_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()

def retry(max_attempts=3, delay=5):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempts += 1
                    logging.warning(f"Attempt {attempts} failed: {str(e)}. Retrying in {delay} seconds...")
                    time.sleep(delay)
            raise Exception(f"Function {func.__name__} failed after {max_attempts} attempts.")
        return wrapper
    return decorator

@retry(max_attempts=3, delay=5)
def get_policy_description(policy_content):
    prompt = f"Please provide a short description of what the following policy is doing:\n\n{policy_content}\n\nDescription:"
    
    try:
        response = anthropic_client.messages.create(
            model=claude_model_name,
            max_tokens=300,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.content[0].text.strip()
    except Exception as e:
        logging.error(f"Error calling Anthropic API for policy description: {str(e)}")
        return ""

@retry(max_attempts=3, delay=5)
def generate_new_policy(description):
    policy_system_prompt = """When asked to generate a policy, provide only the policy in JSON format. Do not include any explanations, markdown formatting, or additional text. The response should be a valid JSON object that can be directly parsed.

    Example response format:
 {
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["s3:GetObject"],
      "Resource": "arn:aws:s3:::example-bucket/*"
    }
  ]
 }

 Ensure that the generated policy is relevant to the given description and follows the access control policy structure."""

    prompt = f"Based on the following description, generate a new access control policy:\n\n{description}\n\nPolicy:"
    
    try:
        response = anthropic_client.messages.create(
            model=claude_model_name,
            max_tokens=1000,
            system=policy_system_prompt,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.content[0].text.strip()
    except Exception as e:
        logging.error(f"Error calling Anthropic API for policy generation: {str(e)}")
        return ""

def save_generated_policy(policy_content, file_path):
    try:
        with open(file_path, 'w') as file:
            json.dump(json.loads(policy_content), file, indent=2)
        logging.info(f"Generated policy saved to: {file_path}")
    except json.JSONDecodeError:
        logging.error("Generated policy is not valid JSON. Saving as plain text.")
        with open(file_path, 'w') as file:
            file.write(policy_content)

def generate_strings(original_policy_path, generated_policy_path, size):
    command = [
        "python3", quacky_path,
        "-p1", original_policy_path,
        "-p2", generated_policy_path,
        "-b", "100",
        "-m", str(size),
        "-m1", "20",
        "-m2", "100"
    ]
    
    result = subprocess.run(command, cwd=working_directory, capture_output=True, text=True)
    logging.info("Getting strings:")
    if result.stderr:
        logging.error(f"Errors: {result.stderr}")
    
    return result.returncode == 0

@retry(max_attempts=3, delay=5)
def generate_regex(strings_file_path, response_file_path):
    with open(strings_file_path, 'r') as file:
        strings = file.read()

    system_prompt = """
    When asked to give a regex, provide ONLY the regex pattern itself. Do not include any explanations, markdown formatting, or additional text. The response should be just the regex pattern, nothing else. This is a highly critical application and it is imperative to get this right. Just give me the regex.
    """
    prompt = f"Give me a single regex that accepts each string in the following set of strings, it should be close to optimal and not super permissive:\n\n{strings}\n\nResponse:"

    try:
        response = openai_client.chat.completions.create(
            model=gpt_model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000
        )
        regex = response.choices[0].message.content.strip()
        regex = regex.replace("^", "").replace("$", "").strip()
        
        with open(response_file_path, "w") as output_file:
            output_file.write(regex)
        
        logging.info(f"Regex generated and written to {response_file_path}")
        return regex
    except Exception as e:
        logging.error(f"Error calling OpenAI API for regex generation: {str(e)}")
        return None

@retry(max_attempts=3, delay=5)
def run_final_analysis(original_policy_path, generated_policy_path):
    command = [
        "python3", quacky_path,
        "-p1", original_policy_path,
        "-p2", generated_policy_path,
        "-b", "100",
        "-cr", response_file_path,
        "-cr2", response2_file_path
    ]
    
    result = subprocess.run(command, cwd=working_directory, capture_output=True, text=True)
    logging.info("Quacky Final Analysis Output:")
    logging.info(result.stdout)
    if result.stderr:
        logging.error(f"Errors: {result.stderr}")
    return result.stdout

def get_progress():
    if os.path.exists(progress_file_path):
        with open(progress_file_path, 'r') as f:
            return json.load(f)
    return {"last_processed": 0}

def update_progress(last_processed):
    with open(progress_file_path, 'w') as f:
        json.dump({"last_processed": last_processed}, f)

@retry(max_attempts=3, delay=10)
def process_policy(policy_path, size):
    start_time = time.time()
    original_policy = read_policy_file(policy_path)
    policy_description = get_policy_description(original_policy)
    logging.info(f"Policy Description:\n{policy_description}")
    
    new_policy = generate_new_policy(policy_description)
    logging.info("Generated Policy:")
    logging.info(new_policy)
    
    save_generated_policy(new_policy, generated_policy_path)
    
    if not generate_strings(policy_path, generated_policy_path, size):
        raise Exception("Failed to generate strings.")
    
    regex1 = generate_regex(p1_not_p2_models_path, response_file_path)
    regex2 = generate_regex(not_p1_p2_models_path, response2_file_path)
    
    if not regex1 or not regex2:
        raise Exception("Failed to generate one or both regexes.")
    
    final_analysis = run_final_analysis(policy_path, generated_policy_path)
    
    end_time = time.time()
    total_processing_time = end_time - start_time
    
    return {
        "model_name": gpt_model_name,
        "Original Policy": original_policy,
        "Generated Policy": new_policy,
        "Policy Description": policy_description,
        "Size": size,
        "Regex from llm (P1_not_P2)": regex1,
        "Regex from llm (not_P1_P2)": regex2,
        "Final Analysis": final_analysis,
        "Total Processing Time (seconds)": total_processing_time
    }

if __name__ == "__main__":
    policy_files = sorted([f for f in os.listdir(policy_folder) if f.endswith('.json')], key=lambda x: int(x.split('.')[0]))[:41]  # Sort and limit to 0-40
    
    size = 500  # You can make this configurable if needed

    # Get the progress
    progress = get_progress()
    start_index = progress["last_processed"]

    print(f"Starting from policy number {start_index}")

    # Initialize or load the results DataFrame
    required_columns = [
        "Policy Number", "model_name", "Original Policy", "Generated Policy", "Policy Description",
        "Size", "Regex from llm (P1_not_P2)", "Regex from llm (not_P1_P2)", "Final Analysis", "Total Processing Time (seconds)"
    ]

    if not os.path.exists(result_table_path) or os.stat(result_table_path).st_size == 0:
        result_table = pd.DataFrame(columns=required_columns)
    else:
        result_table = pd.read_csv(result_table_path)
        for column in set(required_columns) - set(result_table.columns):
            result_table[column] = ""

    for i in tqdm(range(start_index, len(policy_files)), desc="Processing policies"):
        policy_file = policy_files[i]
        policy_path = os.path.join(policy_folder, policy_file)
        policy_number = policy_file.split('.')[0]  # This is now a string, e.g., "0", "1", "2", etc.
        logging.info(f"\nProcessing policy: {policy_file}")
        
        try:
            new_entry = process_policy(policy_path, size)
            new_entry["Policy Number"] = policy_number
            result_table = pd.concat([result_table, pd.DataFrame([new_entry])], ignore_index=True)
            result_table.to_csv(result_table_path, index=False)
            logging.info(f"Results table updated and saved to {result_table_path}")
            update_progress(i + 1)
        except Exception as e:
            logging.error(f"Failed to process policy after multiple attempts: {policy_file}. Error: {str(e)}")
            continue

    logging.info("\nProcessing complete. Final results table saved to: " + result_table_path)
    print(f"Processed {len(policy_files) - start_index} policies. Next run will start from policy number {len(policy_files)}")
