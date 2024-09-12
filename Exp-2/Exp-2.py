import subprocess
import pandas as pd
import os
import anthropic
import json
import logging
from tqdm import tqdm
import signal
import re
import time
policy_folder = "/home/adarsh/Documents/Policy_Verification_with_LLMS/Dataset"
quacky_path = "/home/adarsh/Documents/quacky/src/quacky.py"
working_directory = "/home/adarsh/Documents/quacky/src/"
response_file_path = "/home/adarsh/Documents/quacky/src/response.txt"
p1_not_p2_models_path = "/home/adarsh/Documents/quacky/src/P1_not_P2.models"
fine_tuning_dataset_path = "/home/adarsh/Documents/Experiments/Fine-tuning/fine-tuning-v2/fine_tuning_dataset.jsonl"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("policy_analysis.log"), logging.StreamHandler()])


#Initialize Anthropic API client with API Key

model_name = "claude-3-5-sonnet-20240620"

# Define paths
policy_folder = "/home/adarsh/Documents/Experiments/Dataset"
quacky_path = "/home/adarsh/Documents/quacky/src/quacky.py"
working_directory = "/home/adarsh/Documents/quacky/src/"
response_file_path = "/home/adarsh/Documents/quacky/src/response.txt"
result_table_path = "/home/adarsh/Documents/Experiments/Exp-2/multi-string.csv"
generated_policy_path = "/home/adarsh/Documents/quacky/src/gen_pol.json"
p1_not_p2_models_path = "/home/adarsh/Documents/quacky/src/P1_not_P2.models"
progress_file_path = "/home/adarsh/Documents/Experiments/Exp-2/progress.json"

def read_policy_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()




def generate_strings(policy_path, size):
    command = [
        "python3", quacky_path,
        "-p1", policy_path,
        "-b", "100",
        "-m", str(size),
        "-m1", "20",
        "-m2", "100"
    ]
    
    result = subprocess.run(command, cwd=working_directory, capture_output=True, text=True)
    logging.info("Getting strings:")
    if result.stderr:
        logging.error(f"Errors: {result.stderr}")
    
    with open(p1_not_p2_models_path, 'r') as file:
        strings = file.read()
    
    return strings

def generate_regex(strings):
    system_prompt = """
    When asked to give a regex, provide ONLY the regex pattern itself. Do not include any explanations, markdown formatting, or additional text. The response should be just the regex pattern, nothing else. This is a highly critical application and it is imperative to get this right. Just give me the regex.
    
    Example bad(terrible) response(DO NOT WANT THIS IN ANY CASE):
    
    "Here is the regex pattern based on the provided set of strings: (?:foo|bar)[a-z0-9.-]{0,60}"


    Example good response:

    "(?:foo|bar)[a-z0-9.-]{0,60}"

    """
    prompt = f"Give me a single regex that accepts each string in the following set of strings, Make sure that you carefully go through each string before forming the regex. it should be close to optimal and not super permissive:\n\n{strings}\n\n , Example bad(terrible) response(DO NOT WANT THIS IN ANY CASE): Here is the regex pattern based on the provided set of strings: arn:aws:ec2:us-east-1:(?:\d:?)?(?:(?:key-pair|subnet|security-group|network-interface|volume|instance)|image/ami-)[!-~], Example good response: arn:aws:ec2:us-east-1:(?:\d:?)?(?:(?:key-pair|subnet|security-group|network-interface|volume|instance)|image/ami-)[!-~], (This response acts as an input for a regex analysis application so if you give me any sort of additional text along with the regex like Here's a regex matching the strings , etc. etc. the application will fail, so only reply with the regex and nothing else.)"


    try:
        response = client.messages.create(
            model=model_name,
            max_tokens=1000,
            system=system_prompt,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        regex = response.content[0].text.strip()
        
        with open(response_file_path, "w") as output_file:
            output_file.write(regex)
        
        logging.info(f"Regex generated and written to {response_file_path}")
        return regex
    except Exception as e:
        logging.error(f"Error calling Anthropic API for regex generation: {str(e)}")
        return None

def timeout_handler(signum, frame):
    raise TimeoutError("Analysis took too long")

def run_final_analysis(policy_path, timeout=2000):
    command = [
        "python3", quacky_path,
        "-p1", policy_path,
        "-b", "100",
        "-cr", response_file_path
    ]

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)

    try:
        result = subprocess.run(command, cwd=working_directory, capture_output=True, text=True)
        signal.alarm(0)

        if result.returncode != 0 or "FATAL ERROR FROM ABC" in result.stderr:
            raise Exception("Quacky analysis failed")

        logging.info("Quacky Final Analysis Output:")
        logging.info(result.stdout)
        if result.stderr:
            logging.error(f"Errors: {result.stderr}")
        return result.stdout
    except TimeoutError:
        logging.error(f"Final analysis for policy {policy_path} timed out after {timeout} seconds.")
        return "TIMEOUT"
    except Exception as e:
        logging.error(f"Error in final analysis: {str(e)}")
        return None
    finally:
        signal.alarm(0)


def process_policy(policy_path, size, max_retries=5):
    errors = []
    original_policy = ""

    try:
        original_policy = read_policy_file(policy_path)
    except Exception as e:
        errors.append(f"Error reading policy file: {str(e)}")
        return {
            "model_name": model_name,
            "Original Policy": original_policy,
            "Size": size,
            "Regex from llm": "",
            "Experiment 2_Analysis": "",
            "Errors": "; ".join(errors)
        }

    for attempt in range(max_retries):
        try:
            # Generate strings
            strings = generate_strings(policy_path, size)
            if not strings:
                raise Exception("Failed to generate strings")

            # Generate regex
            regex = generate_regex(strings)
            if not regex:
                raise Exception("Failed to generate regex")

            # Run final analysis
            exp2_raw_output = run_final_analysis(policy_path)
            if exp2_raw_output is None or exp2_raw_output == "TIMEOUT":
                raise Exception("Final analysis failed or timed out")

            # If we reach here, the process was successful
            return {
                "model_name": model_name,
                "Original Policy": original_policy,
                "Size": size,
                "Regex from llm": regex,
                "Experiment 2_Analysis": exp2_raw_output,
                "Errors": ""
            }

        except Exception as e:
            logging.error(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt == max_retries - 1:
                errors.append(f"Process failed after {max_retries} attempts: {str(e)}")
            else:
                time.sleep(5)  # Wait for 5 seconds before retrying

    # If we reach here, all attempts have failed
    return {
        "model_name": model_name,
        "Original Policy": original_policy,
        "Size": size,
        "Regex from llm": "",
        "Experiment 2_Analysis": "",
        "Errors": "; ".join(errors)
    }

def get_progress():
    if os.path.exists(progress_file_path):
        with open(progress_file_path, 'r') as f:
            return json.load(f)
    return {"last_processed": 0}

def update_progress(last_processed):
    with open(progress_file_path, 'w') as f:
        json.dump({"last_processed": last_processed}, f)

if __name__ == "__main__":
    def sort_key(filename):
        # Extract the number from the filename
        match = re.search(r'(\d+)', filename)
        if match:
            return int(match.group(1))
        return 0  # Return 0 if no number found

    policy_files = sorted([f for f in os.listdir(policy_folder) if f.endswith('.json')], key=sort_key)
    total_policies = len(policy_files)
    
    size = 1000  # You can make this configurable if needed

    # Get the number of policies to process
    while True:
        try:
            num_policies = int(input(f"Enter the number of policies to process (1-{total_policies}) or -1 for all remaining policies: "))
            if num_policies == -1 or (1 <= num_policies <= total_policies):
                break
            else:
                print(f"Please enter a number between 1 and {total_policies}, or -1 for all remaining policies.")
        except ValueError:
            print("Invalid input. Please enter a valid number.")

    # Get the progress
    progress = get_progress()
    start_index = progress["last_processed"]

    # Ensure start_index is within valid range
    start_index = max(0, min(start_index, total_policies - 1))

    print(f"Starting from policy number {start_index + 1}")

    # Initialize or load the results DataFrame
    required_columns = [
        "model_name", "Original Policy", "Size", "Regex from llm", "Experiment 2_Analysis", "Errors"
    ]

    if not os.path.exists(result_table_path) or os.stat(result_table_path).st_size == 0:
        result_table = pd.DataFrame(columns=required_columns)
    else:
        try:
            result_table = pd.read_csv(result_table_path)
            # Check if the DataFrame is empty or doesn't have the required columns
            if result_table.empty or not all(col in result_table.columns for col in required_columns):
                result_table = pd.DataFrame(columns=required_columns)
        except pd.errors.EmptyDataError:
            # If the file is empty, create a new DataFrame
            result_table = pd.DataFrame(columns=required_columns)
        
        # Ensure all required columns are present
        for column in set(required_columns) - set(result_table.columns):
            result_table[column] = ""

    end_index = total_policies if num_policies == -1 else min(start_index + num_policies, total_policies)

    for i in tqdm(range(start_index, end_index), desc="Processing policies"):
        if i < total_policies:
            policy_file = policy_files[i]
            policy_path = os.path.join(policy_folder, policy_file)
            logging.info(f"\nProcessing policy: {policy_file}")
            
            new_entry = process_policy(policy_path, size, max_retries=5)
            
            result_table = pd.concat([result_table, pd.DataFrame([new_entry])], ignore_index=True)
            result_table.to_csv(result_table_path, index=False)
            logging.info(f"Results table updated and saved to {result_table_path}")
            update_progress(i + 1)
        else:
            logging.warning(f"Index {i} is out of range. Stopping processing.")
            break

    logging.info("\nProcessing complete. Final results table saved to: " + result_table_path)
    print(f"Processed {end_index - start_index} policies. Next run will start from policy number {min(end_index + 1, total_policies)}")
    
    logging.info("Starting CSV processing...")

    def parse_analysis(analysis):
        if not isinstance(analysis, str):
            return '', {}
        
        fields = {
            'Policy_Analysis': '',
            'Baseline Regex Count': None,
            'Synthesized Regex Count': None,
            'Baseline_Not_Synthesized Count': None,
            'Not_Baseline_Synthesized_Count': None,
            'regex_from_dfa': None,
            'regex_from_llm': None,
            'ops_regex_from_dfa': None,
            'ops_regex_from_llm': None,
            'length_regex_from_dfa': None,
            'length_regex_from_llm': None,
            'jaccard_numerator': None,
            'jaccard_denominator': None
        }

        # Extract Policy Analysis
        policy_match = re.search(r'Policy 1.*?lg\(requests\): [\d.]+', analysis, re.DOTALL)
        if policy_match:
            fields['Policy_Analysis'] = policy_match.group(0)

        # Extract other fields
        for field in fields:
            if field != 'Policy_Analysis':
                match = re.search(rf'{field}\s*:\s*(.*?)(?:\n|$)', analysis)
                if match:
                    fields[field] = match.group(1).strip()

        return fields

    # Read the CSV file
    df = pd.read_csv(result_table_path, encoding='utf-8')

    # Apply the parsing function to the Experiment 2_Analysis column
    parsed_data = df['Experiment 2_Analysis'].apply(parse_analysis)

    # Create new columns for each field
    for field in parsed_data.iloc[0].keys():
        df[field] = parsed_data.apply(lambda x: x[field] if isinstance(x, dict) else '')

    # Reorder columns
    columns_order = ['model_name', 'Original Policy', 'Size', 'Regex from llm', 'Policy_Analysis'] + \
                    [col for col in df.columns if col not in ['model_name', 'Original Policy', 'Size', 'Regex from llm', 'Policy_Analysis', 'Experiment 2_Analysis']] + \
                    ['Errors']
    df = df[columns_order]

    # Save the processed DataFrame to a new CSV file
    processed_csv_path = os.path.join(os.path.dirname(result_table_path), 'Exp-2.csv')
    df.to_csv(processed_csv_path, index=False, encoding='utf-8')
    logging.info(f"CSV file has been processed and saved as '{processed_csv_path}'")
    print(f"Processed CSV file saved as '{processed_csv_path}'")
