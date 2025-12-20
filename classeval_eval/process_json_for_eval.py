#!/usr/bin/env python3
"""
Script to extract completions from a JSON evaluation file and prepare them for evaluation.
"""
import json
import re
from pathlib import Path
import argparse


def extract_code_from_completion(completion: str) -> str:
    """Extract Python code from completion, removing markdown code blocks if present."""
    # Remove markdown code blocks
    code = re.sub(r'^```python\n', '', completion, flags=re.MULTILINE)
    code = re.sub(r'\n```$', '', code, flags=re.MULTILINE)
    code = code.strip()
    return code


def infer_class_name(code: str) -> str | None:
    """Infer the class name from the Python code."""
    match = re.search(r'^class\s+(\w+)', code, re.MULTILINE)
    if match:
        return match.group(1)
    return None


def process_json_file(json_path: Path, output_dir: Path) -> None:
    """
    Process the JSON file and create class files and test files.
    
    Args:
        json_path: Path to the JSON file containing completions
        output_dir: Directory to write the composed code and tests
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    detailed_results = data.get('detailed_results', [])
    
    if not detailed_results:
        raise ValueError("No detailed_results found in JSON file")
    
    # Create output directory structure
    composed_code_dir = output_dir / "composed_code"
    composed_code_dir.mkdir(parents=True, exist_ok=True)
    
    processed_count = 0
    
    for result in detailed_results:
        example_id = result.get('example_id')
        completion = result.get('completion', '')
        tests = result.get('tests', '')
        
        if not completion:
            print(f"Warning: No completion for example_id {example_id}, skipping")
            continue
        
        # Extract code from completion
        code = extract_code_from_completion(completion)
        
        # Infer class name
        class_name = infer_class_name(code)
        if not class_name:
            print(f"Warning: Could not infer class name for example_id {example_id}, skipping")
            continue
        
        # Create directory for this class
        class_dir = composed_code_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        
        # Write class file
        class_file = class_dir / f"{class_name}.py"
        with open(class_file, 'w', encoding='utf-8') as f:
            f.write(code)
        
        # Write test file
        if tests:
            test_file = class_dir / f"full_test_{class_name}.py"
            # Compose full test file with class code + tests + unittest.main()
            full_test_content = f"{code}\n\n{tests}\n\nif __name__ == '__main__':\n    unittest.main()\n"
            with open(test_file, 'w', encoding='utf-8') as f:
                f.write(full_test_content)
        
        processed_count += 1
        print(f"Processed {class_name} (example_id: {example_id})")
    
    print(f"\nTotal processed: {processed_count} classes")
    print(f"Output directory: {composed_code_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract completions from JSON and prepare for evaluation"
    )
    parser.add_argument(
        "--json_file",
        type=str,
        required=True,
        help="Path to the JSON file containing evaluation results with completions"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to write the composed code (will create composed_code subdirectory)"
    )
    
    args = parser.parse_args()
    
    json_path = Path(args.json_file)
    output_dir = Path(args.output_dir)
    
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    
    process_json_file(json_path, output_dir)
    print("\nDone! You can now run evaluate.py with:")
    print(f"  uv run evaluate.py --eval_output_dir {output_dir}")


if __name__ == "__main__":
    main()
