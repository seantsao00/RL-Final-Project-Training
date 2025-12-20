"""
Script to extract and save the prompt for a specific class from ClassEval dataset.
"""
import argparse
from pathlib import Path
from datasets import load_dataset

from src.holisitc_classeval.data import question_to_prompt


def extract_class_prompt(class_name: str, output_file: Path):
    """Extract and save the prompt for a specific class."""
    # Load the full test split from HuggingFace
    dataset = load_dataset(
        "FudanSELab/ClassEval", split="test", trust_remote_code=True
    )
    
    # Find the class
    found = False
    for row in dataset:
        if row["class_name"] == class_name:
            found = True
            prompt = question_to_prompt(row["class_name"], row["skeleton"])
            
            # Format the prompt nicely
            output_content = f"Class: {class_name}\n"
            output_content += "=" * 80 + "\n\n"
            
            for msg in prompt:
                output_content += f"[{msg['role'].upper()}]\n"
                output_content += "-" * 80 + "\n"
                output_content += msg['content'] + "\n"
                output_content += "\n"
            
            # Save to file
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(output_content)
            
            print(f"Prompt for class '{class_name}' saved to: {output_file}")
            print(f"\nPreview:")
            print(output_content[:500] + "..." if len(output_content) > 500 else output_content)
            break
    
    if not found:
        print(f"Class '{class_name}' not found in dataset.")
        print("\nAvailable classes:")
        for i, row in enumerate(dataset):
            print(f"  {i+1}. {row['class_name']}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract prompt for a specific class from ClassEval dataset"
    )
    parser.add_argument(
        "--class_name",
        type=str,
        default="Thermostat",
        help="Name of the class to extract prompt for (default: Thermostat)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output file path (default: ./prompts/{class_name}_prompt.txt)",
    )
    
    args = parser.parse_args()
    
    # Set default output file if not specified
    if args.output_file is None:
        output_file = Path("./prompts") / f"{args.class_name}_prompt.txt"
    else:
        output_file = Path(args.output_file)
    
    extract_class_prompt(args.class_name, output_file)


if __name__ == "__main__":
    main()
