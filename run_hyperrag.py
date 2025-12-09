'''Script to run the full HyperRAG pipeline'''
import subprocess
import sys

def run_step(command, step_name):
    print(f"\n=== Starting {step_name} ===")
    try:
        subprocess.run(command, shell=True, check=True)
        print(f"=== Completed {step_name} ===\n")
    except subprocess.CalledProcessError as e:
        print(f"Error during {step_name}: {e}")
        sys.exit(1)

def main():
    steps = [
        ("python src/spans_detection.py", "Span Detection"),
        ("python src/rag.py", "Candidate Retrieval (RAG)"),
        ("python src/reranking.py", "Reranking"),
        ("python src/evaluation.py", "Evaluation")
    ]

    for cmd, name in steps:
        run_step(cmd, name)

    print("Full pipeline completed successfully!")

if __name__ == "__main__":
    main()
