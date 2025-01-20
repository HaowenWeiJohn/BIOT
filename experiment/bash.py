import subprocess
import sys



TUEV = "TUEV"
TUAB = "TUAB"
IIIC = "IIIC"

SPaRCNet = "SPaRCNet"
CNNTransformer = "CNNTransformer"
FFCL = "FFCL"
ContraWR = "ContraWR"
STTransformer = "STTransformer"
BIOT = "BIOT"

# Get the Python executable path of the current environment
python_path = sys.executable

def main():
    # List your (model, dataset) combinations or however you want to structure the arguments
    combinations = [
        # (SPaRCNet, TUAB),
        # (CNNTransformer, TUAB),
        # (FFCL, TUAB),
        # (ContraWR, TUAB),
        (STTransformer, TUAB),
        (BIOT, TUAB),
    ]

    # The script you want to run sequentially
    script_name = "train_script_arg.py"

    # Loop through the combinations and call the script with the arguments
    for model_name, dataset_name in combinations:
        print(f"\nRunning {script_name} with model={model_name} and dataset={dataset_name}...")

        # Use subprocess.run to call Python, pass the script name and arguments
        # You can decide on whether you pass them as positional or named arguments
        subprocess.run([
            python_path,
            script_name,
            "--model_name", model_name,
            "--dataset_name", dataset_name
        ], check=True)

    print("\nAll runs completed successfully!")


if __name__ == "__main__":
    main()
