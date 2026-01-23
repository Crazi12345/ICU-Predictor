import json
import re

def create_notebook():
    with open('training-Official.py', 'r') as f:
        code = f.read()

    # --- Splitting Logic ---
    # 1. Imports
    imports_end = code.find("# Configure logging")
    imports_code = code[:imports_end] + "\n# Configure logging (Modified for Notebook)\nimport sys\nimport logging\nlogging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout)])\n"
    
    # 2. Config & Seeds
    # Find start of Reproducibility
    repro_start = code.find("# Reproducibility")
    # Find start of Argument Parsing
    args_start = code.find("# Argument Parsing")
    
    config_code = code[repro_start:args_start]
    config_code += """
# --- Notebook Configuration ---
# Replaces command line arguments
class Args:
    use_cache = True
    use_last_final = False
    use_all_pp = True

args = Args()
"""

    # 3. Helper: get_last_final_configs
    glfc_start = code.find("def get_last_final_configs():")
    glfc_end = code.find("def get_data", glfc_start)
    glfc_code = code[glfc_start:glfc_end]

    # 4. Data Loading
    gd_start = code.find("def get_data")
    gd_end = code.find("# --- PyTorch Models ---")
    gd_code = code[gd_start:gd_end]

    # 5. Models
    models_start = code.find("# --- PyTorch Models ---")
    models_end = code.find("# --- Training & Evaluation Functions ---")
    models_code = code[models_start:models_end]

    # 6. Utils
    utils_start = code.find("# --- Training & Evaluation Functions ---")
    utils_end = code.find("def run_training_pipeline")
    utils_code = code[utils_start:utils_end]

    # 7. Pipeline
    pipeline_start = code.find("def run_training_pipeline")
    pipeline_end = code.find("def plot_preprocessing_comparison")
    pipeline_code = code[pipeline_start:pipeline_end]
    
    # 8. Plot Comparison
    plot_start = code.find("def plot_preprocessing_comparison")
    plot_end = code.find("if __name__ == \"__main__\":")
    plot_code = code[plot_start:plot_end]

    # 9. Main Execution
    # We'll rewrite the main execution to be more notebook-friendly
    main_code = """
# --- Execution ---
all_final_results = []

if args.use_all_pp:
    logging.info("=== RUNNING TRAINING ON ALL PREPROCESSING METHODS ===")
    methods = ['neg1', 'mean', 'median', 'linear']

    for method in methods:
        pkl_file = f"data/preprocessed_{method}.pkl"
        if not os.path.exists(pkl_file):
            logging.warning(f"File {pkl_file} not found. Skipping...")
            continue

        logging.info(f"\n\n>>> PROCESSING DATASET: {method} <<<")
        data = get_data(use_cache=True, cache_path=pkl_file)
        results = run_training_pipeline(data, result_prefix=method)
        all_final_results.extend(results)

else:
    # Default single-run behavior
    logging.info("=== RUNNING SINGLE DATASET TRAINING ===")
    data = get_data(use_cache=args.use_cache)
    results = run_training_pipeline(data, result_prefix="Default")
    all_final_results.extend(results)

# Summary
if all_final_results:
    res_df = pd.DataFrame(all_final_results).sort_values('utility', ascending=False)
    logging.info("\n=== FINAL TEST RESULTS SUMMARY (ALL METHODS) ===")
    logging.info(res_df.to_string(index=False))
    
    # Generate Preprocessing Comparison Plot
    try:
        plot_preprocessing_comparison(res_df)
        plt.show() # Ensure it shows in notebook
    except Exception as e:
        print(f"Plotting failed: {e}")
"""

    # --- Assembling Notebook ---
    cells = []
    
    def add_cell(source, type="code"):
        cells.append({
            "cell_type": type,
            "metadata": {},
            "source": [line + "\n" for line in source.splitlines()]
        })

    add_cell("# ICU Sepsis Prediction - Full Training Pipeline", "markdown")
    add_cell("## Imports", "markdown")
    add_cell(imports_code)
    add_cell("## Configuration", "markdown")
    add_cell(config_code)
    add_cell("## Helper Configs", "markdown")
    add_cell(glfc_code)
    add_cell("## Data Loading", "markdown")
    add_cell(gd_code)
    add_cell("## Model Architectures", "markdown")
    add_cell(models_code)
    add_cell("## Utilities", "markdown")
    add_cell(utils_code)
    add_cell("## Training Pipeline", "markdown")
    add_cell(pipeline_code)
    add_cell(plot_code)
    add_cell("## Execution", "markdown")
    add_cell(main_code)

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.10"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    with open("Full_Training.ipynb", "w") as f:
        json.dump(notebook, f, indent=1)
    print("Notebook Full_Training.ipynb created successfully.")

if __name__ == "__main__":
    create_notebook()
