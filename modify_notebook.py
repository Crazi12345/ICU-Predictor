import json
import sys

def modify_notebook():
    try:
        with open('Full_Training.ipynb', 'r') as f:
            nb = json.load(f)
    except FileNotFoundError:
        print("Error: Full_Training.ipynb not found.")
        return

    # 1. Update Args class
    args_updated = False
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            if "class Args:" in source and "use_all_pp = True" in source:
                new_source = source.replace(
                    "    use_all_pp = True\n",
                    "    use_all_pp = True\n    use_tiny_grid = True # Set to True for quick testing with single params\n"
                )
                cell['source'] = [s + "\n" if not s.endswith("\n") else s for s in new_source.splitlines(keepends=True)] 
                args_updated = True
                print("Args class updated.")
                break
    
    if not args_updated:
        print("Warning: Args class not found or already updated.")

    # 2. Update run_training_pipeline with tiny grid logic
    pipeline_updated = False
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            if "def run_training_pipeline" in source and "big_rnn_grid =" in source:
                # We will inject the tiny grid definition and selection logic
                
                # Locate the start of the else block for grid search
                search_str = "    else:\n        logging.info(f\"=== [{result_prefix}] FULL GRID SEARCH ON ALL MODELS (PYTORCH) - OFFICIAL SCORE & AUC ===\")"
                
                if search_str not in source:
                     # Try to match with potential variation in whitespace or quotes
                     pass 

                # Replacement logic: define tiny grids and use if/else
                # We'll replace the definitions of big_rnn_grid etc. with the logic
                
                old_block_start = "        big_rnn_grid = {"
                
                new_block = """
        # --- Tiny Grid (Single Value) ---
        tiny_rnn_grid = {
            'units': [64],
            'dropout': [0.2],
            'lr': [1e-3],
            'batch_size': [32],
            'final_activation': ['sigmoid'],
            'loss': ['binary_crossentropy'],
            'optimizer': ['adam']
        }

        tiny_cnn_grid = {
            'f1': [32],
            'f2': [64],
            'kernel_size': [3],
            'stride': [1],
            'dropout': [0.2],
            'lr': [1e-3],
            'batch_size': [32],
            'final_activation': ['sigmoid'],
            'loss': ['binary_crossentropy'],
            'optimizer': ['adam']
        }

        tiny_lgstm_grid = {
            'u1': [64],
            'u2': [32],
            'dropout': [0.2],
            'lr': [1e-3],
            'batch_size': [16],
            'final_activation': ['sigmoid'],
            'loss': ['binary_crossentropy'],
            'optimizer': ['adam']
        }

        if getattr(args, 'use_tiny_grid', False):
             logging.info(f\"=== [{result_prefix}] TINY GRID SEARCH (SINGLE VALUES) ===")
             rnn_grid = tiny_rnn_grid
             cnn_grid = tiny_cnn_grid
             lgstm_grid = tiny_lgstm_grid
        else:
             logging.info(f\"=== [{result_prefix}] FULL GRID SEARCH ON ALL MODELS (PYTORCH) - OFFICIAL SCORE & AUC ===")
             rnn_grid = {
                'units': [64, 128],
                'dropout': [0.2, 0.4],
                'lr': [1e-3, 5e-4],
                'batch_size': [32, 64],
                'final_activation': ['sigmoid', 'leakyrelu', 'elu', 'tanh'],
                'loss': ['binary_crossentropy', 'mse', 'hinge'],
                'optimizer': ['adam', 'adamw']
            }
             cnn_grid = {
                'f1': [32, 64],
                'f2': [64, 128],
                'kernel_size': [3, 5],
                'stride': [1, 2],
                'dropout': [0.2, 0.4],
                'lr': [1e-3, 5e-4],
                'batch_size': [32, 64],
                'final_activation': ['sigmoid', 'leakyrelu', 'elu', 'tanh'],
                'loss': ['binary_crossentropy', 'mse', 'hinge'],
                'optimizer': ['adam', 'adamw']
            }
             lgstm_grid = {
                'u1': [64, 128],
                'u2': [32, 64],
                'dropout': [0.2, 0.4],
                'lr': [1e-3, 5e-4],
                'batch_size': [8, 16],
                'final_activation': ['sigmoid', 'leakyrelu', 'elu', 'tanh'],
                'loss': ['binary_crossentropy', 'mse', 'hinge'],
                'optimizer': ['adam', 'adamw']
            }

        # Indices for Validation Eval
"""
                # Now we need to construct the new source code. 
                # The original code has big_rnn_grid = ... big_cnn_grid = ... big_lgstm_grid = ...
                # And then it uses them in grid_search_pyt calls.
                # We need to replace the definitions AND the usage.
                
                # Let's find the start of big_rnn_grid and the end of big_lgstm_grid definition
                
                # Heuristic: Find where 'val_indices =' starts, which is after the grids.
                split_point = source.find("        # Indices for Validation Eval")
                if split_point == -1:
                    print("Could not find split point 'Indices for Validation Eval'")
                    continue

                start_point = source.find("        big_rnn_grid = {")
                if start_point == -1:
                     print("Could not find start point 'big_rnn_grid = {'")
                     continue
                
                # Also we need to replace the usage of big_rnn_grid with rnn_grid etc.
                remaining_source = source[split_point:]
                remaining_source = remaining_source.replace("big_rnn_grid", "rnn_grid")
                remaining_source = remaining_source.replace("big_cnn_grid", "cnn_grid")
                remaining_source = remaining_source.replace("big_lgstm_grid", "lgstm_grid")
                
                # We also need to remove the original logging line since we put it inside the if/else
                original_logging = "        logging.info(f\"=== [{result_prefix}] FULL GRID SEARCH ON ALL MODELS (PYTORCH) - OFFICIAL SCORE & AUC ===\")\n\n"
                
                prefix_source = source[:start_point]
                
                # Try to remove the logging line from prefix_source if it exists right before start_point
                # It might have some newlines
                if original_logging.strip() in prefix_source:
                     # This is a bit unsafe with exact string matching on complex multiline strings.
                     # Let's just comment it out or leave it if it's tricky, but better to replace.
                     # Actually, I'll just check if the last few lines contain it.
                     pass

                # Safer regex-less approach:
                # We know the structure.
                # 1. Everything up to "    else:"
                else_idx = source.find("    else:\n")
                if else_idx == -1: 
                    print("Could not find else block")
                    continue
                
                prefix = source[:else_idx + len("    else:\n")]
                
                # 2. The new block
                
                # 3. The rest (from val_indices onwards), with variable replacements
                
                new_source_code = prefix + "\n" + new_block + remaining_source
                
                cell['source'] = [s for s in new_source_code.splitlines(keepends=True)]
                pipeline_updated = True
                print("Pipeline updated.")
                break

    if not pipeline_updated:
        print("Warning: run_training_pipeline not updated.")

    with open('Full_Training.ipynb', 'w') as f:
        json.dump(nb, f, indent=1)
    print("Notebook saved.")

if __name__ == "__main__":
    modify_notebook()
