import os
import sys
import pandas as pd
import numpy as np

def read_and_average_files(task_name, directory):
    pattern = f"{task_name}_outputs.csv"
    files = []
    
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            if pattern in filename:
                full_path = os.path.join(dirpath, filename)
                files.append(full_path)

    if files:
        dataframes = []
        for file in files:
            df = pd.read_csv(file)
            df['predictions_probs'] = df['predictions_probs'].apply(eval)
            dataframes.append(df)
            
        df_concat = pd.concat(dataframes)

        group_cols = [col for col in df_concat.columns if col != 'predictions_probs']
        df_ensemble = df_concat.groupby(group_cols)['predictions_probs'].apply(lambda x: np.mean(np.vstack(x), axis=0)).reset_index()

        df_ensemble['new_prediction'] = df_ensemble['predictions_probs'].apply(lambda x: np.argmax(x))

        return df_ensemble
    else:
        return pd.DataFrame()  

def main(directory):

    task_names = ['cola', 'task2', 'task3'] 
    
    for task_name in task_names:
        df_ensemble = read_and_average_files(task_name, directory)
        if not df_ensemble.empty:
            output_file = os.path.join(directory, f"{task_name}_outputs_ensemble.csv")
            df_ensemble.to_csv(output_file, index=False)
            print(f"Ensemble output file saved: {output_file}")
        else:
            print(f"No files found for {task_name}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python ensemble_script.py <directory>")
        sys.exit(1)
    
    directory = sys.argv[1]
    main(directory)
