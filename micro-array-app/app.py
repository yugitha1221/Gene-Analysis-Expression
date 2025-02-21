import os
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, redirect, flash, url_for, send_from_directory
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PROCESSED_FOLDER'] = 'processed'
app.config['STATIC_FOLDER'] = 'static'
app.secret_key = 'your_secret_key'

# Ensure folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)
os.makedirs(app.config['STATIC_FOLDER'], exist_ok=True)
app.add_url_rule('/processed/<path:filename>', endpoint='processed', view_func=lambda filename: send_from_directory(app.config['PROCESSED_FOLDER'], filename))


# Home page
@app.route('/')
def home():
    return render_template('home.html')

# Preprocessing page
@app.route('/preprocessing', methods=['GET', 'POST'])
def preprocessing():
    if request.method == 'POST':
        print("🔵 Received POST request")

        expression_file = request.files.get('expression_file')
        metadata_file = request.files.get('metadata_file')

        if not expression_file or not metadata_file:
            print("🔴 Error: One or both files missing!")
            flash("Both files are required!", "error")
            return redirect(request.url)

        if expression_file.filename == '' or metadata_file.filename == '':
            print("🔴 Error: Empty filename detected!")
            flash("Invalid file upload!", "error")
            return redirect(request.url)

        # Save files
        expression_path = os.path.join(app.config['UPLOAD_FOLDER'], expression_file.filename)
        metadata_path = os.path.join(app.config['UPLOAD_FOLDER'], metadata_file.filename)
        expression_file.save(expression_path)
        metadata_file.save(metadata_path)

        print(f"✅ Expression file saved at: {expression_path}")
        print(f"✅ Metadata file saved at: {metadata_path}")

        preprocess_type = request.form.get('preprocess_type')
        print(f"📌 Preprocess Type: {preprocess_type}")

        try:
            preprocessed_filepath, plot_path = preprocess_file(expression_path, preprocess_type)
            print(f"✅ Preprocessed file generated: {preprocessed_filepath}")

            return render_template(
                'results.html',
                filename=os.path.basename(preprocessed_filepath),
                plot_filename=os.path.basename(plot_path) if plot_path else None,
                download_link=url_for('processed', filename=os.path.basename(preprocessed_filepath))

            )
        except Exception as e:
            print(f"🔴 Processing Error: {e}")
            flash(f"Processing failed: {e}", "error")
            return redirect(request.url)

    return render_template('preprocessing.html')




@app.route('/analysis', methods=['GET', 'POST'])
def analysis():
    if request.method == 'POST':
        expression_file = request.files.get('expression_file')
        metadata_file = request.files.get('metadata_file')

        if not expression_file or not metadata_file:
            message = "Both expression data and metadata files are required."
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':  # AJAX request
                return {"error": message}, 400
            flash(message)
            return redirect(request.url)

        try:
            fold_change_cutoff = float(request.form.get('fold_change', 1.5))
            p_value_cutoff = float(request.form.get('p_value', 0.05))

            # Save uploaded files
            expression_path = os.path.join(app.config['UPLOAD_FOLDER'], expression_file.filename)
            metadata_path = os.path.join(app.config['UPLOAD_FOLDER'], metadata_file.filename)
            expression_file.save(expression_path)
            metadata_file.save(metadata_path)

            # Perform the analysis and get file paths
            results_filepath, volcano_plot_path = perform_analysis(expression_path, metadata_path, fold_change_cutoff, p_value_cutoff)

            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':  # AJAX request
                return {"filename": os.path.basename(results_filepath), "volcano_plot": os.path.basename(volcano_plot_path)}, 200

            return render_template('analysis_results.html', 
                                   filename=os.path.basename(results_filepath),
                                   volcano_plot_filename=os.path.basename(volcano_plot_path))

        except Exception as e:
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':  # AJAX request
                return {"error": str(e)}, 500
            flash(f"An error occurred during analysis: {e}")
            return redirect(request.url)

    return render_template('analysis.html')


# Visualization page
@app.route('/visualization/<filename>')
def visualization(filename):
    try:
        filepath = os.path.join(app.config['PROCESSED_FOLDER'], filename)
        data = pd.read_csv(filepath)

        # Normalize the data
        normalized_data = data.iloc[:, 1:].apply(lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() != x.min() else x)

        # Plot heatmap
        plt.figure(figsize=(16, 12))
        sns.heatmap(normalized_data, cmap='coolwarm', annot=False, cbar=True, vmin=0, vmax=1)
        plt.xticks(rotation=90, fontsize=8)
        plt.yticks(fontsize=8)

        # Save heatmap
        visualization_path = os.path.join(app.config['PROCESSED_FOLDER'], 'heatmap.png')
        plt.savefig(visualization_path)
        plt.close()

        return render_template('visualization.html', visualization_path=f'static/processed/heatmap.png')
    except Exception as e:
        flash(f"An error occurred during visualization: {e}")
        return redirect(url_for('home'))

# Download processed file
def preprocess_file(filepath, preprocess_type):
    try:
        data = pd.read_csv(filepath)
        numeric_columns = data.select_dtypes(include=[np.number]).columns

        if numeric_columns.empty:
            raise ValueError("No numeric columns found in the dataset.")

        if preprocess_type == 'minmax':
            scaler = MinMaxScaler()
            data[numeric_columns] = scaler.fit_transform(data[numeric_columns])
        elif preprocess_type == 'zscore':
            scaler = StandardScaler()
            data[numeric_columns] = scaler.fit_transform(data[numeric_columns])
        elif preprocess_type == 'log':
            data[numeric_columns] = data[numeric_columns].applymap(lambda x: np.log(x + 1) if x > 0 else 0)
        else:
            raise ValueError("Invalid preprocessing type")

        preprocessed_filepath = os.path.join(app.config['PROCESSED_FOLDER'], os.path.basename(filepath).replace('.csv', '_preprocessed.csv'))
        data.to_csv(preprocessed_filepath, index=False)

        print(f"Processed file saved: {preprocessed_filepath}")

        return preprocessed_filepath, None  # No plot for now
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        raise RuntimeError(f"Error in preprocessing: {e}")


# Perform differential expression analysis
def perform_analysis(expression_path, metadata_path, fold_change_cutoff, p_value_cutoff):
    expression_data = pd.read_csv(expression_path)
    metadata = pd.read_csv(metadata_path)

    # Ensure metadata groups are valid
    groups = metadata['Group'].unique()
    if len(groups) != 2:
        raise ValueError("Differential expression analysis requires exactly two groups.")

    group1, group2 = groups

    # Separate samples by group
    group1_samples = metadata[metadata['Group'] == group1]['Sample'].values
    group2_samples = metadata[metadata['Group'] == group2]['Sample'].values

    results = []
    for _, row in expression_data.iterrows():
        gene = row['Gene']
        try:
            # Extract and clean data for the groups
            group1_values = pd.to_numeric(row[group1_samples], errors='coerce').dropna()
            group2_values = pd.to_numeric(row[group2_samples], errors='coerce').dropna()

            # Skip genes with insufficient data
            if len(group1_values) < 2 or len(group2_values) < 2:
                continue

            # Perform t-test
            t_stat, p_value = ttest_ind(group1_values, group2_values, equal_var=False)

            # Calculate log2 fold change
            mean_group1 = np.mean(group1_values)
            mean_group2 = np.mean(group2_values)
            if mean_group2 == 0:  # Avoid division by zero
                continue
            log2fc = np.log2(mean_group1 / mean_group2)

            # Apply thresholds
            results.append([gene, log2fc, p_value])
        except Exception as e:
            print(f"Error processing gene {gene}: {e}")
            continue

    if not results:
        raise ValueError("No significant genes found based on the given thresholds.")

    # Convert results into a DataFrame
    results_df = pd.DataFrame(results, columns=['Gene', 'Log2FC', 'P-Value'])

    # Add a column to label upregulated, downregulated, or significant
    results_df['Regulation'] = results_df.apply(lambda row: 'Upregulated' if row['Log2FC'] >= 1 else ('Downregulated' if row['Log2FC'] <= -1 else 'Not Significant'), axis=1)

    # Save the results as a CSV
    results_filepath = os.path.join(app.config['PROCESSED_FOLDER'], 'differential_expression_results.csv')
    results_df.to_csv(results_filepath, index=False)

    # Create and save the volcano plot in the static folder
    plot_path = create_volcano_plot(results_df)  # Ensure this function returns the file path of the plot

    return results_filepath, plot_path


import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend

def create_volcano_plot(results_df):
    matplotlib.use('Agg')  # Ensure non-interactive backend
    plot_path = os.path.join(app.config['STATIC_FOLDER'], 'volcano_plot.png')

    plt.figure(figsize=(10, 6))
    plt.scatter(
        results_df['Log2FC'], 
        -np.log10(results_df['P-Value']), 
        c='blue', alpha=0.5, label='Significant Genes'
    )

    upregulated = results_df[results_df['Log2FC'] >= 1]
    downregulated = results_df[results_df['Log2FC'] <= -1]
    plt.scatter(upregulated['Log2FC'], -np.log10(upregulated['P-Value']), color='red', label='Upregulated')
    plt.scatter(downregulated['Log2FC'], -np.log10(downregulated['P-Value']), color='green', label='Downregulated')

    plt.axhline(y=-np.log10(0.05), color='black', linestyle='--', label='p-value=0.05')
    plt.title('Volcano Plot - Differential Expression')
    plt.xlabel('Log2 Fold Change')
    plt.ylabel('-Log10(p-value)')
    plt.legend()

    plt.savefig(plot_path)
    plt.close()

    return plot_path

if __name__ == '__main__':
    app.run(debug=True)  