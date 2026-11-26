# AutoWorth — Used Car Price Predictor

An interactive machine-learning app that estimates a second-hand car's selling price from its brand, model/variant, year, kilometres driven, fuel, seller type, transmission, and owner history. Prices are shown in Indian rupees.

## Problem statement

Used-car buyers and sellers often lack a consistent starting point for deciding a fair listing price. AutoWorth learns patterns from public CarDekho listings and gives an indicative estimate based on a car's details. It is an educational decision-support tool, not a formal appraisal.

## Project contents

- `app.py` — Streamlit application, data cleaning, model selection, prediction, and charts.
- `CarDekho_Used_Car_Analysis.ipynb` — reproducible data exploration and model training notebook.
- `requirements.txt` — Python dependencies.
- `data/car_details.csv` — bundled copy of the CarDekho listings so the app can run without downloading data at launch.
- `screenshots/` — add screenshots of your running application here before final submission.

## App features

- Brand and model/variant selection, with searchable model dropdown.
- Price estimate with an indicative guide range based on the held-out test MAE.
- Comparable listing median and count, with a chart comparing the model estimate to the dataset benchmark.
- Downloadable valuation summary as a CSV file.
- Market insight charts and support for uploading a compatible CSV.

## Dataset

The app uses the public [CarDekho used car dataset](https://github.com/bagassenop/cardekho/blob/main/CAR%20DETAILS%20FROM%20CAR%20DEKHO.csv). A copy is included at `data/car_details.csv`, so normal local use does not depend on a live dataset download. To refresh it, download the linked CSV and replace the local copy, or upload a compatible CSV in the app sidebar.

The app accepts CSV columns `name`, `year`, `selling_price`, `km_driven`, `fuel`, `seller_type`, `transmission`, and `owner`. In the estimator, choose a brand and then the closest matching model/variant (for example, **Maruti 800 AC**); type in the model dropdown to search.

## Run locally

1. Install Python 3.10 or newer.
2. Open a terminal in this project folder.
3. Create and activate a virtual environment (optional, recommended):

   ```bash
   python -m venv .venv
   # Windows PowerShell
   .venv\Scripts\Activate.ps1
   # macOS / Linux: source .venv/bin/activate
   ```

4. Install libraries and start the app:

   ```bash
   pip install -r requirements.txt
   streamlit run app.py
   ```

5. Open the local URL printed by Streamlit (usually `http://localhost:8501`). The app reads the included dataset and trains/evaluates the models on first launch. Internet is not needed to load the bundled dataset.

## Model training and evaluation

The app holds out 20% of the rows using a fixed random seed. It compares Random Forest and Extra Trees regressors by mean absolute error (MAE), selects the lower-error model, and refits it on all rows for prediction. Brand and model/variant are included as categorical features and one-hot encoded; missing numeric values are imputed. The app reports the held-out MAE and R² at runtime. Metrics depend on the exact dataset version downloaded; do not quote fixed values without recording the run used for your report.

For notebook-based exploration, install the requirements plus Jupyter (`pip install notebook`) and open `CarDekho_Used_Car_Analysis.ipynb`.

## Deploy on Streamlit Community Cloud

1. Create a GitHub repository and upload `app.py`, `requirements.txt`, this README, and the notebook.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/) with GitHub and choose **Create app**.
3. Select your repository, branch, and `app.py` as the main file, then deploy.
4. Wait for the app to finish loading. Try a prediction and open **Market insights** to confirm both pages work.
5. Add the resulting live URL below and include screenshots in `screenshots/`.

**GitHub repository:** _Add your repository URL_  
**Live application:** _Add the deployed application URL_

## Results and limitations

The app calculates metrics from a held-out test split at startup and presents them in the interface. Keep a screenshot of those actual results for the report. Listings may be dated and omit city, exact variant, condition, service history, or accident history; predictions can therefore differ from transaction prices.

## Credits

Dataset: publicly shared CarDekho listing data. This student project is not affiliated with CarDekho.
