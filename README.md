# Homewise — House Price Assistant

## Windows par chalane ka tareeqa
1. ZIP ko Extract All karein.
2. Homewise_Source_Code folder VS Code mein kholen.
3. Terminal > New Terminal mein command chalayein:

   py -m http.server 8000 --bind 127.0.0.1 --directory dist

4. Browser mein http://localhost:8000 kholen.
5. Band karne ke liye terminal mein Ctrl+C.

Python installed hona chahiye. Agar py command na chale to python use karein.
Ya START_WINDOWS.bat double-click karke upar wala address browser mein kholen.
index.html directly double-click na karein: model JSON load hone ke liye HTTP server chahiye.
Training dobara zaroori nahi. Trained weights included hain. Koi API key nahi chahiye.

## Included files
- dist/index.html: webpage
- dist/style.css: responsive design
- dist/app.js: guided chat, validations, property form and examples
- dist/predict.js: JavaScript feature engineering, Box–Cox, one-hot encoding and Ridge inference
- dist/model.json: trained coefficients, transformation parameters, category schema and example values
- scripts/train.py: reproducible training/export workflow
- scripts/verify.cjs: comparison against Python validation predictions (Node.js optional)
- validation.json: 365 held-out input rows and Python predictions for numerical parity checks
- predictions.csv: 1,459 test.csv predictions in Kaggle submission format
- data/: your supplied train.csv, test.csv and data_description.txt
- house-price-prediction.ipynb: original notebook, unchanged

## Usage and assumptions
The assistant asks 8 structured questions. It is not a general LLM or an open-ended chatbot. All predictions run locally in the browser. An editable complete sample from test.csv supplies unspecified inputs; these are assumptions, not confirmed facts. The advanced panel exposes all 75 retained raw fields. Sample mode loads a complete example. Guided mode adjusts correlated floor areas when living area changes, assumes 240 square feet per garage space, assumes an unfinished basement with no basement bathrooms, and uses sale year 2010 after a build-year answer. Other fields retain sample values until edited. Review all fields for accurate input.

Predictions are historical USD sale prices for Ames, Iowa (2006–2010), not current market prices or Pakistan estimates. No conversion to PKR or inflation adjustment is made. Do not treat the result as a professional appraisal. Out-of-range or inconsistent inputs are rejected; unusual combinations within range may still be unreliable.

## Notebook integration
The notebook's chosen final estimator is Ridge(alpha=0.1). This app follows its GrLivArea<4500 filter, log1p target, categorical conversions, missing-value filling, engineered features, skew transformations, dummy encoding, 25% validation split with random_state=0 and expm1 output inversion. Model fitted on 1,093 rows; validated on 365; 324 encoded features. This is a newly trained implementation of that workflow, not a recovered serialized notebook model.

Compatibility fixes: modern SciPy's Pearson Box–Cox search failed on LotArea and 1stFlrSF; a bounded [-5,5] optimizer is used as fallback for these. Binary indicators remain untransformed to avoid unstable/ill-defined Box–Cox optimization. These adjustments slightly change results from the original notebook. They are implemented explicitly in scripts/train.py.

Validation log-RMSE: 0.1136782832. Dollar MAE: $14,379.55. Not an accuracy percentage, not a prediction interval. Following the original workflow, preprocessing was fitted on the combined training and unlabeled competition-test feature data before splitting. Thus validation is exploratory/transductive and should not be represented as an independent production benchmark. Simplified guided inputs have not been independently benchmarked.

All 365 full-input JavaScript predictions match Python to less than $0.0001. Desktop browser walkthrough and sample prediction were checked. Responsive CSS is included; a separate mobile-device test was not performed.

## Retrain (optional)
From the project root:

   py -m pip install -r requirements.txt
   py scripts/train.py

This overwrites dist/model.json, validation.json and predictions.csv. The website reads model metrics dynamically.
For a numerical parity check, if Node.js is installed:

   node scripts/verify.cjs

Google Fonts is optional; system sans-serif fonts are used if it is unavailable.
