# Artificial Pancreas Control Lab — Interactive Streamlit App

## Features
- Responsive dark-mode interface designed for desktop and mobile layouts.
- Searchable bundled food catalog covering Indian meals, continental foods, fast food/snacks, sweets/desserts, drinks, fruit, and basic foods.
- Primary meal input is food + quantity; calories, carbohydrates, estimated sugar, protein, and fat are calculated from the bundled per-100 g/ml catalog. Meal times are assigned automatically in one-hour increments in the order foods are added.
- Custom food form for local dishes/brands not in the catalog; edit `food_database.csv` to expand the bundled catalog.
- Six-hour (0–6 h) simulation timeline; live run targets 30 seconds of wall-clock time. Glucose, insulin delivery, plasma insulin, and insulin-on-board charts share the same time axis.
- PID, constrained PI with IOB, and LQR + observer; controller comparison, performance summaries, and CSV export.
- Food data are practical estimates informed by public references; mixed dishes, restaurant recipes, portion sizes, and brands vary. References: USDA FoodData Central (https://fdc.nal.usda.gov/) and ICMR–NIN Indian Food Composition Tables 2017 (https://www.nin.res.in/ebooks/IFCT2017.pdf). The bundled file is not a verbatim bulk copy of either source.

## Run locally
```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Deployment
Commit `app.py`, `food_database.csv`, `requirements.txt`, and `README_INTERACTIVE.md` to the same GitHub repository used by Streamlit Community Cloud. Streamlit Cloud installs dependencies from `requirements.txt`.

## Important limitation
This is an educational control-systems simulation, not a medical device. The Bergman model and controller are simplified and must not be used to make treatment decisions or calculate real insulin doses.
