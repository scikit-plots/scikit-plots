# churn_basic workflow

A minimal end-to-end workflow template.

## Run

From the copied folder:

- Train:
  - `python train.py --data data.csv --model-out model.txt`
- HPO:
  - `python hpo.py --data data.csv --best-out best.json --trials 20`
- Predict:
  - `python predict.py --data data.csv --model model.txt --pred-out preds.txt`

Or via dispatcher:

- `python cli.py train -- --data data.csv --model-out model.txt`
