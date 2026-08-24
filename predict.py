"""Score new applications using the bundle produced by train_model.py."""
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from train_model import add_safe_features, find_column, read_csv_robust

def main() -> None:
    p=argparse.ArgumentParser(description="Predict bad-debt probability")
    p.add_argument("--model",type=Path,default=Path("artifacts/bad_debt_model.joblib"))
    p.add_argument("--input",required=True,type=Path)
    p.add_argument("--output",type=Path,default=Path("artifacts/predictions.csv")); args=p.parse_args()
    if not args.model.is_file(): raise FileNotFoundError(f"Model not found: {args.model}")
    bundle=joblib.load(args.model); model,metadata=bundle["model"],bundle["metadata"]
    data=add_safe_features(read_csv_robust(args.input),metadata["as_of_date"])
    expected=metadata["features"]; rename={}
    for feature in expected:
        match=find_column(data.columns.tolist(),feature)
        if match: rename[match]=feature
    data=data.rename(columns=rename); missing=[x for x in expected if x not in data.columns]
    if missing: raise ValueError(f"Input is missing required model features: {missing}")
    probability=model.predict_proba(data[expected])[:,1]; threshold=float(metadata["threshold"])
    result=data.copy(); result["ProbabilityBadDebt"]=probability
    result["PredictedBadDebt"]=(probability>=threshold).astype("int8")
    result["RiskBand"]=pd.cut(probability,[-np.inf,.2,.5,.8,np.inf],labels=["Low","Medium","High","Very High"])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    result.to_csv(args.output,index=False,encoding="utf-8-sig")
    print(f"Scored {len(result)} rows -> {args.output.resolve()}")
if __name__ == "__main__": main()
