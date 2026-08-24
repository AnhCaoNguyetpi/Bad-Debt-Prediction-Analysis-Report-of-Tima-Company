"""Reproducible training pipeline for pre-disbursement bad-debt prediction."""
from __future__ import annotations

import argparse, json, logging, re, unicodedata
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, PrecisionRecallDisplay,
    RocCurveDisplay, accuracy_score, average_precision_score,
    balanced_accuracy_score, classification_report, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LOGGER = logging.getLogger("bad_debt_training")
RANDOM_STATE = 42
TARGET = "HasBadDebt"

def canonical(value: str) -> str:
    value = unicodedata.normalize("NFKD", str(value))
    value = "".join(c for c in value if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", "", value.casefold())

POST_OUTCOME_NAMES = {"HasLatePayment", "LongestOverdue", "SoTienConLai",
    "TienGiaiNgan", "ToDate", "CheckTime", "Trạng thái", "Trang thai",
    "PaymentStatus", "DaysPastDue", "DPD", "OutstandingBalance", "RemainingAmount"}
POST_OUTCOME_NAMES |= {"TyLeGiaiNgan", "TyLeNoSoVoiThuNhap", "DisbursementRatio",
                       "DebtToIncomeFromOutstandingBalance"}
IDENTIFIER_NAMES = {"ID", "STT", "LoanID", "CustomerID", "Name", "FullName",
    "Address", "Phone", "PhoneNumber", "CardNumber", "CMND", "CCCD", "Email",
    "Số điện thoại khách hàng", "MaKhachHang", "MaKhoanVay"}
GROUP_CANDIDATES = {"CustomerID", "Phone", "PhoneNumber", "CardNumber", "CMND",
    "CCCD", "Số điện thoại khách hàng", "MaKhachHang"}

@dataclass
class SplitInfo:
    strategy: str
    group_column: str | None
    train_rows: int
    validation_rows: int
    test_rows: int

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train and evaluate a leakage-safe bad-debt classifier")
    p.add_argument("--input", required=True, type=Path, help="Raw CSV file")
    p.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    p.add_argument("--target", default=TARGET)
    p.add_argument("--as-of-date", default="2019-03-17", help="Age reference date (YYYY-MM-DD)")
    p.add_argument("--exclude-columns", nargs="*", default=[], help="Extra fields to exclude")
    p.add_argument("--group-column", help="Customer ID used to prevent split overlap")
    p.add_argument("--include-post-outcome", action="store_true",
                   help="Monitoring-only: permit fields observed after approval")
    p.add_argument("--threshold-objective", choices=["f1", "recall"], default="f1")
    p.add_argument("--min-precision", type=float, default=.20,
                   help="Precision floor when optimizing recall")
    return p.parse_args()

def read_csv_robust(path: Path) -> pd.DataFrame:
    if not path.is_file(): raise FileNotFoundError(f"Input file not found: {path}")
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "cp1258", "latin1"):
        try:
            df = pd.read_csv(path, encoding=encoding, sep=None, engine="python")
            if df.shape[1] > 1:
                df.columns = df.columns.astype(str).str.strip()
                return df
        except (UnicodeDecodeError, pd.errors.ParserError) as error: last_error = error
    raise ValueError(f"Could not read CSV {path}: {last_error}")

def find_column(columns: list[str], requested: str) -> str | None:
    wanted = canonical(requested)
    return next((c for c in columns if canonical(c) == wanted), None)

def coerce_target(data: pd.DataFrame, requested: str) -> tuple[pd.DataFrame, str]:
    target = find_column(data.columns.tolist(), requested)
    if target is None: raise ValueError(f"Missing target '{requested}'. Columns: {data.columns.tolist()}")
    raw = data[target]
    if raw.dtype == "object":
        mapping = {"0":0,"1":1,"false":0,"true":1,"no":0,"yes":1,
                   "good":0,"bad":1,"tot":0,"xau":1}
        converted = raw.astype(str).map(lambda x: mapping.get(canonical(x), np.nan))
    else: converted = pd.to_numeric(raw, errors="coerce")
    invalid = int((~converted.isin([0, 1])).sum())
    if invalid: LOGGER.warning("Dropping %d rows with missing/non-binary targets", invalid)
    result = data.loc[converted.isin([0, 1])].copy()
    result[target] = converted.loc[result.index].astype("int8")
    if len(result) < 50 or result[target].nunique() != 2:
        raise ValueError("Need at least 50 valid rows and both target classes")
    minority = int(result[target].value_counts().min())
    if minority < 10: raise ValueError(f"Minority class has only {minority} rows; need at least 10")
    return result, target

def add_safe_features(data: pd.DataFrame, as_of_date: str) -> pd.DataFrame:
    result = data.copy()
    birthday = find_column(result.columns.tolist(), "Birthday")
    if birthday:
        dates = pd.to_datetime(result[birthday], errors="coerce", dayfirst=True)
        age = (pd.Timestamp(as_of_date) - dates).dt.days / 365.25
        result["Age"] = age.where(age.between(18, 100))
    amount, salary = (find_column(result.columns.tolist(), x)
                      for x in ("SoTienDKVayBanDau", "Salary"))
    if amount and salary:
        loan = pd.to_numeric(result[amount], errors="coerce")
        income = pd.to_numeric(result[salary], errors="coerce").replace(0, np.nan)
        ratio = (loan / income).replace([np.inf, -np.inf], np.nan)
        result["LoanToIncome"] = ratio.where(ratio.between(0, 1000))
    return result

def choose_group(data: pd.DataFrame, requested: str | None) -> str | None:
    if requested:
        match = find_column(data.columns.tolist(), requested)
        if match is None: raise ValueError(f"Group column '{requested}' not found")
        return match
    wanted = {canonical(x) for x in GROUP_CANDIDATES}
    return next((c for c in data.columns if canonical(c) in wanted and data[c].nunique()>1), None)

def split_indices(data: pd.DataFrame, y: pd.Series, group: str | None):
    idx = np.arange(len(data))
    if group:
        fallback = pd.Series([f"missing_{i}" for i in data.index], index=data.index)
        groups = data[group].where(data[group].notna(), fallback).astype(str)
        outer = GroupShuffleSplit(n_splits=1, test_size=.2, random_state=RANDOM_STATE)
        development, test = next(outer.split(idx, y, groups))
        inner = GroupShuffleSplit(n_splits=1, test_size=.25, random_state=RANDOM_STATE+1)
        tr_rel, va_rel = next(inner.split(development, y.iloc[development], groups.iloc[development]))
        train, validation = development[tr_rel], development[va_rel]
        strategy = "customer_grouped_60_20_20"
    else:
        development, test = train_test_split(idx, test_size=.2, random_state=RANDOM_STATE, stratify=y)
        train, validation = train_test_split(development, test_size=.25,
            random_state=RANDOM_STATE+1, stratify=y.iloc[development])
        strategy = "stratified_random_60_20_20"
    for name, subset in (("train",train),("validation",validation),("test",test)):
        if y.iloc[subset].nunique()!=2: raise ValueError(f"{name} split has only one class")
    return train, validation, test, strategy

def select_features(data: pd.DataFrame, target: str, group: str | None,
                    include_outcome: bool, extras: list[str]) -> tuple[list[str], list[str]]:
    names = set(IDENTIFIER_NAMES) | set(extras) | {target, "Birthday"}
    if not include_outcome: names |= POST_OUTCOME_NAMES
    keys = {canonical(x) for x in names}
    if group: keys.add(canonical(group))
    features, excluded = [], []
    for c in data.columns:
        if canonical(c) in keys or data[c].nunique(dropna=True)<=1 or pd.api.types.is_datetime64_any_dtype(data[c]):
            excluded.append(c)
        else: features.append(c)
    if not features: raise ValueError("No usable features remain after exclusions")
    return features, excluded

def make_preprocessor(frame: pd.DataFrame) -> ColumnTransformer:
    numeric = frame.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [c for c in frame.columns if c not in numeric]
    return ColumnTransformer([
        ("numeric", Pipeline([("imputer",SimpleImputer(strategy="median",add_indicator=True)),
                              ("scaler",StandardScaler())]), numeric),
        ("categorical", Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),
            ("onehot",OneHotEncoder(handle_unknown="ignore",min_frequency=2))]), categorical)],
        remainder="drop")

def candidate_models(frame: pd.DataFrame) -> dict[str, Pipeline]:
    def pipe(model: Any) -> Pipeline:
        return Pipeline([("preprocessor",make_preprocessor(frame)),("classifier",model)])
    return {
        "logistic_regression": pipe(LogisticRegression(class_weight="balanced",max_iter=3000,
                                                        random_state=RANDOM_STATE)),
        "random_forest": pipe(RandomForestClassifier(n_estimators=500,min_samples_leaf=3,
            class_weight="balanced_subsample",n_jobs=-1,random_state=RANDOM_STATE))}

def metrics_at_threshold(y: pd.Series, probability: np.ndarray, threshold: float) -> dict[str, Any]:
    pred = (probability >= threshold).astype(int)
    return {"threshold":float(threshold), "accuracy":float(accuracy_score(y,pred)),
        "balanced_accuracy":float(balanced_accuracy_score(y,pred)),
        "precision_bad_debt":float(precision_score(y,pred,zero_division=0)),
        "recall_bad_debt":float(recall_score(y,pred,zero_division=0)),
        "f1_bad_debt":float(f1_score(y,pred,zero_division=0)),
        "roc_auc":float(roc_auc_score(y,probability)),
        "pr_auc":float(average_precision_score(y,probability)),
        "confusion_matrix":confusion_matrix(y,pred,labels=[0,1]).tolist(),
        "classification_report":classification_report(y,pred,labels=[0,1],
            target_names=["good","bad_debt"],zero_division=0,output_dict=True)}

def choose_threshold(y: pd.Series, probability: np.ndarray, objective: str, floor: float) -> float:
    best_threshold, best_score = .5, -1.
    for threshold in np.linspace(.05,.95,181):
        pred = probability >= threshold
        precision, recall = precision_score(y,pred,zero_division=0), recall_score(y,pred,zero_division=0)
        score = f1_score(y,pred,zero_division=0) if objective=="f1" else recall
        if objective=="recall" and precision<floor: continue
        if score>best_score: best_threshold,best_score=float(threshold),float(score)
    if best_score<0: raise ValueError("No threshold satisfies --min-precision")
    return best_threshold

def save_plots(y: pd.Series, probability: np.ndarray, threshold: float, output: Path) -> None:
    pred = probability >= threshold
    fig, axes = plt.subplots(1,3,figsize=(16,4.5))
    ConfusionMatrixDisplay.from_predictions(y,pred,labels=[0,1],display_labels=["Good","Bad debt"],ax=axes[0],colorbar=False)
    PrecisionRecallDisplay.from_predictions(y,probability,ax=axes[1])
    RocCurveDisplay.from_predictions(y,probability,ax=axes[2])
    fig.suptitle("Final holdout-test evaluation"); fig.tight_layout()
    fig.savefig(output/"evaluation.png",dpi=180,bbox_inches="tight"); plt.close(fig)

def main() -> None:
    args=parse_args(); logging.basicConfig(level=logging.INFO,format="%(levelname)s: %(message)s")
    if not 0<=args.min_precision<=1: raise ValueError("--min-precision must be between 0 and 1")
    pd.Timestamp(args.as_of_date); args.output_dir.mkdir(parents=True,exist_ok=True)
    raw=read_csv_robust(args.input); data,target=coerce_target(raw,args.target)
    data=add_safe_features(data,args.as_of_date).reset_index(drop=True)
    group=choose_group(data,args.group_column)
    features,excluded=select_features(data,target,group,args.include_post_outcome,args.exclude_columns)
    X,y=data[features].copy(),data[target]
    train,val,test,strategy=split_indices(data,y,group)
    Xtr,Xv,Xte=X.iloc[train],X.iloc[val],X.iloc[test]; ytr,yv,yte=y.iloc[train],y.iloc[val],y.iloc[test]
    leaderboard,fitted={},{}
    for name,model in candidate_models(Xtr).items():
        LOGGER.info("Training %s",name); model.fit(Xtr,ytr); probability=model.predict_proba(Xv)[:,1]
        threshold=choose_threshold(yv,probability,args.threshold_objective,args.min_precision)
        leaderboard[name]=metrics_at_threshold(yv,probability,threshold); fitted[name]=model
    best=max(leaderboard,key=lambda n: leaderboard[n]["pr_auc"])
    model=fitted[best]; threshold=float(leaderboard[best]["threshold"])
    probability=model.predict_proba(Xte)[:,1]; test_metrics=metrics_at_threshold(yte,probability,threshold)
    save_plots(yte,probability,threshold,args.output_dir)
    split=SplitInfo(strategy,group,len(train),len(val),len(test))
    metadata={"created_at_utc":datetime.now(timezone.utc).isoformat(),"input_file":args.input.name,
        "rows_used":len(data),"target":target,"target_rate":float(y.mean()),
        "prediction_point":"post_outcome_monitoring" if args.include_post_outcome else "pre_disbursement",
        "as_of_date":args.as_of_date,"features":features,"excluded_columns":excluded,
        "split":asdict(split),"selection_metric":"validation_pr_auc",
        "threshold_objective":args.threshold_objective,"selected_model":best,"threshold":threshold}
    joblib.dump({"model":model,"metadata":metadata},args.output_dir/"bad_debt_model.joblib")
    report={"metadata":metadata,"validation_leaderboard":leaderboard,"test_metrics":test_metrics}
    (args.output_dir/"metrics.json").write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
    pd.DataFrame({"actual":yte.to_numpy(),"probability_bad_debt":probability,
        "predicted":(probability>=threshold).astype(int)}).to_csv(args.output_dir/"test_predictions.csv",index=False)
    LOGGER.info("Selected %s; test PR-AUC=%.4f",best,test_metrics["pr_auc"])
    LOGGER.info("Saved outputs to %s",args.output_dir.resolve())

if __name__ == "__main__": main()
