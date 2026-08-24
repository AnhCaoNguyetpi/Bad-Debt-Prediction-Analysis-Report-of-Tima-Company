# Data dictionary and prediction contract

Input is a CSV with one row per loan application. Column matching ignores accents,
spaces, punctuation and case.

## Required field

| Field | Type | Meaning |
|---|---|---|
| `HasBadDebt` | binary | Target: `1` is bad debt; `0` is good debt. |

Accepted labels also include `true/false`, `yes/no`, `bad/good`, and `xấu/tốt`.

## Typical application-time fields

| Field | Type | Meaning |
|---|---|---|
| `TS_CREDIT_SCORE_V2` | numeric | Credit score at application time. |
| `SoTienDKVayBanDau` | numeric | Requested loan amount. |
| `Salary` | numeric | Verified monthly income in one unit. |
| `Birthday` | date | Used to calculate age at the configured reference date. |
| `NumberOfLoans` | numeric | Prior-loan count known at application time. |
| `Gender`, `JobName`, `ProductCreditName` | category | Applicant/product information. |

Identifiers and post-outcome fields such as late payment, overdue days, disbursed
amount, outstanding balance and payment status are excluded automatically. Add any
project-specific proxy with `--exclude-columns`.

Before production, document the target observation window, verify that every feature
existed at decision time, use one unit for all money values, review protected
attributes such as gender, and never commit personal customer data.
