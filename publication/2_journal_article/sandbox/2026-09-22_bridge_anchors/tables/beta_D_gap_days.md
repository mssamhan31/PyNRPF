### beta_D labelled RPF days whose span abuts a missing reading

`truth` is the labelled span (slots, inclusive); `gap` says which side of the span has a missing reading next to it. Under each rule: the best window, or in brackets the best window when the null won; evidence `r`; held-out `p`; decision (AC = AUTO_CORRECT, AK = AUTO_KEEP, UNC = UNCERTAIN); and whether the shown window equals the truth exactly, has slot IoU ≥ 0.8, or its IoU otherwise.

| date | conf | truth | gap | nearest window | nearest r | nearest p | nearest dec | nearest match | edge window | edge r | edge p | edge dec | edge match | gap_edge window | gap_edge r | gap_edge p | gap_edge dec | gap_edge match |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2023-10-05 | sure | 45–60 | before | 45–62 | 32.5 | 1.0 | AC | IoU≥0.8 | 45–67 | 20.0 | 0.5 | UNC | IoU 0.70 | 45–66 | 20.4 | 0.9 | AC | IoU 0.73 |
| 2023-10-06 | sure | 46–59 | before | 46–62 | 28.6 | 0.9 | AC | IoU≥0.8 | 46–62 | 14.2 | 0.3 | AK | IoU≥0.8 | 46–61 | 13.8 | 0.8 | AC | IoU≥0.8 |
| 2023-10-11 | sure | 51–59 | before | 51–67 | 25.2 | 0.9 | AC | IoU 0.53 | 51–62 | 4.7 | 0.0 | AK | IoU 0.75 | 51–67 | 6.2 | 0.4 | UNC | IoU 0.53 |
| 2023-10-12 | sure | 45–52 | both | 45–52 | 5.3 | 0.4 | UNC | exact | 45–52 | 1.1 | 0.0 | AK | exact | 45–52 | 1.1 | 0.1 | AK | exact |
| 2023-10-14 | sure | 48–55 | both | 48–51 | 5.9 | 0.4 | UNC | IoU 0.50 | 48–55 | 0.9 | 0.0 | AK | exact | 48–55 | 0.9 | 0.0 | AK | exact |
| 2023-10-20 | sure | 46–58 | both | 46–58 | 13.4 | 0.8 | AC | exact | 53–55 | 4.6 | 0.0 | AK | IoU 0.23 | 46–58 | 3.6 | 0.2 | AK | exact |
| 2023-10-23 | sure | 44–58 | both | 44–58 | 18.8 | 0.9 | AC | exact | 44–55 | 5.5 | 0.0 | AK | IoU≥0.8 | 44–58 | 5.5 | 0.4 | UNC | exact |
| 2023-11-10 | sure | 43–56 | both | 47–56 | 6.8 | 0.5 | UNC | IoU 0.71 | 30–33 | 4.7 | 0.0 | AK | IoU 0.00 | 47–56 | 3.6 | 0.2 | AK | IoU 0.71 |
| 2023-11-18 | sure | 47–59 | both | 47–59 | 14.8 | 0.8 | AC | exact | 28–32 | 6.7 | 0.0 | AK | IoU 0.00 | 47–59 | 2.5 | 0.1 | AK | exact |
| 2023-12-22 | sure | 48–59 | after | 48–59 | 15.5 | 0.8 | AC | exact | 48–59 | 12.6 | 0.2 | AK | exact | 49–59 | 12.4 | 0.7 | AC | IoU≥0.8 |
| 2024-04-11 | sure | 45–52 | both | 45–52 | 7.6 | 0.5 | UNC | exact | 45–52 | 1.4 | 0.0 | AK | exact | 45–52 | 1.4 | 0.1 | AK | exact |
| 2024-08-22 | sure | 49–54 | after | 45–54 | 11.0 | 0.7 | UNC | IoU 0.60 | 47–54 | 7.9 | 0.0 | AK | IoU 0.75 | 43–54 | 9.2 | 0.6 | UNC | IoU 0.50 |
| 2024-08-26 | sure | 46–54 | both | 46–54 | 4.6 | 0.3 | UNC | exact | 46–54 | 5.0 | 0.0 | AK | exact | 46–54 | 5.0 | 0.3 | UNC | exact |
| 2024-08-30 | sure | 45–52 | both | 45–52 | 6.7 | 0.5 | UNC | exact | 45–52 | 1.3 | 0.0 | AK | exact | 45–52 | 1.3 | 0.1 | AK | exact |
| 2024-08-31 | sure | 47–50 | both | 47–50 | 0.9 | 0.0 | AK | exact | 27–29 | 0.2 | 0.0 | AK | IoU 0.00 | (47–50) | -0.3 | 0.0 | AK | exact |
| 2024-09-02 | sure | 43–55 | before | 43–57 | 24.9 | 0.9 | AC | IoU≥0.8 | 43–61 | 16.4 | 0.4 | UNC | IoU 0.68 | 43–63 | 18.0 | 0.9 | AC | IoU 0.62 |
| 2024-09-17 | sure | 41–54 | after | 34–54 | 29.3 | 0.9 | AC | IoU 0.67 | 34–54 | 15.2 | 0.3 | UNC | IoU 0.67 | 35–54 | 15.3 | 0.8 | AC | IoU 0.70 |
| 2024-09-21 | sure | 44–56 | before | 44–57 | 23.2 | 0.9 | AC | IoU≥0.8 | 44–62 | 14.1 | 0.2 | AK | IoU 0.68 | 44–61 | 14.1 | 0.8 | AC | IoU 0.72 |
| 2024-09-22 | sure | 44–54 | both | 44–54 | 12.0 | 0.7 | AC | exact | 44–54 | 4.2 | 0.0 | AK | exact | 44–54 | 4.2 | 0.3 | AK | exact |
| 2023-10-04 | unsure | 49–60 | before | 49–56 | 5.1 | 0.3 | UNC | IoU 0.67 | 51–59 | 5.5 | 0.0 | AK | IoU 0.75 | 49–56 | 3.6 | 0.2 | AK | IoU 0.67 |
| 2023-10-07 | unsure | 55–57 | after | 43–57 | 2.9 | 0.2 | AK | IoU 0.20 | 48–52 | 2.4 | 0.0 | AK | IoU 0.00 | 52–53 | 2.1 | 0.1 | AK | IoU 0.00 |
| 2023-10-08 | unsure | 53–57 | both | 53–57 | 1.2 | 0.1 | AK | exact | 32–35 | 2.7 | 0.0 | AK | IoU 0.00 | 53–57 | 0.4 | 0.0 | AK | exact |
| 2023-10-16 | unsure | 45–58 | before | 45–61 | 12.8 | 0.8 | AC | IoU≥0.8 | 45–64 | 7.1 | 0.0 | AK | IoU 0.70 | 45–62 | 7.9 | 0.5 | UNC | IoU 0.78 |
| 2023-10-19 | unsure | 58–60 | before | 61–64 | 7.9 | 0.6 | UNC | IoU 0.00 | 40–46 | 1.1 | 0.0 | AK | IoU 0.00 | 61–64 | 7.9 | 0.6 | UNC | IoU 0.00 |
| 2023-10-24 | unsure | 46–55 | both | 46–55 | 10.2 | 0.7 | UNC | exact | 46–55 | 7.6 | 0.0 | AK | exact | 46–55 | 7.6 | 0.5 | UNC | exact |
| 2023-10-29 | unsure | 48–57 | both | 48–57 | 4.5 | 0.3 | UNC | exact | 48–57 | 2.6 | 0.0 | AK | exact | 48–57 | 2.6 | 0.2 | AK | exact |
| 2023-10-30 | unsure | 44–54 | both | 44–54 | 9.6 | 0.6 | UNC | exact | 44–54 | 2.9 | 0.0 | AK | exact | 44–54 | 2.9 | 0.2 | AK | exact |
| 2023-11-03 | unsure | 46–59 | after | 36–59 | 20.9 | 0.9 | AC | IoU 0.58 | 35–59 | 13.0 | 0.2 | AK | IoU 0.56 | 36–59 | 13.6 | 0.8 | AC | IoU 0.58 |
| 2023-11-14 | unsure | 40–54 | after | 37–54 | 26.0 | 0.9 | AC | IoU≥0.8 | 33–54 | 18.6 | 0.5 | UNC | IoU 0.68 | 34–54 | 19.6 | 0.9 | AC | IoU 0.71 |
| 2023-11-19 | unsure | 49–53 | after | 48–53 | 3.3 | 0.2 | AK | IoU≥0.8 | 47–53 | 0.4 | 0.0 | AK | IoU 0.71 | 47–53 | 0.4 | 0.0 | AK | IoU 0.71 |
| 2023-12-04 | unsure | 54–56 | before | 57–59 | 3.4 | 0.2 | AK | IoU 0.00 | 33–38 | 2.3 | 0.0 | AK | IoU 0.00 | 34–38 | 1.6 | 0.1 | AK | IoU 0.00 |
| 2023-12-07 | unsure | 55–58 | before | 55–60 | 6.3 | 0.4 | UNC | IoU 0.67 | 58–67 | 4.1 | 0.0 | AK | IoU 0.08 | 55–59 | 1.3 | 0.1 | AK | IoU≥0.8 |
| 2024-02-21 | unsure | 51–55 | before | 51–64 | 8.3 | 0.6 | UNC | IoU 0.36 | 68–71 | 2.2 | 0.0 | AK | IoU 0.00 | 51–61 | 4.6 | 0.3 | UNC | IoU 0.45 |
| 2024-02-22 | unsure | 42–44 | after | 41–44 | 3.1 | 0.2 | AK | IoU 0.75 | 40–44 | 1.7 | 0.0 | AK | IoU 0.60 | 41–44 | 1.7 | 0.1 | AK | IoU 0.75 |
| 2024-04-09 | unsure | 46–50 | before | 46–62 | 7.1 | 0.5 | UNC | IoU 0.29 | 46–62 | 6.8 | 0.0 | AK | IoU 0.29 | 46–61 | 6.1 | 0.4 | UNC | IoU 0.31 |
| 2024-04-16 | unsure | 49–52 | before | 52–55 | 7.1 | 0.5 | UNC | IoU 0.14 | 35–38 | 2.4 | 0.0 | AK | IoU 0.00 | 52–55 | 7.1 | 0.5 | UNC | IoU 0.14 |
| 2024-05-16 | unsure | 44–48 | after | 41–42 | 3.9 | 0.3 | AK | IoU 0.00 | 38–48 | 2.2 | 0.0 | AK | IoU 0.45 | 41–42 | 3.9 | 0.3 | AK | IoU 0.00 |
| 2024-09-14 | unsure | 48–52 | both | 48–52 | 1.5 | 0.1 | AK | exact | 48–52 | 0.4 | 0.0 | AK | exact | 48–52 | 0.4 | 0.0 | AK | exact |

### Summary over those days

| confidence | anchors | days | auto_correct | uncertain | exact | iou_ge_0_8 | median_r | median_p |
|---|---|---|---|---|---|---|---|---|
| sure | nearest | 19 | 11 | 7 | 10 | 14 | 13.44 | 0.77 |
| sure | edge | 19 | 0 | 3 | 7 | 9 | 5.02 | 0.01 |
| sure | gap_edge | 19 | 6 | 4 | 10 | 12 | 5.02 | 0.35 |
| unsure | nearest | 19 | 3 | 9 | 5 | 8 | 6.26 | 0.44 |
| unsure | edge | 19 | 0 | 1 | 4 | 4 | 2.60 | 0.00 |
| unsure | gap_edge | 19 | 2 | 6 | 5 | 6 | 3.55 | 0.23 |
