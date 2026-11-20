# SMS Spam Collection Dataset

This directory contains the canonical SMS Spam Collection dataset used for training, cross-validation, and benchmarking the machine learning models.

---

## Dataset Overview

The SMS Spam Collection is a public, curated set of SMS tagged messages that have been collected for SMS Spam research. It contains a total of **5,574 English, real, and non-encoded messages**, tagged according to being legitimate (`ham`) or spam (`spam`).

### Class Distribution

| Class Label | Designation | Number of Messages | Percentage | Class Ratio |
|:------------|:------------|:-------------------|:-----------|:------------|
| **`ham`**   | Legitimate  | **4,827**          | **86.60%** | Baseline    |
| **`spam`**  | Malicious   | **747**            | **13.40%** | **6.46 : 1**|
| **Total**   | All Records | **5,574**          | **100.0%** | —           |

The intrinsic **6.46 : 1 class imbalance** mirrors the real-world operational reality of cellular telecommunications networks, where genuine user messages substantially outnumber unsolicited spam broadcasts.

---

## Dataset File Location & Structure

Place the primary dataset file at:
```
data/spam.csv
```

### File Schema

The CSV contains two core columns:

| Column Header | Data Type | Description | Example |
|:--------------|:----------|:------------|:--------|
| `v1` (or `label`) | `string` | Binary ground-truth category (`ham` or `spam`) | `ham` / `spam` |
| `v2` (or `text`)  | `string` | Raw text payload of the SMS message | `"URGENT! Call 09061701461 to claim your £1000 prize..."` |

Any extra trailing columns (such as `Unnamed: 2`, `Unnamed: 3`, `Unnamed: 4` sometimes present in Kaggle exports) are automatically stripped by the data ingestion pipeline in `src/train.py`.

---

## Automatic Download & Offline Fallback

The ingestion pipeline in [`src/train.py`](../src/train.py) is engineered to never crash on first run:

1. **Local Check**: Checks if `data/spam.csv` exists.
2. **Automated Download**: If `data/spam.csv` is absent, it automatically pulls and extracts the canonical dataset archive from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/228/sms+spam+collection).
3. **Offline Fallback Generator**: If running in an air-gapped or offline environment without internet access, it automatically generates a local synthetic dataset with representative legitimate transactional/conversational messages and smishing/lottery scams, ensuring reproducibility across all evaluation tiers.

---

## Provenance & Attribution

The dataset was compiled by Tiago A. Almeida, José María Gómez Hidalgo, and Akebo Yamakami by aggregating authentic SMS messages from four distinct sources:
1. **Grumbled SMS Corpus**: 1,002 English SMS spam messages collected from public complaints.
2. **Caroline Tag's PhD Thesis**: 1,271 legitimate ham messages collected from volunteer mobile users.
3. **Singapore NUS SMS Corpus**: 1,848 legitimate ham messages collected from Singaporean students.
4. **UK Mobile Spam Corpus**: 425 SMS spam messages manually gathered from UK consumer forums.

### Academic Citation

```bibtex
@inproceedings{almeida2011contributions,
  title={Contributions to the study of SMS spam filtering: New collection and results},
  author={Almeida, Tiago A and Hidalgo, Jos{\'e} Mar{\'\i}a G and Yamakami, Akebo},
  booktitle={Proceedings of the 11th ACM Symposium on Document Engineering (DocEng '11)},
  pages={259--262},
  year={2011},
  publisher={ACM}
}
```
