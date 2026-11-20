"""
SMS Spam Inference & Prediction CLI
===================================

Loads the trained machine learning model and TF-IDF vectorizer from `artifacts/`
and performs real-time classification on arbitrary SMS messages.

Outputs:
- Preprocessed token sequence (entity masks, stemmed tokens)
- Binary classification: [SPAM] (Smishing / Phishing / Fraud) vs [HAM] (Legitimate)
- Calibrated confidence percentage & decision score
- Smishing risk tier and actionable entity detection breakdown

CLI Usage:
    python src/predict.py "Claim your £1000 cash prize now! Call 08000930705"
    python src/predict.py "Hey, are we still meeting for lunch today?"
    python src/predict.py --interactive
    python src/predict.py --file messages.txt
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, Optional, Tuple

import joblib

# Ensure project root is in sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.preprocess import mask_entities, preprocess_text

ARTIFACTS_DIR = os.path.join(project_root, "artifacts")
DEFAULT_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "best_model.joblib")
DEFAULT_VECTORIZER_PATH = os.path.join(ARTIFACTS_DIR, "tfidf_vectorizer.joblib")
DEFAULT_METADATA_PATH = os.path.join(ARTIFACTS_DIR, "model_metadata.json")


def load_artifacts(
    model_name: str = "best",
    model_path: Optional[str] = None,
    vectorizer_path: str = DEFAULT_VECTORIZER_PATH,
    metadata_path: str = DEFAULT_METADATA_PATH,
) -> Tuple[Any, Any, Optional[Dict[str, Any]]]:
    """
    Load serialized model, TF-IDF vectorizer, and metadata from disk.
    If artifacts are missing, raises a descriptive FileNotFoundError.
    """
    if model_path is None:
        if model_name.lower() in ["mnb", "nb", "multinomial_nb"]:
            model_path = os.path.join(ARTIFACTS_DIR, "multinomial_nb.joblib")
        elif model_name.lower() in ["raw", "linear_svc", "svc"]:
            model_path = os.path.join(ARTIFACTS_DIR, "linear_svc_raw.joblib")
        else:
            model_path = DEFAULT_MODEL_PATH

    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        err_msg = (
            f"\n[ERROR] Model artifacts not found in '{ARTIFACTS_DIR}'.\n"
            "Please train the model first by running:\n"
            "    python src/train.py\n"
        )
        raise FileNotFoundError(err_msg)

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    metadata = None
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception:
            metadata = None

    return model, vectorizer, metadata


def predict_message(
    raw_message: str,
    model: Any,
    vectorizer: Any,
) -> Dict[str, Any]:
    """
    Classify a single SMS message as HAM or SPAM with confidence and risk profiling.

    Args:
        raw_message (str): Raw input SMS message.
        model: Trained scikit-learn model / calibrated pipeline.
        vectorizer: Fitted TfidfVectorizer.

    Returns:
        Dict[str, Any]: Prediction results including label, confidence, score, and token masks.
    """
    masked_message = mask_entities(raw_message)
    cleaned_tokens = preprocess_text(raw_message)

    # Transform through vectorizer
    X_vec = vectorizer.transform([cleaned_tokens])

    # Predict label
    pred_int = int(model.predict(X_vec)[0])
    label = "SPAM" if pred_int == 1 else "HAM"

    # Compute probability & decision score
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_vec)[0]
        spam_prob = float(probabilities[1])
        ham_prob = float(probabilities[0])
        confidence = spam_prob if label == "SPAM" else ham_prob
    elif hasattr(model, "decision_function"):
        margin = float(model.decision_function(X_vec)[0])
        # Sigmoid calibration fallback
        import numpy as np
        spam_prob = float(1.0 / (1.0 + np.exp(-margin)))
        ham_prob = 1.0 - spam_prob
        confidence = spam_prob if label == "SPAM" else ham_prob
    else:
        spam_prob = 1.0 if label == "SPAM" else 0.0
        confidence = 1.0

    # Risk tiering
    if label == "SPAM":
        if confidence >= 0.90:
            risk_tier = "CRITICAL (High Smishing Threat)"
        elif confidence >= 0.70:
            risk_tier = "HIGH (Suspicious Phishing / Promo)"
        else:
            risk_tier = "MODERATE (Ambiguous Commercial Content)"
    else:
        if confidence >= 0.95:
            risk_tier = "SAFE (Legitimate Transactional / Personal)"
        else:
            risk_tier = "LOW RISK (Probable Genuine Message)"

    # Identify semantic entities triggered
    detected_entities = []
    if "httpaddr" in masked_message:
        detected_entities.append("Hyperlink / URL (httpaddr)")
    if "moneysymb" in masked_message:
        detected_entities.append("Financial Currency (moneysymb)")
    if "phonenum" in masked_message:
        detected_entities.append("Phone Sequence (phonenum)")
    if "numbr" in masked_message:
        detected_entities.append("Numeric Quantities (numbr)")
    if "emailaddr" in masked_message:
        detected_entities.append("Email Address (emailaddr)")

    return {
        "raw_message": raw_message,
        "masked_message": masked_message.strip(),
        "cleaned_tokens": cleaned_tokens,
        "label": label,
        "is_spam": bool(label == "SPAM"),
        "confidence": confidence,
        "spam_probability": spam_prob,
        "risk_tier": risk_tier,
        "detected_entities": detected_entities,
    }


def format_cli_output(result: Dict[str, Any]) -> str:
    """
    Format the prediction result into a readable ASCII terminal display.
    """
    sep = "=" * 70
    label = result["label"]
    badge = f" [ {label} ] "

    # Terminal styling tags
    tag = "[!!! SPAM / SMISHING DETECTED !!!]" if label == "SPAM" else "[✓ LEGITIMATE MESSAGE (HAM)]"

    lines = [
        sep,
        f" SMS CLASSIFICATION RESULT: {badge}  {tag}",
        sep,
        f"  Input SMS         : \"{result['raw_message']}\"",
        f"  Normalized Tokens : {result['cleaned_tokens'] if result['cleaned_tokens'] else '(none)'}",
        f"  Detected Entities : {', '.join(result['detected_entities']) if result['detected_entities'] else 'None'}",
        "-" * 70,
        f"  Classification    : {label}",
        f"  Confidence Score  : {result['confidence'] * 100:.2f}%",
        f"  Spam Probability  : {result['spam_probability'] * 100:.2f}%",
        f"  Assessed Risk     : {result['risk_tier']}",
        sep,
    ]
    return "\n".join(lines)


def interactive_mode(model: Any, vectorizer: Any):
    """
    Interactive REPL to evaluate messages interactively.
    """
    print("\n" + "=" * 70)
    print(" SMS SPAM DETECTOR: INTERACTIVE MODE")
    print(" Type an SMS message and press Enter. Type 'quit' or 'exit' to leave.")
    print("=" * 70 + "\n")

    while True:
        try:
            user_msg = input("Enter SMS > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting interactive mode. Goodbye!")
            break

        if not user_msg:
            continue
        if user_msg.lower() in ["exit", "quit", "q"]:
            print("Exiting interactive mode. Goodbye!")
            break

        res = predict_message(user_msg, model, vectorizer)
        print("\n" + format_cli_output(res) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Classify an SMS message as SPAM or HAM using the trained ML model."
    )
    parser.add_argument(
        "message",
        nargs="?",
        default=None,
        help="The SMS message string to classify (enclosed in quotes).",
    )
    parser.add_argument(
        "--message",
        "-m",
        dest="opt_message",
        default=None,
        help="Alternative flag to supply the SMS message.",
    )
    parser.add_argument(
        "--file",
        "-f",
        dest="file_path",
        default=None,
        help="Path to a text file containing SMS messages (one per line).",
    )
    parser.add_argument(
        "--model",
        default="best",
        choices=["best", "linear_svc", "multinomial_nb"],
        help="Which model artifact to load: 'best' (Calibrated Linear SVM), 'linear_svc' (Raw Linear SVM), or 'multinomial_nb' (Multinomial Naive Bayes).",
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Run an interactive command-line session to test multiple messages.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON instead of human-readable text.",
    )

    args = parser.parse_args()

    # Determine message input
    input_text = args.message or args.opt_message

    # Auto-train check
    if not os.path.exists(DEFAULT_MODEL_PATH) or not os.path.exists(DEFAULT_VECTORIZER_PATH):
        print(f"[INFO] Saved model artifacts not found at '{ARTIFACTS_DIR}'.")
        print("[INFO] Initiating automated model training to build artifacts...")
        from src.train import train_and_benchmark
        train_and_benchmark()

    model, vectorizer, _ = load_artifacts(model_name=args.model)

    if args.interactive:
        interactive_mode(model, vectorizer)
        return

    if args.file_path:
        if not os.path.exists(args.file_path):
            print(f"[ERROR] File not found: {args.file_path}")
            sys.exit(1)
        with open(args.file_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = [line.strip() for line in f if line.strip()]

        print(f"\nProcessing {len(lines)} messages from '{args.file_path}':\n")
        results = [predict_message(msg, model, vectorizer) for msg in lines]
        if args.json:
            print(json.dumps(results, indent=2))
        else:
            for r in results:
                print(format_cli_output(r))
                print()
        return

    if not input_text:
        # Default demonstration if no arguments provided
        demo_msg = "URGENT! You have won a £1,000 cash prize! Call 09061701461 to claim your reward now."
        print(f"[INFO] No message argument provided. Running demo prediction on:\n  \"{demo_msg}\"\n")
        input_text = demo_msg

    result = predict_message(input_text, model, vectorizer)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(format_cli_output(result))


if __name__ == "__main__":
    main()
