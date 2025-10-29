
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer   
from src.models.text_classifier import TextClassifier

texts = [
    "This movie is fantastic and I love it!",
    "I hate this film, it's terrible.",
    "The acting was superb, a truly great experience.",
    "What a waste of time, absolutely boring.",
    "Highly recommend this, a masterpiece.",
    "Could not finish watching, so bad."
]
labels = [1, 0, 1, 0, 1, 0]

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

print(f"Train: {len(X_train)}  Test: {len(X_test)}")

# ---- sklearn TF‑IDF -------------------------------------------------
vectorizer = TfidfVectorizer(
    lowercase=True,
    token_pattern=r'\b\w+\b',     
    norm='l2'                      
)

clf = TextClassifier(vectorizer)
clf.fit(X_train, y_train)

pred = clf.predict(X_test)
metrics = clf.evaluate(y_test, pred)

print("\n=== EVALUATION RESULTS ===")
for k, v in metrics.items():
    print(f"{k:10}: {v:.4f}")

print("\nPredictions vs True labels:")
for txt, true, p in zip(X_test, y_test, pred):
    print(f"Text: {txt}")
    print(f"   True: {'POSITIVE' if true else 'NEGATIVE'} | "
          f"Pred: {'POSITIVE' if p else 'NEGATIVE'}\n")