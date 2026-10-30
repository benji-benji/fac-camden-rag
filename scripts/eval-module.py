## Each tuple is (actual_label, predicted_label)
results = [
    ("spam",     "spam"),      # TP
    ("spam",     "spam"),      # TP
    ("spam",     "spam"),      # TP
    ("spam",     "spam"),  # TP
    ("spam",     "spam"),  # TP
    ("not spam", "not spam"),  # TN
    ("not spam", "not spam"),  # TN
    ("not spam", "not spam"),  # TN
    ("not spam", "not spam"),  # TN
    ("not spam", "not spam"),  # TN
    ("not spam", "not spam"),  # TN
    ("not spam", "not spam"),  # TN
    ("not spam", "not spam"),  # TN
    ("not spam", "not spam"),  # TN
    ("not spam", "not spam"),  # TN
    ("not spam", "not spam"),  # TN
    ("not spam", "spam"),      # FP
    ("not spam", "spam"),      # FP
    ("not spam", "spam"),      # FP
    ("not spam", "spam"),      # FP
]

tp = sum(1 for actual, predicted in results if actual == "spam" and predicted == "spam")
fp = sum(1 for actual, predicted in results if actual == "not spam" and predicted == "spam")
tn = sum(1 for actual, predicted in results if actual == "not spam" and predicted == "not spam")
fn = sum(1 for actual, predicted in results if actual == "spam" and predicted == "not spam")

print (f"True positives: {tp}")
print (f"False positives: {fp}")
print (f"True negatives: {tn}")
print (f"False negative : {fn}")

accuracy = (tp + tn) / len(results)
precision = tp / (tp +fp)
recall = tp / (tp+fn)
f1 = 2 * ((accuracy * precision)/(accuracy+precision))

print (f"Accuracy: {accuracy:.2f}")
print (f"Precision: {precision:.2f}")
print (f"Recall: {recall:.2f}")
print (f"F1: {f1:.2f}")

# perfect recall - flag everything as spam
# perfect precision - only flag an email as spam if we are 100% sure its spam

## Simulate the "flag everything" approach
recall_all = 5 / 5  # caught all 5 spam
precision_all = 5 / 20  # but flagged all 20 emails

print(f"Flag everything:  precision={precision_all:.2f}, recall={recall_all:.2f}")
print(f"Our classifier:   precision={precision:.2f}, recall={recall:.2f}")

# confusion matrix 

print(f"\n **CONFUSION MATRIX** \n\n     TP | FN \n     {tp}  |  {fn} \n     -------\n     FP | TN \n     {fp}  | {tn} \n")

## Our classifier
print(f"Our classifier:   F1 = {2 * (0.60 * 0.60) / (0.60 + 0.60):.2f}")

## Flag everything: high recall, low precision
print(f"Flag everything:  F1 = {2 * (0.25 * 1.00) / (0.25 + 1.00):.2f}")

## Very cautious: high precision, low recall
print(f"Very cautious:    F1 = {2 * (1.00 * 0.20) / (1.00 + 0.20):.2f}")

## Perfect system
print(f"Perfect:          F1 = {2 * (1.00 * 1.00) / (1.00 + 1.00):.2f}")

def compute_precision(tp, fp):
    if tp + fp == 0:
        return 0.0
    return tp / (tp + fp)

def compute_recall(tp, fn):
    if tp + fn == 0:
        return 0.0
    return tp / (tp + fn)

def compute_f1(precision, recall):
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

def evaluate(results, positive_label="spam"):
    tp = sum(1 for actual, predicted in results if actual == positive_label and predicted == positive_label)
    fp = sum(1 for actual, predicted in results if actual != positive_label and predicted == positive_label)
    fn = sum(1 for actual, predicted in results if actual == positive_label and predicted != positive_label)

    precision = compute_precision(tp, fp)
    recall = compute_recall(tp, fn)
    f1 = compute_f1(precision, recall)

    return {"precision": precision, "recall": recall, "f1": f1}

metrics = evaluate(results, positive_label="spam")
print(f"Precision: {metrics['precision']:.2f}")
print(f"Recall:    {metrics['recall']:.2f}")
print(f"F1:        {metrics['f1']:.2f}")
