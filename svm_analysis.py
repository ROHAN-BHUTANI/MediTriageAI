import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
import json
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('meditriage/data/processed/dataset.csv')
train = df[df['split'] == 'train']
test = df[df['split'] == 'test']

vec = TfidfVectorizer()
X_train = vec.fit_transform(train['text'].fillna(''))
X_test = vec.transform(test['text'].fillna(''))

y_train = train['department_code']
y_test = test['department_code']

svm = LinearSVC(random_state=42)
svm.fit(X_train, y_train)
preds = svm.predict(X_test)

labels = sorted(y_test.unique())
cm = confusion_matrix(y_test, preds, labels=labels)

confusions = []
for i, true_label in enumerate(labels):
    for j, pred_label in enumerate(labels):
        if i != j and cm[i, j] > 0:
            confusions.append({'true': true_label, 'pred': pred_label, 'count': int(cm[i, j])})

confusions = sorted(confusions, key=lambda x: x['count'], reverse=True)[:5]

examples = {}
test_df = test.copy()
test_df['pred'] = preds
for c in confusions:
    pair_df = test_df[(test_df['department_code'] == c['true']) & (test_df['pred'] == c['pred'])]
    samples = pair_df['text'].head(3).tolist()
    examples[f"{c['true']} -> {c['pred']}"] = samples

out = {'labels': list(labels), 'cm': cm.tolist(), 'top_pairs': confusions, 'examples': examples}
with open('svm_analysis.json', 'w', encoding='utf-8') as f:
    json.dump(out, f, indent=2)
print('Analysis complete')
