import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, NearMiss
from imblearn.over_sampling import RandomOverSampler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

data = pd.read_csv("Creditcard_data.csv")


print(data.head())
print(data['Class'].value_counts())



X = data.drop('Class', axis=1)
y = data['Class']

smote = SMOTE(random_state=42)
X_bal, y_bal = smote.fit_resample(X, y)

print(y_bal.value_counts())



samples = []
X_temp, y_temp = X_bal, y_bal

for i in range(5):
    X_part, X_temp, y_part, y_temp = train_test_split(
        X_temp, y_temp, test_size=0.8, random_state=i
    )
    samples.append((X_part, y_part))


samplers = {
    "Sampling1": RandomUnderSampler(),
    "Sampling2": RandomOverSampler(),
    "Sampling3": SMOTE(),
    "Sampling4": TomekLinks(),
    "Sampling5": NearMiss()
}



models = {
    "M1": LogisticRegression(max_iter=1000),
    "M2": DecisionTreeClassifier(),
    "M3": RandomForestClassifier(),
    "M4": KNeighborsClassifier(),
    "M5": GaussianNB()
}



results = pd.DataFrame(index=models.keys(), columns=samplers.keys())

for m_name, model in models.items():
    for s_name, sampler in samplers.items():
        X_s, y_s = sampler.fit_resample(X_bal, y_bal)

        X_train, X_test, y_train, y_test = train_test_split(
            X_s, y_s, test_size=0.3, random_state=42
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred) * 100
        results.loc[m_name, s_name] = round(acc, 2)

print("\nAccuracy Table:\n")
print(results)

best = results.stack().astype(float).idxmax()
best_value = results.stack().max()

print("\nBest Model + Sampling:")
print(best)
print("Best Accuracy:", best_value)

results.to_csv("accuracy_results.csv")
