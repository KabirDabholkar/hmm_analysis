from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ValidationCurveDisplay
import numpy as np

X, y = make_classification(1000, 10, random_state=0, n_informative=3, n_classes = 3)

# _ = ValidationCurveDisplay.from_estimator(
#     LogisticRegression(),
#     X,
#     y,
#     param_name="C",
#     param_range=np.geomspace(1e-5, 1e3, num=9),
#     score_type="both",
#     score_name="Accuracy",
# )

# print(y[:5])

LR = LogisticRegression().fit(X[:700],y[:700])
y_pred = LR.predict_proba(X[700:])
print(y_pred.shape)
print(
    np.stack([np.random.choice(y_pred.shape[1],p=y_pred[i]) for i in range(len(y_pred))])
)

