
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score

# -----------------------------------
# 1. Load Dataset
# -----------------------------------

data = pd.read_csv("student_performance.csv")

# Create performance categories
def get_category(marks):
    if marks >= 85:
        return "Excellent"
    elif marks >= 70:
        return "Good"
    elif marks >= 50:
        return "Average"
    else:
        return "Needs Improvement"


data["performance"] = data["final_marks"].apply(get_category)

# -----------------------------------
# 2. Select Input Features
# -----------------------------------

features = [
    "attendance",
    "study_hours",
    "assignment_score",
    "previous_marks"
]

X = data[features]
y_regression = data["final_marks"]
y_classification = data["performance"]

# -----------------------------------
# 3. Split Dataset
# -----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y_regression, test_size=0.2, random_state=42
)

X_train_c, X_test_c, yc_train, yc_test = train_test_split(
    X, y_classification, test_size=0.2, random_state=42,
    stratify=y_classification
)

# -----------------------------------
# 4. Train Regression Model
# -----------------------------------

regression_model = LinearRegression()
regression_model.fit(X_train, y_train)

regression_predictions = regression_model.predict(X_test)

mae = mean_absolute_error(y_test, regression_predictions)
r2 = r2_score(y_test, regression_predictions)

# -----------------------------------
# 5. Train Classification Model
# -----------------------------------

classification_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

classification_model.fit(X_train_c, yc_train)

classification_predictions = classification_model.predict(X_test_c)

accuracy = accuracy_score(yc_test, classification_predictions)

# -----------------------------------
# 6. Take Student Input
# -----------------------------------

print("\n======================================")
print(" AI-BASED STUDENT PERFORMANCE PREDICTOR")
print("======================================")

print("\nEnter Student Details:")

attendance = float(input("Attendance percentage (0-100): "))
study_hours = float(input("Daily study hours: "))
assignment_score = float(input("Assignment score (0-100): "))
previous_marks = float(input("Previous exam marks (0-100): "))

# Validate input
if not (0 <= attendance <= 100):
    raise ValueError("Attendance must be between 0 and 100.")

if not (0 <= assignment_score <= 100):
    raise ValueError("Assignment score must be between 0 and 100.")

if not (0 <= previous_marks <= 100):
    raise ValueError("Previous marks must be between 0 and 100.")

if not (0 <= study_hours <= 24):
    raise ValueError("Study hours must be between 0 and 24.")

student = pd.DataFrame([{
    "attendance": attendance,
    "study_hours": study_hours,
    "assignment_score": assignment_score,
    "previous_marks": previous_marks
}])

# -----------------------------------
# 7. Predict Performance
# -----------------------------------

predicted_marks = regression_model.predict(student)[0]
predicted_marks = np.clip(predicted_marks, 0, 100)

predicted_category = classification_model.predict(student)[0]

print("\n========== PREDICTION RESULT ==========")

print(f"Predicted Final Marks: {predicted_marks:.2f}/100")
print(f"Performance Category: {predicted_category}")

# -----------------------------------
# 8. Weak Subject Identification
# -----------------------------------

print("\nEnter Subject Marks:")

maths = float(input("Mathematics marks: "))
science = float(input("Science marks: "))
english = float(input("English marks: "))

subject_marks = {
    "Mathematics": maths,
    "Science": science,
    "English": english
}

if any(not 0 <= mark <= 100 for mark in subject_marks.values()):
    raise ValueError("Subject marks must be between 0 and 100.")

weak_subject = min(subject_marks, key=subject_marks.get)

print("\n========== SUBJECT ANALYSIS ==========")

print("Weakest Subject:", weak_subject)

for subject, marks in subject_marks.items():
    if marks < 50:
        print(f"{subject}: Needs urgent improvement.")
    elif marks < 70:
        print(f"{subject}: Practice more questions.")
    else:
        print(f"{subject}: Good performance.")

# -----------------------------------
# 9. Improvement Suggestions
# -----------------------------------

print("\n========== IMPROVEMENT SUGGESTIONS ==========")

if attendance < 75:
    print("- Improve attendance and attend classes regularly.")

if study_hours < 3:
    print("- Increase daily study hours.")

if assignment_score < 70:
    print("- Complete assignments on time.")

if previous_marks < 60:
    print("- Revise basic concepts and practice regularly.")

if subject_marks[weak_subject] < 70:
    print(f"- Give extra attention to {weak_subject}.")

if (
    attendance >= 75
    and study_hours >= 3
    and assignment_score >= 70
    and previous_marks >= 60
    and all(mark >= 70 for mark in subject_marks.values())
):
    print("- Maintain your current study routine.")

# -----------------------------------
# 10. Model Evaluation
# -----------------------------------

print("\n========== MODEL EVALUATION ==========")

print(f"Regression MAE: {mae:.2f}")
print(f"Regression R2 Score: {r2:.2f}")
print(f"Classification Accuracy: {accuracy * 100:.2f}%")

# -----------------------------------
# 11. Performance Graph
# -----------------------------------

plt.figure(figsize=(8, 5))

subjects = list(subject_marks.keys())
marks = list(subject_marks.values())

plt.bar(subjects, marks)

plt.axhline(
    y=70,
    linestyle="--",
    label="Target Marks (70)"
)

plt.title("Student Subject-wise Performance")
plt.xlabel("Subjects")
plt.ylabel("Marks")
plt.ylim(0, 100)
plt.legend()
plt.tight_layout()

plt.savefig("performance_graph.png")

plt.show()

print("\nGraph saved as performance_graph.png")
print("\nProject executed successfully!")
