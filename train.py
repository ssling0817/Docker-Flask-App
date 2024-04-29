from sklearn.svm import SVC
from sklearn import datasets
import joblib

# Load your dataset
iris = datasets.load_iris()

# Create Classifier
clf = SVC()

# Train the model
model = clf.fit(iris.data, iris.target_names[iris.target])

# Save the model
joblib.dump(model, 'iris_model.pkl')