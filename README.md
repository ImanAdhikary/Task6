# 🌸 Iris Dataset K-Nearest Neighbors (KNN) Classifier

A machine learning project to implement and visualize K-Nearest Neighbors (KNN) classification on the famous Iris dataset, using dimensionality reduction via PCA for decision boundary visualization.

---

## 📌 Problem Statement  

Classify iris flowers into three species — *Iris-setosa*, *Iris-versicolor*, and *Iris-virginica* — based on four flower measurements.

---

## 📚 Project Workflow  

### ✅ 1️⃣ Choose a Classification Dataset  

- Used the **Iris dataset** available via `sklearn.datasets`.
- Features:
  - Sepal Length (cm)
  - Sepal Width (cm)
  - Petal Length (cm)
  - Petal Width (cm)
- Target:
  - Species (*setosa*, *versicolor*, *virginica*)

---

### ✅ 2️⃣ Normalize Features  

- Scaled numerical features using **StandardScaler** from `sklearn.preprocessing` to improve KNN performance as it’s distance-based.


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

---

### ✅ 3️⃣ Apply PCA for Dimensionality Reduction

Since the Iris dataset has 4 features and decision boundaries are typically visualized in 2D, we applied Principal Component Analysis (PCA) to reduce it to 2 components for visualization.

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

---

### ✅ 4️⃣ Use KNeighborsClassifier from sklearn

Imported KNeighborsClassifier from sklearn.neighbors.

Trained the classifier with different values of K (1, 3, 5, 7, 9) on both original scaled data and PCA-reduced data (for visualization).

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)

---

### ✅ 5️⃣ Experiment with Different Values of K

Tested multiple K values.
Recorded accuracy for each using accuracy_score.
Evaluated models using Confusion Matrix and ConfusionMatrixDisplay from sklearn.metrics.

from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(confusion_matrix=cm).plot()

---

### ✅ 6️⃣ Visualize Decision Boundaries
Plotted decision boundaries on PCA-reduced 2D data.
Created a meshgrid using numpy.meshgrid and predicted classifications on the grid.
Visualized using matplotlib.contourf.

#### Create meshgrid
x_min, x_max = x_train[:, 0].min() - 1, x_train[:, 0].max() + 1
y_min, y_max = x_train[:, 1].min() - 1, x_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))

#### Predict for meshgrid points
Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

#### Plot decision boundary
plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdYlBu)
plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, edgecolors='k')
plt.title("KNN Decision Boundary (k=5)")
plt.xlabel('Sepal Length (scaled)')
plt.ylabel('Sepal Width (scaled)')
plt.show()




