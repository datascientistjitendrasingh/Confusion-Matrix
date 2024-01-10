# Confusion matrics Program:
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
data={
    'EmployeeID': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
    'Department': ['Sales', 'HR', 'IT', 'Sales', 'HR', 'IT', 'Sales', 'HR', 'IT', 'Sales'],
    'YearsAtCompany': [5, 2, 7, 10, 1, 3, 4, 8, 2, 5],
    'PerformanceRating': [3, 4, 2, 5, 3, 5, 3, 4, 4, 3],
    'ActualPromotion': ['No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'No'],
    'PredictedPromotion': ['No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'No', 'No', 'No']
}
df=pd.DataFrame(data)
df
conf_matrix = confusion_matrix(df['ActualPromotion'],df['PredictedPromotion'],labels=['Yes','No'])
print(conf_matrix)
sns.set()
plt.figure(figsize=(8,6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=["Predicted: Yes", "Predicted: No"], yticklabels=["Actual: Yes", "Actual: No"])
plt.xlabel('Predicted Promotion')
plt.ylabel('Actual Promotion')
plt.title('Confusion matrix for Employee Promotion')
plt.show()
