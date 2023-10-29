#
import csv
import random
from faker import Faker
import pandas as pd
import matplotlib.pyplot as plt

fake = Faker()
student_data = []

for _ in range(10000):
    student_id = fake.unique.random_number(digits=8)
    gender = random.choice(["M", "F"])
    height = round(random.uniform(1.5, 1.9), 2)
    weight = round(random.uniform(40, 80), 2)
    blood_type = random.choice(["A", "B", "AB", "O"])
    physical_result = random.choice(["A", "B", "C", "D"])
    monthly_consumption = round(random.uniform(300, 1500), 2)
    student_data.append([student_id, gender, height, weight, blood_type, physical_result, monthly_consumption])

filename = "./stu-data.csv"

with open(filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["StudentID", "Gender", "Height", "Weight", "BloodType", "PhysicalResult", "MonthlyConsumption"])
    writer.writerows(student_data)

# Load data into pandas
df = pd.read_csv(filename)
df["BMI"] = df["Weight"] / (df["Height"] ** 2)
df["Gender"] = df["Gender"].map({"M": 0, "F": 1})
df["BloodType"] = df["BloodType"].map({"A": 0, "B": 1, "AB": 2, "O": 3})
df["PhysicalResult"] = df["PhysicalResult"].map({"A": 0, "B": 1, "C": 2, "D": 3})
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]


correlations = df.corr()["MonthlyConsumption"]


print(correlations)


correlations.drop("MonthlyConsumption").plot(kind='bar', title="Factors affecting Monthly Cafeteria Spending")
plt.ylabel("Correlation Strength")
plt.savefig('./analyse.png')
plt.show()

