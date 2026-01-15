import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

class ExpenseAnalyzer:

    def __init__(self, file_path):
        self.df = pd.read_csv(file_path)
        self.prepare_data()

    def prepare_data(self):
        self.df["Date"] = pd.to_datetime(self.df["Date"])
        self.df["Month"] = self.df["Date"].dt.month
        self.df["Day"] = self.df["Date"].dt.day

    def basic_analysis(self):
        print("\nTotal Expense")
        print(self.df["Amount"].sum())

        print("\nAverage Daily Expense")
        print(self.df["Amount"].mean())

        print("\nCategory Wise Expense")
        print(self.df.groupby("Category")["Amount"].sum())

    def visualize_expenses(self):
        plt.figure()
        sns.barplot(x="Category", y="Amount", data=self.df, estimator=sum)
        plt.title("Category Wise Spending")
        plt.show()

        plt.figure()
        self.df.groupby("Date")["Amount"].sum().plot()
        plt.title("Daily Expense Trend")
        plt.xlabel("Date")
        plt.ylabel("Amount")
        plt.show()

    def monthly_prediction(self):
        daily_expense = self.df.groupby("Day")["Amount"].sum().reset_index()
        X = daily_expense[["Day"]]
        y = daily_expense["Amount"]

        model = LinearRegression()
        model.fit(X, y)

        future_days = np.array(range(1, 31)).reshape(-1, 1)
        predictions = model.predict(future_days)

        print("\nPredicted average daily expense next month")
        print(round(predictions.mean(), 2))

        plt.figure()
        plt.plot(future_days, predictions)
        plt.title("Predicted Daily Expense Trend")
        plt.xlabel("Day")
        plt.ylabel("Predicted Amount")
        plt.show()

    def smart_suggestions(self):
        category_sum = self.df.groupby("Category")["Amount"].sum()
        max_category = category_sum.idxmax()

        print("\nSmart Financial Advice")
        print("You spend most on", max_category)

        if category_sum[max_category] > category_sum.mean():
            print("Reduce spending on", max_category, "to save more money")

    def run(self):
        self.basic_analysis()
        self.visualize_expenses()
        self.monthly_prediction()
        self.smart_suggestions()

if __name__ == "__main__":
    analyzer = ExpenseAnalyzer("data/expenses.csv")
    analyzer.run()
