import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

customers = pd.read_csv(r'C:\Users\Harshavardhan rao\Downloads\Customers.csv')
products = pd.read_csv(r'C:\Users\Harshavardhan rao\Downloads\Products.csv')
transactions = pd.read_csv(r'C:\Users\Harshavardhan rao\Downloads\Transactions.csv')

#EDA Visualizations

#Customer distribution by region
plt.figure(figsize=(8, 6))
sns.countplot(data=customers, x='Region', palette='viridis')
plt.title('Customer Distribution by Region')
plt.xlabel('Region')
plt.ylabel('Number of Customers')
plt.xticks(rotation=45)
plt.show()

#Most popular product categories
plt.figure(figsize=(8, 6))
sns.countplot(data=products, y='Category', order=products['Category'].value_counts().index, palette='viridis')
plt.title('Most Popular Product Categories')
plt.xlabel('Count')
plt.ylabel('Category')
plt.show()

#Total sales by region
region_sales = merged_data.groupby('Region')['TotalValue'].sum().sort_values(ascending=False)
region_sales.plot(kind='bar', figsize=(8, 6), color='teal')
plt.title('Total Sales by Region')
plt.xlabel('Region')
plt.ylabel('Total Sales (USD)')
plt.xticks(rotation=45)
plt.show()

#Monthly transactions over time
merged_data['TransactionDate'] = pd.to_datetime(merged_data['TransactionDate'])
merged_data['MonthYear'] = merged_data['TransactionDate'].dt.to_period('M')
monthly_transactions = merged_data.groupby('MonthYear').size()
monthly_transactions.plot(kind='line', figsize=(10, 6), marker='o', color='orange')
plt.title('Monthly Transactions Over Time')
plt.xlabel('Month-Year')
plt.ylabel('Number of Transactions')
plt.grid(True)
plt.show()

#Top 10 products by sales
top_products = merged_data.groupby('ProductName')['TotalValue'].sum().sort_values(ascending=False).head(10)
top_products.plot(kind='bar', figsize=(10, 6), color='purple')
plt.title('Top 10 Products by Total Sales')
plt.xlabel('Product Name')
plt.ylabel('Total Sales (USD)')
plt.xticks(rotation=45)
plt.show()


