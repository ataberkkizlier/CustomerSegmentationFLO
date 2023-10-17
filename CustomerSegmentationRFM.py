import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import seaborn as sns

'''Task 1: Understanding and Preparing Data'''
'''Step1: Read the flo_data_20K.csv file and make a copy of the dataframe.'''

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

df_ = pd.read_csv("/Users/ataberk/Desktop/Miuul Bootcamp/week 3/FLOMusteriSegmentasyonu/flo_data_20k.csv")
df = df_.copy()

''' Step2: In the dataset.'''

# a. Top 10 observations,
df.head(10)
# b. Variable names,
df.describe().T
# c. Descriptive statistics,
df.shape
df.nunique()
# d. Empty value,
df.isnull().sum()
# e. Examine variable types.
df.dtypes

''' Step 3: Omnichannel means that customers shop both online and offline. Create new variables for the total number of
# purchases and spending of each customer.'''

df["total_number_of_purchases"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df['total_number_of_spending'] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]

df.info

'''Step 4: Examine variable types. Change the type of variables that express date to date'''

df.dtypes

date_vars = df.columns[df.columns.str.contains("date")]
for col in date_vars:
    df[col] = pd.to_datetime(df[col])
df.info()

'''Step 5: Look at the distribution of the number of customers, total number of products purchased and total
# expenditures across shopping channels. Also create a barchart using seaborn.'''

df.groupby("order_channel").agg({"total_number_of_purchases": ["count", "sum"],
                                 "total_number_of_spending": ["count", "sum"]})

# Group the data by order channel and calculate the total number of purchases and total number of spending
grouped_df = df.groupby("order_channel").agg({"total_number_of_purchases": ["count", "sum"],
                                            "total_number_of_spending": ["count", "sum"]})

# Get the total number of purchases and total number of spending for each order channel
total_number_of_purchases = grouped_df["total_number_of_purchases"]["sum"]
total_number_of_spending = grouped_df["total_number_of_spending"]["sum"]

# Create a bar chart of the total number of purchases and total number of spending for each order channel using Seaborn
sns.barplot(x=total_number_of_purchases.index, y=total_number_of_purchases, color="blue")
sns.barplot(x=total_number_of_spending.index, y=total_number_of_spending, bottom=total_number_of_purchases, color="red")

# Set the chart title and labels
plt.title("Total Number of Purchases and Total Number of Spending by Order Channel")
plt.xlabel("Order Channel")
plt.ylabel("Total Number of Purchases/Total Number of Spending")

# Display the chart
plt.show()

'''Step 6: Rank the top 10 most profitable customers.'''

df[["master_id", "total_number_of_spending", "total_number_of_purchases"]].sort_values("total_number_of_spending", ascending=False).head(10)


'''Step 7: List the top 10 customers who placed the most orders.'''

df[["master_id", "total_number_of_spending", "total_number_of_purchases"]].sort_values("total_number_of_purchases", ascending=False).head(10)


'''Step 8: Functionalize the data preparation process.'''


def functionalize(df):
    df_ = pd.read_csv(df)
    df = df_.copy()
    df["total_number_of_purchases"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
    df['total_number_of_spending'] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]

    date_vars = df.columns[df.columns.str.contains("date")]
    for col in date_vars:
        df[col] = pd.to_datetime(df[col])
    return df

df = functionalize("/Users/ataberk/Desktop/Miuul Bootcamp/week 3/FLOMusteriSegmentasyonu/flo_data_20k.csv")
df.head()

'''#######Task 2: Calculating RFM Metrics#######'''

'''Step 1: Define Recency, Frequency and Monetary.'''
# Recency : When was the last time our customer purchase something? (Today's date - Last purchase date)
# Frequency : How frequently has the customer been shopping?
# Monetary: The total money the customer spent


# NOTE: To calculate the recency value, you can select the analysis date 2 days after the maximum date.
'''Step 2: Calculate Recency, Frequency and Monetary metrics for the customer.
Step 3: Assign the metrics you calculated to a variable named rfm.
Step 4: Change the names of the metrics you created to recency, frequency and monetary.'''


df["last_order_date"].max()
today_date= df["last_order_date"].max() + dt.timedelta(days=2)
today_date

rfm = pd.DataFrame()
rfm["master_id"] = df["master_id"]
rfm["recency"] = (today_date - df["last_order_date"])
rfm["frequency"] = df["total_number_of_purchases"]
rfm["monetary"] = df["total_number_of_spending"]

rfm.head()
rfm= rfm[rfm["monetary"]> 0]  #Monetary can't be less than 0.
rfm.describe().T

'''Task 3: Calculating the RF Score'''

'''Step 1: Convert Recency, Frequency and Monetary metrics into scores between 1-5 with qcut.
Step 2: Save these scores as recency_score, frequency_score and monetary_score.
Step 3: Express recency_score and frequency_score as a single variable and save them as RF_SCORE.'''

rfm["recency_Score"] = pd.qcut(rfm["recency"], q=5, labels=[5, 4, 3, 2, 1])
rfm["frequency_Score"] = pd.qcut(rfm["frequency"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5])
rfm["monetary_Score"] = pd.qcut(rfm["monetary"], q=5, labels=[1, 2, 3, 4, 5])
rfm["RF_Score"] = rfm["recency_Score"].astype(str) + rfm["frequency_Score"].astype(str)

rfm.head()


'''Task 4: Segmental Identification of the RF Score'''

'''Step 1: Make segment definitions for the created RF scores. 
Step 2: Convert the scores into segments with the help of seg_map below.'''

seg_map = {
    r"[1-2][1-2]": "hibernating",
    r"[1-2][3-4]": "at_Risk",
    r"[1-2]5": "cant_loose",
    r"3[1-2]": "about_to_sleep",
    r"33": "need_attention",
    r"[3-4][4-5]": "loyal_customers",
    r"41": "promising",
    r"51": "new_customers",
    r"[4-5][2-3]": "potential_loyalists",
    r"5[4-5]": "champions"
}

rfm["segment"] = rfm["RF_Score"].replace(seg_map, regex=True)

rfm.head()



'''Task 5: Time for Action!'''

'''Step 1: Examine the recency, frequnecy and monetary averages of the segments.'''


rfm.groupby("segment").agg({"recency": ["mean", "count"],
                           "frequency": ["mean", "count"],
                           "monetary": ["mean", "count"]})
#rfm[["segment", "recency", "frequency", "monetary"]].groupby("segment").agg(["mean", "count"])

'''Step 2: With the help of RFM analysis, find the customers in the relevant profile for the 2 cases given below and 
save the customer ids as csv. a. FLO is adding a new women's footwear brand. The product prices of the brand are 
above the general customer preferences. For this reason, it wants to contact customers who are interested in 
promoting the brand and selling its products. Loyal customers (champions, loyal_customers) and shoppers from the 
women's category are the customers to be contacted specifically. Save the id numbers of these customers in csv file.'''



rfm.head()
rfm_final = rfm.merge(df, on='master_id', how='left')
rfm_final.head()


flo_women = rfm_final.loc[(rfm_final["segment"].isin(["champions", 'loyal_customers'])) &
                       (rfm_final["interested_in_categories_12"].str.contains("KADIN"))]
flo_women.shape
flo_women.to_csv('flo_women')


'''b. A 40% discount is planned for Men's and Children's products. Customers who are interested in the categories 
related to this discount, who have been good customers in the past but have not been shopping for a long time, 
customers who should not be lost, dormant customers and new customers are to be specifically targeted. Save the ids 
of the customers in the appropriate profile to the csv file.'''

flo_men_kid_40_off = rfm_final.loc[(rfm_final["segment"].isin(["cant_loose", "about_to_sleep", 'new_customers'])) &
                    (rfm_final["interested_in_categories_12"].str.contains('COCUK', 'ERKEK'))]

flo_men_kid_40_off.shape
flo_men_kid_40_off.to_csv("flo_men_kid_40_off")
