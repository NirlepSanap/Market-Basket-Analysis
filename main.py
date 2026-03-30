import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# Load dataset
data = []
with open('data.csv') as f:
    for line in f:
        data.append(line.strip().split(','))

# Encoding
te = TransactionEncoder()
te_data = te.fit(data).transform(data)
df = pd.DataFrame(te_data, columns=te.columns_)

# Apriori
frequent_items = apriori(df, min_support=0.4, use_colnames=True)
print("Frequent Itemsets:\n", frequent_items)

# Association Rules
rules = association_rules(frequent_items, metric="confidence", min_threshold=0.4)
print("\nAssociation Rules:\n", rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])