import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
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
frequent_items = apriori(df, min_support=0.2, use_colnames=True)

# Rules
rules = association_rules(frequent_items, metric="confidence", min_threshold=0.5)

# UI
st.title("🛒 Smart Product Recommendation System")

# -----------------------------
# 🔹 Product Selection
# -----------------------------
products = list(df.columns)
product = st.selectbox("Select a product:", products)

if product:
    st.subheader(f"Recommendations for {product}")

    found = False
    for _, row in rules.iterrows():
        if product in list(row['antecedents']):
            st.write(f"👉 {list(row['consequents'])[0]}")
            st.write(f"Confidence: {round(row['confidence'],2)} | Lift: {round(row['lift'],2)}")
            found = True

    if not found:
        st.write("❌ No strong recommendations found")

# -----------------------------
# 📊 Top Products Chart
# -----------------------------
st.subheader("📊 Product Popularity")

product_counts = df.sum().sort_values(ascending=False)
st.bar_chart(product_counts)

# -----------------------------
# 🔗 Network Graph (SNA)
# -----------------------------
st.subheader("🔗 Product Network Graph")

G = nx.Graph()

for basket in data:
    for i in range(len(basket)):
        for j in range(i + 1, len(basket)):
            if G.has_edge(basket[i], basket[j]):
                G[basket[i]][basket[j]]['weight'] += 1
            else:
                G.add_edge(basket[i], basket[j], weight=1)

plt.figure(figsize=(8,6))
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='lightblue')
nx.draw_networkx_edge_labels(G, pos)

st.pyplot(plt)