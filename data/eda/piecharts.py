import matplotlib.pyplot as plt

hobbies = {"No": 18,
           "Chronic illness": 1,
           "Neurodivergent": 1}

sorted_hobbies = sorted(hobbies.items(), key=lambda x: x[1], reverse=True)
top_7 = dict(sorted_hobbies[:3])
# others = sum(value for _, value in sorted_hobbies[9:])

# # Adicionando "Others" aos dados
# top_7["Others"] = others

# Dados para o gráfico
labels = list(top_7.keys())
sizes = list(top_7.values())

colors = ["#92befc", "#261CE6", "#6CA6FE", "#3F89FD", "#85E3FF", 
          "#92E6E6", "#C7CEFF", "#99A6FF", "#5874E6", "#4056E6", "#AEC6FF"]

# Criando o gráfico de pizza
plt.figure(figsize=(10, 8))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, textprops={'fontsize': 16}, colors=colors)
plt.tight_layout()
plt.savefig(f"diseases_userstudies.svg", format="svg")
plt.show()