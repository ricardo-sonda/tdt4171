import pulp as p

# Definer problemet
model = p.LpProblem('HammersAndNails', p.LpMaximize)

# Her skal du definere beslutningsvariablene

# Eksempel:
h = p.LpVariable("x", lowBound = 0, upBound= 40)
s = p.LpVariable("y", lowBound = 0)

# Her skal du legge til den linÃ¦re funksjonen som skal optimeres

# Eksempel:
model += 3000 * h + 1000 * s

# Her skal du legge til de lineÃ¦re ulikhetene som pÃ¥ oppfylles

# Eksempel:
model += 2*h + s <= 100
model += h + 2*s <= 80

# Print modellen
print(model)

# LÃ¸s det lineÃ¦re programmet
status = model.solve()

# Print lÃ¸sningen

# Print verdien til en beslutningsvariabel:
# print(p.value(x))

# Print optimal verdi:
print(p.value(model.objective))