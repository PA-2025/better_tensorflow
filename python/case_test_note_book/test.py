from collections import defaultdict


# -------- MAP --------
def map_function(data):
    mapped = []
    for subject, note in data:
        mapped.append((subject, note))
    return mapped


# -------- SHUFFLE & SORT --------
def shuffle_sort(mapped_data):
    grouped = defaultdict(list)
    for key, value in mapped_data:
        grouped[key].append(value)
    return grouped


# -------- REDUCE --------
def reduce_function(grouped_data):
    reduced = {}
    for subject, notes in grouped_data.items():
        reduced[subject] = sum(notes) / len(notes)
    return reduced


# -------- EXECUTION --------
data = [("Math", 12), ("Math", 15), ("Info", 18), ("Info", 14), ("Physique", 16)]

mapped = map_function(data)
print("mapped:", mapped)
grouped = shuffle_sort(mapped)
print("grouped:", grouped)
result = reduce_function(grouped)
print(r"result:", result)
