import matplotlib.pyplot as plt

# Your list with some indices having no value
your_list = [10, None, 20, None, 30, 40, None, 50]

# Populate a list of tuples with index and value pairs, skipping None values
index_value_pairs = [(i, val) for i, val in enumerate(your_list) if val is not None]

# Unpack the index and value pairs
indices, values = zip(*index_value_pairs)

# Plotting
plt.plot(indices, values, marker='o', linestyle='-')
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Index vs Value')
plt.grid(True)
plt.show()