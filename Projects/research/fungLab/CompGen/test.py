file = open("./loss_stuff", "r")
x = file.read();
y = x.split(",");
print(len(y))
y[0] = y[0][1:];
y[-1] = y[-1][:-2];

temp = []

for a in range(len(y)):
    y[a] = float(y[a].strip());

# print(y[0])

import matplotlib.pyplot as plt
plt.plot(y)
plt.show();
