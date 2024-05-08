import matplotlib.pyplot as plt
fig = plt.figure()
plt.imshow(sim.cpu(), cmap='hot', interpolation='nearest')
plt.colorbar()
plt.title('Cosine Similarity')
fig.savefig('test.png')