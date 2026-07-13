from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
vector1 = np.array([[1, 2, 3, 4]])
vector2 = np.array([[2, 4, 6, 8]])
similarity = cosine_similarity(vector1, vector2)

print("Cosine Similarity:", similarity[0][0])
if similarity[0][0] == 1:
    print("The vectors are identical in direction.")
elif similarity[0][0] > 0.5:
    print("The vectors are highly similar.")
else:
    print("The vectors are not very similar.")
