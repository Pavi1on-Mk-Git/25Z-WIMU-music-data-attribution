from numpy.lib.format import open_memmap

scores = open_memmap("results/trak_musicgen_test/scores/musicgen_test.mmap")

top_1_acc = 0
top_5_acc = 0

for test_idx in range(len(scores)):
    top_trak_scores = scores[:, test_idx].argsort()[-5:][::-1]
    top_1_acc += top_trak_scores[0] == test_idx
    top_5_acc += test_idx in top_trak_scores

print(f"top 1 accuracy: {top_1_acc / len(scores):.2f}")
print(f"top 5 accuracy: {top_5_acc / len(scores):.2f}")
