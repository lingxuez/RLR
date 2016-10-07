## pre-process training data
## to get all 17 labels
import sys

def getLabels(filename):
	with open(filename, mode="rt") as fin:
		data = fin.read()

	allLabels = set()
	for line in data.splitlines():
		tokens = line.split()
		labels = tokens[1].split(",")

		for label in labels:
			allLabels.add(label)

	for label in allLabels:
		print label


if __name__ == "__main__":
	if len(sys.argv) != 2:
		print "Usage: python getAllLabels.py trainData.txt"
	else:
		filename = sys.argv[1]
		getLabels(filename)
