import pickle

with open("correlation.pkl", "rb") as f:
	corr = pickle.load(f)

result = open("corr_result.csv", "w")

for i in corr:
	result.write(str(i[0][0]))
	result.write(str(','))
	result.write(str(i[0][1]))
	result.write(str(','))
	result.write(str(i[1]))
	result.write(str('\n'))
result.close()
