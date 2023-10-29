filename = './test.txt'
table = dict()
cnt = 0
with open(filename) as file:
    for line in file:
        l = line.strip()
        for c in line:
            if not c.isalpha():
                continue
            else:
                cnt+=1
                if c in table:
                    table[c]+=1
                else:
                    table[c] = 1

for key in table.keys():
    table[key]/=cnt
print(table)