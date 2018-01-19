"""
learn the list
"""
list = [0, 2, 3, 4, 5]
for i in  list:
    print i

for i, value in enumerate(list):
    print i, "-" , value


for i in range(len(list)):
    print list[i]

print "~~~~~~~~~~~~~~~"
for i, value in enumerate(list, 2):
    print i, "-", value

list2 = [6, 7, 8, 9, 10]

print "~~~~~~~~~~~~~~"

for i in list, list2:
    result = i[:]

print "test list: " , result

print '~~~~~~~~~~~~~~~~~~~~'
dict = {'A':'a', 'B':'b'}
x = dict.keys()
value = dict.values()

print 'x: ', x
print 'y: ', value
for i in x, value:
    answer, quest = i[:]
    print answer, ", ", quest

print "~~~~~~~~~~~"
for j in x:
    answer = j[:]
    print answer


print '~~~~~~~~~~~~~~~'
for x, y in zip(list, list2):
    print x, "-" , y