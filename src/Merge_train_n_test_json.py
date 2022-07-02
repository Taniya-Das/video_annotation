"To merge train and test json files"
import json

f1data = f2data = ""

with open('C:/Users/Taniya/Downloads/MSVD_train.json') as f1:
    f1data = f1.read()
with open('C:/Users/Taniya/Downloads/MSVD_test.json') as f2:
    f2data = f2.read()

f1data += "\n"
f1data += f2data
with open('C:/Users/Taniya/Downloads/json_data_list.json', 'a') as f3:
    f3.write(f1data)



# #Read JSON File
# # Opening JSON file
# f = open('MSVD_train.json')
# # returns JSON object as a dictionary
# data = json.load(f)
# # Iterating through the json list
# for key,value in data.items():
#     print(key, value[0])
# # Closing file
# f.close()


