import json
def load_data(file_location):
  data = []
  with open(file_location,'r') as f:
    for line in f:
      data.append(json.loads(line))
  return data
