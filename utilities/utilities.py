import csv
import pickle as pkl

with open( '../utilities/pickle_files/LETTERS.pickle', 'rb') as file:
    LETTERS = pkl.load(file)
with open( '../utilities/pickle_files/DIACRITICS.pickle', 'rb') as file:
    DIACRITICS = pkl.load(file)
with open( '../utilities/pickle_files/DIACRITIC_TO_ID.pickle', 'rb') as file:
    DIACRITIC_TO_ID = pkl.load(file)

def convert_ids_and_labels(input_txt_file_path, output_csv_file_path):
  with open(input_txt_file_path, 'r', encoding='utf-8') as file:
      lines = file.readlines()
  id=0
  ids=[]
  labels=[]
  for line in lines:
      for index, char in enumerate(line):
          if char in LETTERS:
              ids.append(id)
              id+=1
              char_diac = ''
              if index + 1 < len(line) and line[index + 1] in DIACRITICS:
                  char_diac = line[index + 1]
                  if index + 2 < len(line) and line[index + 2] in DIACRITICS and char_diac + line[index + 2] in DIACRITIC_TO_ID:
                      char_diac += line[index + 2]
                  elif index + 2 < len(line) and line[index + 2] in DIACRITICS and line[index + 2] + char_diac in DIACRITIC_TO_ID:
                      char_diac = line[index + 2] + char_diac
              labels.append(DIACRITIC_TO_ID[char_diac])

  # Save ids and labels to a CSV file

  with open(output_csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
      csv_writer = csv.writer(csvfile)
      csv_writer.writerow(['ID', 'Label'])  # Write header
      csv_writer.writerows(zip(ids, labels))  # Write data


"""
this function counts number of erroneous output and gets rate of errors
by dividing erroneous with total number of examples
returned value is a float that is NOT multiplied by 100
"""

def DER(model_output, true_output):
    
    total_sentences = len(true_output)
    errors = 0
    
    for i in range(0,len(true_output)):
        if model_output[i]['diacritic'] != true_output[i]['diacritic']:
            errors +=1
            
    diactric_error_rate = errors/total_sentences
    
    return diactric_error_rate



def csv_calculate_accuracy(gold_csv_file_path,test_csv_file_path):

  # Read the gold outputs
  gold_labels = {}
  with open(gold_csv_file_path, 'r', encoding='utf-8') as gold_file:
      gold_reader = csv.reader(gold_file)
      next(gold_reader)  # Skip header
      for row in gold_reader:
          id, label = row
          gold_labels[int(id)] = label

  # Read the test outputs
  test_labels = {}
  with open(test_csv_file_path, 'r', encoding='utf-8') as test_file:
      test_reader = csv.reader(test_file)
      next(test_reader)  # Skip header
      for row in test_reader:
          id, label = row
          test_labels[int(id)] = label

  # Compare labels and calculate accuracy

  correct_count = 0

  for id in test_labels:
      gold_label = gold_labels.get(id, None)
      test_label = test_labels.get(id, None)

      if gold_label == test_label:
          correct_count += 1

  accuracy = correct_count / len(test_labels)
  return accuracy

