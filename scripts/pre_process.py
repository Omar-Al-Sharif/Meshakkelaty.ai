import re
import textwrap

#double damma, double fatha, double kasera, damma, fatha, kasera, sukoon, shadd
TASHKEEL_SET = {'ٌ', 'ً', 'ٍ', 'ُ', 'َ', 'ِ', 'ْ', 'ٌّ', 'ّ'}
DIACRITICS_REGEX = re.compile('|'.join(TASHKEEL_SET))
def clean_data(data):
    return re.sub(DIACRITICS_REGEX, '', data)


def convert_to_one_hot(indices, size):
    return [[1 if i == elem else 0 for i in range(size)] for elem in indices]



def process_line(line):
    # Define a pattern to match specific punctuation marks
    punctuation_pattern = r'([.,:;؛()\[\]{}«»،])'

    # Replace matched punctuation marks with the same followed by a line break
    processed_line = re.sub(punctuation_pattern, r'\1\n', line)

    # Split the processed line into substrings based on line breaks
    substrings = processed_line.split('\n')

    # Filter out empty substrings
    substrings = [substring for substring in substrings if len(substring.strip()) > 0]

    return substrings


def wrap_lines(lines):
    # Initialize an empty list to store the processed data
    processed_data = []

    # Iterate through each line in the raw data
    for line in lines:
        # Process the line and split it into substrings
        substrings = process_line(line)

        # Wrap lines and append non-empty substrings with length <= 500
        for substring in substrings:
            cleaned_substring = clean_data(substring).strip()
            if len(cleaned_substring) > 0:
                processed_data.extend(
                    wrapped_line.strip() for wrapped_line in textwrap.wrap(cleaned_substring, width=500)
                )

    # Return the processed data
    return processed_data

