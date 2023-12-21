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
            
    

