import logging

from bert_score import BERTScorer
import transformers

import matplotlib.pyplot as plt
from matplotlib import rcParams

"""
    This script compute semantic similarity between sentences using the bert-score library
    - This script is inspired by the BERTScore tutorial https://github.com/Tiiiger/bert_score/blob/master/example/Demo.ipynb
    The output for this metric if the rescale_with_baseline=True is between 0.0 and 1.0 where a score of 0.0 denotes a perfect mismatch and a score of 1.0 denotes a perfect match between candidate sentence and reference sentence.
    - Run the anaconda galacticaEnv env to use this script: conda activate galacticaEnv
"""

def load_data_pair():
    """
        This function laod the candidates paraphrases and the references utterances.
    """
    with open("hyps.txt") as f:
        cands = [line.strip() for line in f]

    with open("refs.txt") as f:
        refs = [line.strip() for line in f]
    
    return cands, refs



# hide the loading messages
transformers.tokenization_utils.logger.setLevel(logging.ERROR)
transformers.configuration_utils.logger.setLevel(logging.ERROR)
transformers.modeling_utils.logger.setLevel(logging.ERROR)


scorer = BERTScorer(lang="en", rescale_with_baseline=True)
print(type(scorer))
sys.exit()
refs = [
    'book a flight from Lyon to Sydney'.strip(),

]
cands = [
    "I want to book a flight ticket from Lyon to Sydney, can you do that?".strip(),
    'book a flight from Lyon to Sydney'.strip(),
    'Find a flight going from Lyon to Sydney.'.strip(),
    'I need to book a flight going from Lyon to Sydney'.strip()
]

for c in cands:
    r = [refs[0]]
    P, R, F1 = scorer.score([c], r)
    print(type(F1))
    print(f"P: {P}, R:{R}, F1:{F1.item()}")
    print("-"*80)

# scorer.plot_example(cands[0], refs[0])
