import multiprocessing
import pandas as pd # TODO - remove pandas dependency
import tensorflow as tf


def parse_entries(entry):
    lines = entry.split("\n")
    pid = lines.pop(0)
    seq = "".join(lines)
    return pid,seq

def gd_internal_fn(filter_input):
    return len(filter_input)>0

# creates tf.data Dataset from fasta file
# also, sorts proteins by sequence length
def get_dataset(path):
    with open(path,"r") as file:
        text = file.read()
    text = text.split(">")
    text = filter(gd_internal_fn,text) # remove empty entries
    
    core_num = os.cpu_count() # grab the number of cpu cores available
    with multiprocessing.Pool(core_num) as p:
        samples = p.map(parse_entries,text)
    pid,seq = zip(*samples)
    df = pd.DataFrame({"ProteinID":pid,"sequence":seq})
    # sort the data by protein length
    # this is by done by request of seqvec itself
    # and avoids wasteful computation in tf model.
    idx_sorted_by_length = df.sequence.str.len().sort_values().index
    df = df.reindex(idx_sorted_by_length)
    df = df.reset_index(drop=True)
    
    return tf.data.Dataset.from_tensor_slices((df.ProteinID,df.sequence))

"""
This operation works by leveraging the fact that each fasta file begins with 
a > character to split the raw text into a list of entries, each of which are
then passed through the parse_entries function after filtering out empty lines

This function separates the entries into protein id and protein sequence by
leveraging the fact that the first line of each entry is dedicated entirely to 
the protein id, while all subsequent lines which make up the entry contain the 
amino acid residues. To do this, it splits each entry by the new line character,
pops off the first entry to grab the pid, and then joins the remaining lines
into a single string.

Next, the output of this initial parser is then unzipped to get a list of the 
protein ids and a list of the sequences, and is then used to create the tf.data
pipeline
"""
