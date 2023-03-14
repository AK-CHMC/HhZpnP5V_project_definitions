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

to read more about why this is useful, refer to: 
https://www.tensorflow.org/guide/data_performance#parallelizing_data_extraction

In brief summary, it's far less memory-intensive, as it allows the data to be
streamed directly from storage, meaning it requires less resources to train, and
often runs significantly faster. It also makes it easy to parallelize any operations
that need completed at runtime as well as prefetch data batches. In other words,
it generally makes model training significantly faster and cheaper.
"""

class PipelineParser():
    def __init__(self,embedder,normalize_fn):
        self.embedder = embedder
        self.normalize_fn = normalize_fn
    
    # python function that passes the fasta amino acid sequences through seqveq
    def python_fn(self,x):
        decoded_sequence = x.decode("utf-8") # decode bytestring to string
        embedded_seq = self.embedder.embed(decoded_sequence) # generate raw seqvec embedding
        embedded_seq = np.sum(embedded_seq,axis=0) # calculate per-residue embeddings
        embedded_seq = embedded_seq.astype(np.float32) # cast to float32 for consistency
        return embedded_seq
    
    # function to wrap python function into numpy tf.data pipeline functions
    # most functions do not require this wrapping; seqvec forces its output
    # to be a numpy function, and requires base python string input
    def numpy_fn(self,x):
        return tf.numpy_function(self.python_fn,
                                inp=[x],
                                Tout=tf.float32,
                                stateful=False)
    
    def embed_sequence(self,pid,seq):
        emb = self.numpy_fn(seq)
        # emb = tf.expand_dims(emb,0)
        emb = emb[tf.newaxis,:,:] # expand dim to allow variable protein length
        # ^ may result in batch shape: [batch_size,1,<protein_lengths>,1024]
        emb = self.normalize_fn(emb) # normalize data samples
        emb = tf.RaggedTensor.from_tensor(emb) # convert to ragged tensor
        return pid, seq, emb # return as protein id, fasta sequence, seqvec embedding

"""
This class serves to feed each protein sequence through seqvec,
convert the corresponding numpy array to a tensorflow output. 

Example usage:


    embedder = SeqVecEmbedder()
    normalizer = load_normalizer(norm_path)
    parser = PipelineParser(embedder,newnorm)

    ds = get_dataset(fasta_path)
    ds = ds.map(parser.embed_sequence,num_parallel_calls=tf.data.AUTOTUNE)

    # sample from ds using iterator:
    iterator = iter(ds)
    sample = next(iterator)
    print(sample)

# note, batching may require something like the following due to ragged arrays:
    batcher = tf.data.experimental.dense_to_ragged_batch(32,drop_remainder=True,row_splits_dtype=tf.dtypes.float64)
    
    ds = ds.apply(batcher)



"""
