import numpy as np
import pandas as pd
import numpy as np
import string, os 
import re
import time
import model1
import os
import tensorflow as tf
from random import randint
from tensorflow.contrib.seq2seq.python.ops import beam_search_ops
raw_text = open("out.txt","r") 
raw_text = raw_text
corpus = [x for x in raw_text]
raw=[]
for i in corpus:
 
 i = re.sub('[\n]+', '', str(i))
 raw.append(i.lower())
def loadGloveModel(gloveFile):
    print("Loading Glove Model")
    f = open(gloveFile,'r',encoding="utf8")
    embedding_index = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        embedding_index[word] = embedding
    print ("Done.",len(embedding_index)," words loaded!")
    return embedding_index
#for getting the frequency for each word in the text file
def count_word_frequency(word_frequency, data):    
    for text in data:
        for token in text.split():
            if token not in word_frequency:
                word_frequency[token] = 1
            else:
                word_frequency[token] += 1
    return word_frequency
#for calculating missing words in the glove vector and also converting the words into int
def create_conversion_dictionaries(word_frequency, embeddings_index, threshold=10):
    """
    Cleans the dataset by removing the words from a corpus which appear below
    a predefined threshold (default 20).
    
    Input:
        word_frequency: dictionary with word frequencies in the corpus
        embeddings_index: dictionary of words and their corresponding vectors
        threshold: frequency threshold under which words will be discarded
    Returns:
        vocab2int: dictionary to convert vocabulary to integers
        int2vocab: dictionary to reverse conversion, integers and their vocab
    """
    print('Removing token which frequency in the corpus is under specified threshold')
    missing_words = 0
    
    for token, freq in word_frequency.items():
        if freq > threshold:
            if token not in embeddings_index:
                missing_words += 1
                
    missing_ratio = round(missing_words/len(word_frequency), 4) * 100
    print('Number of words missing from glove after removing rare words:', missing_words)
    print('Percent of words that are missing from vocabulary after removing rare words: ', missing_ratio, '%')

    # Dictionary to convert words to integers
    print('Creating vocab_to_int dictionary')
    vocab2int = {}
    
    value = 0
    for token, freq in word_frequency.items():
        if freq >= threshold or token in embeddings_index:
            vocab2int[token] = value
            value += 1
    
    # Special tokens that will be added to our vocab. Those tokens will guide the
    # sequence to sequence model
    codes = ['<UNK>', '<PAD>', '<EOS>', '<GO>']   
    
    print('Adding special tokens to vocab_to_int dictionary.')
    # Add the codes to the vocab list
    for code in codes:
        vocab2int[code] = len(vocab2int)
    
    # Dictionary to convert integers to words
    print('Creating int_to_vocab dictionary.')
    int2vocab = {}
    for token, index in vocab2int.items():
        int2vocab[index] = token
    
    usage_ratio = round(len(vocab2int) / len(word_frequency), 4) * 100
    print("Total number of words:", len(word_frequency))
    print("Number of words we will use:", len(vocab2int))
    print("Percent of words we will use: {}%".format(usage_ratio))
    #print(vocab2int)
    return vocab2int, int2vocab
def create_embedding_matrix(vocab2int, embeddings_index, embedding_dimensions=300):
    """
    Creates embedding matrix for each token left in the corpus, as denoted by 
    vocab to int. If the word vector is not available in the embeddings_index
    then we create a random embedding for it.
    
    Input:
        vocab2int: dictionary which contains vocab to integer conversions for corpus
        embeddings_index: word vectors
        embedding_dimensions: dimensions of word vectors. 300 by default to match Conceptnet
    Returns:
        word_embedding_matrix: final word vectors for our corpus
    """
    # Number of words in total in the corpus
    num_words = len(vocab2int)
    
    # Create a default matrix with all values set to zero and fill it out
    print('Creating word embedding matrix with all the tokens and their corresponding vectors.')
    word_embedding_matrix = np.zeros((num_words, embedding_dimensions), dtype=np.float32)
    #print(word_embedding_matrix)
    for token, index in vocab2int.items():
        if token in embeddings_index:            
            #import pdb
            #pdb.set_trace()
            word_embedding_matrix[index] = embeddings_index[token]            
        else:
            # Else, create a random embedding for it
            new_embedding = np.array(np.random.uniform(-1.0, 1.0, embedding_dimensions))
            word_embedding_matrix[index] = new_embedding
            #embeddings_index[token]=new_embedding
    print(word_embedding_matrix)	
            
    return word_embedding_matrix
#for appending UNK for unique tokens and end of text by EOS
def convert_data_to_ints(data, vocab2int, word_count, unk_count, eos=True):
    """
    Converts the words in the data into their corresponding integer values.
    
    Input:
        data: a list of texts in the corpus
        vocab2list: conversion dictionaries
        word_count: an integer to count the words in the dataset
        unk_count: an integer to count the <UNK> tokens in the dataset
        eos: boolean whether to append <EOS> token at the end or not (default true)
    Returns:
        converted_data: a list of corpus texts converted to integers
        word_count: updated word count
        unk_count: updated unk_count
    """    
    converted_data = []
    for text in data:
        converted_text = []
        for token in text.split():
            word_count += 1
            if token in vocab2int:
                # Convert each token in the paragraph to int and append it
                converted_text.append(vocab2int[token])
            else:
                # If it's not in the dictionary, use the int for <UNK> token instead
                converted_text.append(vocab2int['<UNK>'])
                unk_count += 1
        if eos:
            # Append <EOS> token if specified
            converted_text.append(vocab2int['<EOS>'])
            
        converted_data.append(converted_text)
    
    assert len(converted_data) == len(data)
    return converted_data, word_count, unk_count
def unk_counter(data, vocab2int):
    """Count <UNK> tokens in data"""
    unk_count = 0
    for token in data:
        if token == vocab2int['<UNK>']:
            unk_count += 1
    return unk_count
#for getting the length of each paragraph
def build_summary(data):
    """
    Build pandas data frame summary for dataset, useful for finding the length
    our sequence should be
    """
    summary = []
    for text in data:
        summary.append(len(text))
    return pd.DataFrame(summary, columns=['counts'])
def BLEU(candidate, references):
    precisions = []
    for i in range(4):
        pr, bp = count_ngram(candidate, references, i+1)
        precisions.append(pr)
    bleu = geometric_mean(precisions) * bp
    return bleu


def remove_wrong_length_data(coverted_inputs, converted_targets, vocab2int,
                             start_inputs_length, max_inputs_length, max_targets_length, 
                             min_inputs_length=5, min_targets_lengths=5,
                             unk_inputs_limit=1, unk_targets_limit=0):
    """
    Sort the paragraphs and questions by the length of their texts, shortest to longest
    Limit the length of summaries and texts based on the min and max ranges.
    This step is important especially if some of the texts in the corpus provided
    are very long, compared to others. For long texts, it would take too much
    memory to learn sequence, thus we want to avoid those. Short sequences
    might act as unneccesary noise in the learning process.
    """
    sorted_inputs = []
    sorted_targets = []
    
    print('Doing final preprocessing - sorting the texts and keeping only those ' + \
          'of appropriate length.')
    for length in range(start_inputs_length, max_inputs_length): 
        for index, words in enumerate(converted_targets):
            if (len(converted_targets[index]) >= min_targets_lengths and
                len(converted_targets[index]) <= max_targets_length and
                len(coverted_inputs[index]) >= min_inputs_length and
                unk_counter(converted_targets[index], vocab2int) <= unk_targets_limit and
                unk_counter(coverted_inputs[index], vocab2int) <= unk_inputs_limit and
                length == len(coverted_inputs[index])
               ):
                sorted_targets.append(converted_targets[index])
                sorted_inputs.append(coverted_inputs[index])
        
    # Ensure the lenght os sorten paragraph and questions match
    assert len(sorted_inputs) == len(sorted_targets)
    print('Got {} inputs/targets pairs!'.format(len(sorted_inputs)))
    
    return sorted_inputs, sorted_targets
file = "C:\\Users\\user\\AppData\\Local\\Programs\\Python\\Python36\\glove.6B.300d.txt"
model= loadGloveModel(file)
#print((model["your"]))
word_frequency = {} # Use of word frequency???
#print(count_word_frequency(word_frequency, raw))
#for getting the word frequency in the text file
word_frequency=count_word_frequency(word_frequency, raw)
print(word_frequency)
vocab2int, int2vocab = create_conversion_dictionaries(word_frequency, model) 
#print(vocab2int)
word_embedding_matrix = create_embedding_matrix(vocab2int, model)
#word_embedding_matrix = create_embedding_matrix(vocab2int, embeddings_index)
#print((word_embedding_matrix[]))
word_count = 0
unk_count = 0
print('Converting text to integers')
converted_inputs, word_count, unk_count = convert_data_to_ints(raw, vocab2int,word_count,unk_count)
converted_targets, word_count, unk_count = convert_data_to_ints(raw, vocab2int,word_count,unk_count)
#print(word_frequency)
#print(word_embedding_matrix)
summary_inputs = build_summary(converted_inputs)
#print(summary_inputs)
summary_targets = build_summary(converted_targets)
#print(summary_targets)
sorted_inputs, sorted_targets = remove_wrong_length_data(converted_inputs,
                                                         converted_targets,vocab2int,start_inputs_length=min(summary_inputs.counts),max_inputs_length=int(np.percentile(summary_inputs.counts, 100)),
                                                         max_targets_length=int(np.percentile(summary_targets.counts, 100)),
                                                         min_inputs_length=5,
                                                         min_targets_lengths=5,
                                                         unk_inputs_limit=1,
                                                         unk_targets_limit=0)
enc_inputs = sorted_inputs
dec_targets = sorted_targets
print(enc_inputs)
assert len(enc_inputs) == len(dec_targets)
assert len(vocab2int) == len(int2vocab)
epochs = 1
batch_size = 128
rnn_size = 512
num_layers = 2
learning_rate = 0.005
keep_probability = 0.8     
beam_width = 20
print('Building graph')
# Build the graph
train_graph = tf.Graph()
from nltk import bleu_score

# Set the graph to default to ensure that it is ready for training
with train_graph.as_default():
    
    # Load the model inputs    
    input_data, targets, lr, keep_prob, target_length, max_target_length, input_length = model1.model_inputs()

    # Create the training and inference logits
    training_logits, inference_logits = model1.seq2seq_model(tf.reverse(input_data, [-1]),
                                                            targets, 
                                                            keep_prob,   
                                                            input_length,
                                                            target_length,
                                                            max_target_length,
                                                            len(vocab2int)+1,
                                                            rnn_size, 
                                                            num_layers, 
                                                            vocab2int,
                                                            word_embedding_matrix,
                                                            batch_size,
                                                            beam_width)
    #print(training_logits)
    #print(inference_logits)
	# Create tensors for the training logits and inference logits
    training_logits = tf.identity(training_logits.rnn_output, 'logits')
    inference_logits = tf.identity(inference_logits.predicted_ids, name='predictions')
    #print(training_logits)
    #print(inference_logits)
    # Create the weights for sequence_loss
    masks = tf.sequence_mask(target_length, max_target_length, dtype=tf.float32, name='masks')
    #loss='bleu'
	#loss = BLEU(candidate, references)	
    #predictions = inference_logits.predicted_ids

    #predictions_ = tf.identity(predictions,name="predicitions")
    print(inference_logits)
    with tf.name_scope("optimization"):
	    
        #bleu=BLEU(raw, raw)
        #softmax_loss_function=bleu		
        # Loss function
        #bleu=bleu_score.corpus_bleu(training_logits,targets)
        cost = tf.contrib.seq2seq.sequence_loss(training_logits,targets,masks)
		
        #cost=compute_bleu(sorted_inputs,sorted_targets)
			#,softmax_loss_function=bleu)

        # Optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate)

        # Gradient Clipping
        gradients = optimizer.compute_gradients(cost)
        capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
        train_op = optimizer.apply_gradients(capped_gradients)
print("Graph is built.")

#==============================================================================
# Train the model
#==============================================================================
learning_rate_decay = 0.95
min_learning_rate = 0.0005
display_step = 20 # Check training loss after every 20 batches
stop_early = 0 
stop = 3 # If the update loss does not decrease in 3 consecutive update checks, stop training
per_epoch = 3 # Make 3 update checks per epoch
update_check = (len(enc_inputs)//batch_size//per_epoch)-1

update_loss = 0 
batch_loss = 0
# Record the update losses for saving improvements in the model
question_update_loss = [] 
checkpoint_dir = 'ckpt' 
checkpoint_path = os.path.join(checkpoint_dir, 'model.ckpt')
restore = 0
#print(embeddings_index)
#print()
print('Initializing session and training')
with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver() 
    
    # If we want to continue training a previous session
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and restore:
        print('Restoring old model parameters from %s...' % ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
    
    for epoch_i in range(1, epochs+1):
        update_loss = 0
        batch_loss = 0
        for batch_i, (targets_batch, inputs_batch, targets_lengths, inputs_lengths) in enumerate(
                model1.get_batches(dec_targets, enc_inputs, vocab2int, batch_size)):
            start_time = time.time()
            _, loss = sess.run(
                [train_op, cost],
                {input_data: inputs_batch,
                 targets: targets_batch,
                 lr: learning_rate,
                 target_length: targets_lengths,
                 input_length: inputs_lengths,
                 keep_prob: keep_probability})

            batch_loss += loss
            update_loss += loss
            end_time = time.time()
            batch_time = end_time - start_time

            if batch_i % display_step == 0:
                print('Epoch {:>3}/{} Batch {:>4}/{} - Loss: {:>6.3f}, Seconds: {:>4.2f}'
                      .format(epoch_i,
                              epochs, 
                              batch_i, 
                              len(enc_inputs) // batch_size, 
                              batch_loss / display_step, 
                              batch_time*display_step))
                batch_loss = 0

            if batch_i % update_check == 0 and batch_i > 0:
                print("Average loss for this update:", round(update_loss/update_check, 3))
                question_update_loss.append(update_loss)
                
                # If the update loss is at a new minimum, save the model
                if update_loss <= min(question_update_loss):
                    print('New Record! Saving the model.') 
                    stop_early = 0
                    saver.save(sess, checkpoint_path)

                else:
                    print("No Improvement.")
                    stop_early += 1
                    if stop_early == stop:
                        break
                update_loss = 0
            
        # Reduce learning rate, but not below its minimum value
        learning_rate *= learning_rate_decay
        if learning_rate < min_learning_rate:
            learning_rate = min_learning_rate
        
        if stop_early == stop:
            print("Stopping Training.")
            break
			
random_example = randint(0, len(raw))
input_sequence = raw[random_example]
#text=text_to_seq(input_sequence)
text, word_count1, unk_count1 = convert_data_to_ints(input_sequence, vocab2int,word_count,unk_count)
# Set hyperparameters (same as training)
epochs = 1
batch_size = 128
rnn_size = 512
num_layers = 2
learning_rate = 0.005
keep_probability = 0.75     
beam_width = 3

#text = text_to_seq(input_sequence)
checkpoint_path = 'ckpt/model.ckpt'

loaded_graph = tf.Graph()
with tf.Session(graph=loaded_graph) as sess:
    # Load saved model
    try:
        print('Restoring old model from %s...' % checkpoint_path)
        loader = tf.train.import_meta_graph(checkpoint_path + '.meta')
        loader.restore(sess, checkpoint_path)
    except: 
        raise 'Checkpoint directory not found!'

    input_data = loaded_graph.get_tensor_by_name('input:0')
    logits = loaded_graph.get_tensor_by_name('predictions:0')
    input_length = loaded_graph.get_tensor_by_name('input_length:0')
    target_length = loaded_graph.get_tensor_by_name('target_length:0')
    keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')
    
    #Multiply by batch_size to match the model's input parameters
    answer_logits = sess.run(logits, {input_data: [text]*batch_size, 
                                      target_length: [25], 
                                      input_length: [len(text)]*batch_size,
                                      keep_prob: 1.0})

# Remove the padding from the tweet
pad = vocab2int["<PAD>"] 
new_logits = []
for i in range(batch_size):
    new_logits.append(answer_logits[i].T)

print('Original Text:', input_sequence)

print('\nGenerated Questions:')
for index in range(beam_width):
    print(' -- : {}'.format(" ".join([int2vocab[i] for i in new_logits[1][index] if i != pad and i != -1])))
import numpy as np
print(len(vocab2int))
print(len(word_embedding_matrix[0]))

