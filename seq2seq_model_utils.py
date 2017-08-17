from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

from config import FLAGS, BUCKETS
import data_utils
import seq2seq_model
#from config import global_memory,global_output

#global global_memory,global_output
def create_model(session, forward_only,batch_size=None):
  """Create translation model and initialize or load parameters in session."""
  model = seq2seq_model.Seq2SeqModel(
      vocab_size=FLAGS.vocab_size,
      embedding_dim=FLAGS.embedding_dim,
      buckets=BUCKETS,
      size=FLAGS.size,
      num_layers=FLAGS.num_layers,
      max_gradient_norm=FLAGS.max_gradient_norm,
      batch_size=FLAGS.batch_size if not batch_size else batch_size,
      learning_rate=FLAGS.learning_rate,
      learning_rate_decay_factor=FLAGS.learning_rate_decay_factor,
      use_lstm=True,
      forward_only=forward_only)

  ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
  #print('path:',ckpt.model_checkpoint_path)
  #print('gfile:',gfile.Exists(ckpt.model_checkpoint_path))
  #if ckpt and gfile.Exists(ckpt.model_checkpoint_path):
  if ckpt:
    print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
    model.saver.restore(session, ckpt.model_checkpoint_path)
  else:
    print("Created model with fresh parameters.")
    session.run(tf.global_variables_initializer())
  return model


def get_predicted_sentence(input_sentence, vocab, rev_vocab, model, sess):
    input_token_ids = data_utils.sentence_to_token_ids(input_sentence, vocab)
    print(input_token_ids)
    # Which bucket does it belong to?
    if len(input_token_ids)>=BUCKETS[-1][0]:
      input_token_ids = input_token_ids[:BUCKETS[-1][0]-1]
    bucket_id = min([b for b in xrange(len(BUCKETS)) if BUCKETS[b][0] > len(input_token_ids)])
    outputs = []

    feed_data = {bucket_id: [(input_token_ids, outputs)]}
    # Get a 1-element batch to feed the sentence to the model.
    encoder_inputs, decoder_inputs, target_weights = model.get_batch(feed_data, bucket_id)
    global_memory['inp']=1
    # Get output logits for the sentence.
    _,_,output_logits = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, forward_only=True,beam_search=True)
    #print('global_output:')
    #print(global_output)
    outputs = []
    # This is a greedy decoder - outputs are just argmaxes of output_logits.
    for logit in output_logits:
        selected_token_id = int(np.argmax(logit, axis=1))
        if selected_token_id == data_utils.EOS_ID:
            	break
        else:
           	outputs.append(selected_token_id)
    # Forming output sentence on natural language
    outputs = ' '.join([rev_vocab[i] for i in outputs])

    return outputs
def beam_search(input_sentence,vocab,revocab,model,sess,beam_size=10):
    input_token_ids = data_utils.sentence_to_token_ids(input_sentence, vocab)
    # Which bucket does it belong to?
    if len(input_token_ids)>=BUCKETS[-1][0]:
      input_token_ids = input_token_ids[:BUCKETS[-1][0]-1]
    bucket_id = min([b for b in xrange(len(BUCKETS)) if BUCKETS[b][0] > len(input_token_ids)])
    def func(decoder_tokens,position):
      feed_data = {bucket_id: [(input_token_ids, decoder_tokens)]}
      # Get a 1-element batch to feed the sentence to the model.
      encoder_inputs, decoder_inputs, target_weights = model.get_batch(feed_data, bucket_id)
      #print('ei:',encoder_inputs)
      #print('de:',decoder_inputs)
      #print('ps:',position)
      # Get output logits for the sentence.
      _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id,forward_only=False,position=position)
      #print(output_logits)
      return output_logits
      # This is a greedy decoder - outputs are just argmaxes of output_logits.
    outputs,logP = [],[]
    i = 0
    while i<BUCKETS[bucket_id][1] and len(outputs)<beam_size:
      if i==0:
        prob = func([],i)
        #print(prob)
        prob = np.log(np.reshape(prob,[-1]))
        scores,next_tokens = sess.run(tf.nn.top_k(prob,beam_size))
        #print(scores)
        beams = [[next_tokens[j]] for j in range(beam_size)]
        i += 1
        continue
      table = []
      for ind,candidate in enumerate(beams):
        prob = func(candidate,i)
        #print(type(prob))
        #print(prob)
        table.append(scores[ind]+np.log(prob))
      table = np.reshape(np.array(table),[-1])
      scores,next_tokens = sess.run(tf.nn.top_k(table,beam_size-len(outputs)))
      #print(scores)
      #print('next_tokens',next_tokens)
      parent = next_tokens//FLAGS.vocab_size
      #print('parent',parent)
      next_tokens = next_tokens%FLAGS.vocab_size
      #print('next_tokens',next_tokens)
      beams_ = []
      #print('parent shape:',parent.shape)
      #print(len(beams))
      #print(parent)
      for j,token in enumerate(next_tokens):
        #print('j:',j)
        if token==data_utils.EOS_ID:
          outputs.append(beams[parent[j]])
          logP.append(scores[j])
        else:
          beams_.append(beams[parent[j]]+[token])
      i += 1
      beams = beams_
    if len(outputs)==0:
      outputs = beams
      logP = scores
    print(i,len(outputs))
    logP = [ele/len(outputs[ind]) for ind,ele in enumerate(logP)]
    best_ind = np.argmax(logP)  
    outputs = [' '.join([revocab[ele] for ele in output]) for output in outputs]
    best = outputs[best_ind]
    # Forming output sentence on natural language
    #outputs = '\n'.join(outputs)
    #return None,outputs,logP
    return best,outputs,logP



def get_predicted_sentences(input_sentences, vocab, rev_vocab, model, sess):
    feed_data = [[] for i in range(len(BUCKETS))]
    map_dict = []
    for ind,input_sentence in enumerate(input_sentences):
      input_token_ids = data_utils.sentence_to_token_ids(input_sentence, vocab)
      if len(input_token_ids)>=BUCKETS[-1][0]:
        input_token_ids = input_token_ids[:BUCKETS[-1][0]-1]

    # Which bucket does it belong to?
      bucket_id = min([b for b in range(len(BUCKETS)) if BUCKETS[b][0] > len(input_token_ids)])
      outputs = []
      map_dict.append((bucket_id,len(feed_data[bucket_id])))
      feed_data[bucket_id].append((input_token_ids, outputs))
    print([(i,len(feed_data[i])) for i in range(len(BUCKETS))])
    # Get a 1-element batch to feed the sentence to the model.
    output_sentences = [[] for i in range(len(BUCKETS))]
    batch_size = model.batch_size
    for bucket_id in range(len(BUCKETS)):
      if len(feed_data[bucket_id])==0:
        continue
      tmp = 0
      break_ = False
      while True:
        if (tmp+1)*model.batch_size < len(feed_data[bucket_id]):
          data = {bucket_id:feed_data[bucket_id][tmp*model.batch_size:(tmp+1)*model.batch_size]}
          encoder_inputs, decoder_inputs, target_weights = model.get_batch(data, bucket_id,train_mode=False)
        else:
          break_ = True
          data = {bucket_id:feed_data[bucket_id][tmp*model.batch_size:]}
          print(len(data[bucket_id]))
          model.batch_size = len(data[bucket_id])
          encoder_inputs,decoder_inputs,target_weights = model.get_batch(data,bucket_id,train_mode=False)

    # Get output logits for the sentence.
        _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, forward_only=True)

        outputs_logit = []
        # This is a greedy decoder - outputs are just argmaxes of output_logits.
        for logit in output_logits:
          selected_token_id = np.argmax(logit, axis=1)
          outputs_logit.append(selected_token_id)
        outputs_logit = list(np.array(outputs_logit).T)
        for ele in outputs_logit:
          outputs = []
          for selected_token_id in ele:
            selected_token_id = int(selected_token_id)
            if selected_token_id == data_utils.EOS_ID:
              break
            else:
              outputs.append(selected_token_id)


    # Forming output sentence on natural language
          output_sentence = ' '.join([rev_vocab[output] for output in outputs])           
          #f.write(output_sentence+'\n')     
          output_sentences[bucket_id].append(output_sentence)
        model.batch_size = batch_size
        if break_:
          break
        tmp += 1
    print([(i,len(output_sentences[i])) for i in range(len(BUCKETS))])
    return [output_sentences[ele[0]][ele[1]] for ele in map_dict]
