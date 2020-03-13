from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import sys
# sys.path.append("model/model/")
# import os
# os.chdir("model/model")
import bert
import collections
import json
import math
import os
import random
import bert.modeling as modeling
import bert.optimization as optimization
import bert.tokenization as tokenization
import six
import tensorflow as tf

# flags = tf.flags

# FLAGS = flags.FLAGS
_estimator=None
_tokenizer=None
_vocab_file="model/chinese_L-12_H-768_A-12/vocab.txt" 
_bert_config_file="model/chinese_L-12_H-768_A-12/bert_config.json" 
_init_checkpoint="model/chinese_L-12_H-768_A-12/bert_model.ckpt" 
_do_train=False, 
_train_file="model/SQAD/DRCD/DRCD_training.json" 
_do_predict=True, 
_predict_file="model/SQAD/DRCD/my_test.json" 
_train_batch_size=12 
_learning_rate=3e-5 
_num_train_epochs=2.0,
_max_seq_length=320 
_doc_stride=128,
_output_dir="model/qa_modelsize12_aptg_20200205"
_use_tpu=False
_tpu_name=None
_tpu_zone=None
_version_2_with_negative=False
_do_lower_case =True
_num_tpu_cores =8 #
_master=None
_predict_batch_size=8
_max_query_length=64
_iterations_per_loop=1000
_save_checkpoints_steps=1000
_n_best_size=20
_max_answer_length=30
_warmup_proportion=0.1
_gcp_project=None
_verbose_logging=False
_null_score_diff_threshold=0.0

class SquadExample(object):
    """A single training/test example for simple sequence classification.

     For examples without an answer, the start and end position are -1.
    """

    def __init__(self,
               qas_id,
               question_text,
               doc_tokens,
               orig_answer_text=None,
               start_position=None,
               end_position=None,
               is_impossible=False):
        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % (tokenization.printable_text(self.qas_id))
        s += ", question_text: %s" % (
            tokenization.printable_text(self.question_text))
        s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
        if self.start_position:
          s += ", start_position: %d" % (self.start_position)
        if self.start_position:
          s += ", end_position: %d" % (self.end_position)
        if self.start_position:
          s += ", is_impossible: %r" % (self.is_impossible)
        return s


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
               unique_id,
               example_index,
               doc_span_index,
               tokens,
               token_to_orig_map,
               token_is_max_context,
               input_ids,
               input_mask,
               segment_ids,
               start_position=None,
               end_position=None,
               is_impossible=None):
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible


def read_squad_examples(input_file, is_training):
    """Read a SQuAD json file into a list of SquadExample."""
    with tf.gfile.Open(input_file, "r") as reader:
        input_data = json.load(reader)["data"]

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    examples = []
    for entry in input_data:
        for paragraph in entry["paragraphs"]:
            paragraph_text = paragraph["context"]
            doc_tokens = []
            char_to_word_offset = []
            prev_is_whitespace = True
            for c in paragraph_text:
                if is_whitespace(c):
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace:
                        doc_tokens.append(c)
                    else:
                        doc_tokens[-1] += c
                    prev_is_whitespace = False
                char_to_word_offset.append(len(doc_tokens) - 1)

    for qa in paragraph["qas"]:
        qas_id = qa["id"]
        question_text = qa["question"]
        start_position = None
        end_position = None
        orig_answer_text = None
        is_impossible = False
        if is_training:

            if _version_2_with_negative:
                is_impossible = qa["is_impossible"]
            if (len(qa["answers"]) != 1) and (not is_impossible):
                raise ValueError(
                    "For training, each question should have exactly 1 answer.")
            if not is_impossible:
                answer = qa["answers"][0]
                orig_answer_text = answer["text"]
                answer_offset = answer["answer_start"]
                answer_length = len(orig_answer_text)
                start_position = char_to_word_offset[answer_offset]
                end_position = char_to_word_offset[answer_offset + answer_length -
                                                   1]
                # Only add answers where the text can be exactly recovered from the
                # document. If this CAN'T happen it's likely due to weird Unicode
                # stuff so we will just skip the example.
                #
                # Note that this means for training mode, every example is NOT
                # guaranteed to be preserved.
                actual_text = " ".join(
                    doc_tokens[start_position:(end_position + 1)])
                cleaned_answer_text = " ".join(
                    tokenization.whitespace_tokenize(orig_answer_text))
                if actual_text.find(cleaned_answer_text) == -1:
                    tf.logging.warning("Could not find answer: '%s' vs. '%s'",
                                     actual_text, cleaned_answer_text)
                    continue
            else:
                start_position = -1
                end_position = -1
                orig_answer_text = ""

        example = SquadExample(
            qas_id=qas_id,
            question_text=question_text,
            doc_tokens=doc_tokens,
            orig_answer_text=orig_answer_text,
            start_position=start_position,
            end_position=end_position,
            is_impossible=is_impossible)
        examples.append(example)

    return examples


def convert_examples_to_features(examples, tokenizer, max_seq_length,
                                 doc_stride, max_query_length, is_training,
                                 output_fn):
    """Loads a data file into a list of `InputBatch`s."""

    unique_id = 1000000000

    for (example_index, example) in enumerate(examples):
        query_tokens = tokenizer.tokenize(example.question_text)

        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]

        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        for (i, token) in enumerate(example.doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)

        tok_start_position = None
        tok_end_position = None
        if is_training and example.is_impossible:
            tok_start_position = -1
            tok_end_position = -1
        if is_training and not example.is_impossible:
            tok_start_position = orig_to_tok_index[example.start_position]
            if example.end_position < len(example.doc_tokens) - 1:
                tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
            else:
                tok_end_position = len(all_doc_tokens) - 1
            (tok_start_position, tok_end_position) = _improve_answer_span(
              all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
              example.orig_answer_text)

        # The -3 accounts for [CLS], [SEP] and [SEP]
        max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

        # We can have documents that are longer than the maximum sequence length.
        # To deal with this we do a sliding window approach, where we take chunks
        # of the up to our max length with a stride of `doc_stride`.
        _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
            "DocSpan", ["start", "length"])
        doc_spans = []
        start_offset = 0
        while start_offset < len(all_doc_tokens):
            length = len(all_doc_tokens) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            if start_offset + length == len(all_doc_tokens):
                break
            start_offset += min(length, doc_stride)

        for (doc_span_index, doc_span) in enumerate(doc_spans):
            tokens = []
            token_to_orig_map = {}
            token_is_max_context = {}
            segment_ids = []
            tokens.append("[CLS]")
            segment_ids.append(0)
            for token in query_tokens:
                tokens.append(token)
                segment_ids.append(0)
            tokens.append("[SEP]")
            segment_ids.append(0)

            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

                is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                                       split_token_index)
                token_is_max_context[len(tokens)] = is_max_context
                tokens.append(all_doc_tokens[split_token_index])
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            start_position = None
            end_position = None
            if is_training and not example.is_impossible:
                # For training, if our document chunk does not contain an annotation
                # we throw it out, since there is nothing to predict.
                doc_start = doc_span.start
                doc_end = doc_span.start + doc_span.length - 1
                out_of_span = False
                if not (tok_start_position >= doc_start and
                        tok_end_position <= doc_end):
                    out_of_span = True
                if out_of_span:
                    start_position = 0
                    end_position = 0
                else:
                    doc_offset = len(query_tokens) + 2
                    start_position = tok_start_position - doc_start + doc_offset
                    end_position = tok_end_position - doc_start + doc_offset

            if is_training and example.is_impossible:
                start_position = 0
                end_position = 0

            if example_index < 20:
                tf.logging.info("*** Example ***")
                tf.logging.info("unique_id: %s" % (unique_id))
                tf.logging.info("example_index: %s" % (example_index))
                tf.logging.info("doc_span_index: %s" % (doc_span_index))
                tf.logging.info("tokens: %s" % " ".join(
                    [tokenization.printable_text(x) for x in tokens]))
                tf.logging.info("token_to_orig_map: %s" % " ".join(
                    ["%d:%d" % (x, y) for (x, y) in six.iteritems(token_to_orig_map)]))
                tf.logging.info("token_is_max_context: %s" % " ".join([
                    "%d:%s" % (x, y) for (x, y) in six.iteritems(token_is_max_context)
                ]))
                tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                tf.logging.info(
                    "input_mask: %s" % " ".join([str(x) for x in input_mask]))
                tf.logging.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                if is_training and example.is_impossible:
                    tf.logging.info("impossible example")
                if is_training and not example.is_impossible:
                    answer_text = " ".join(tokens[start_position:(end_position + 1)])
                    tf.logging.info("start_position: %d" % (start_position))
                    tf.logging.info("end_position: %d" % (end_position))
                    tf.logging.info(
                      "answer: %s" % (tokenization.printable_text(answer_text)))

            feature = InputFeatures(
              unique_id=unique_id,
              example_index=example_index,
              doc_span_index=doc_span_index,
              tokens=tokens,
              token_to_orig_map=token_to_orig_map,
              token_is_max_context=token_is_max_context,
              input_ids=input_ids,
              input_mask=input_mask,
              segment_ids=segment_ids,
              start_position=start_position,
              end_position=end_position,
              is_impossible=example.is_impossible)

            # Run callback
            output_fn(feature)

            unique_id += 1


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""

    # The SQuAD annotations are character based. We first project them to
    # whitespace-tokenized words. But then after WordPiece tokenization, we can
    # often find a "better match". For example:
    #
    #   Question: What year was John Smith born?
    #   Context: The leader was John Smith (1895-1943).
    #   Answer: 1895
    #
    # The original whitespace-tokenized answer will be "(1895-1943).". However
    # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
    # the exact answer, 1895.
    #
    # However, this is not always possible. Consider the following:
    #
    #   Question: What country is the top exporter of electornics?
    #   Context: The Japanese electronics industry is the lagest in the world.
    #   Answer: Japan
    #
    # In this case, the annotator chose "Japan" as a character sub-span of
    # the word "Japanese". Since our WordPiece tokenizer does not split
    # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
    # in SQuAD, but does happen.
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""

    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 use_one_hot_embeddings):
    """Creates a classification model."""
    model = modeling.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings)

    final_hidden = model.get_sequence_output()

    final_hidden_shape = modeling.get_shape_list(final_hidden, expected_rank=3)
    batch_size = final_hidden_shape[0]
    seq_length = final_hidden_shape[1]
    hidden_size = final_hidden_shape[2]

    output_weights = tf.get_variable(
      "cls/squad/output_weights", [2, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
      "cls/squad/output_bias", [2], initializer=tf.zeros_initializer())

    final_hidden_matrix = tf.reshape(final_hidden,
                                   [batch_size * seq_length, hidden_size])
    logits = tf.matmul(final_hidden_matrix, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)

    logits = tf.reshape(logits, [batch_size, seq_length, 2])
    logits = tf.transpose(logits, [2, 0, 1])

    unstacked_logits = tf.unstack(logits, axis=0)

    (start_logits, end_logits) = (unstacked_logits[0], unstacked_logits[1])

    return (start_logits, end_logits)


def model_fn_builder(bert_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        unique_ids = features["unique_ids"]
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        (start_logits, end_logits) = create_model(
            bert_config=bert_config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            use_one_hot_embeddings=use_one_hot_embeddings)

        tvars = tf.trainable_variables()

        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names
            ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            if use_tpu:

                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                          init_string)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            seq_length = modeling.get_shape_list(input_ids)[1]

            def compute_loss(logits, positions):
                one_hot_positions = tf.one_hot(
                    positions, depth=seq_length, dtype=tf.float32)
                log_probs = tf.nn.log_softmax(logits, axis=-1)
                loss = -tf.reduce_mean(
                    tf.reduce_sum(one_hot_positions * log_probs, axis=-1))
                return loss

            start_positions = features["start_positions"]
            end_positions = features["end_positions"]

            start_loss = compute_loss(start_logits, start_positions)
            end_loss = compute_loss(end_logits, end_positions)

            total_loss = (start_loss + end_loss) / 2.0

            train_op = optimization.create_optimizer(
              total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
              mode=mode,
              loss=total_loss,
              train_op=train_op,
              scaffold_fn=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {
              "unique_ids": unique_ids,
              "start_logits": start_logits,
              "end_logits": end_logits,
            }
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
              mode=mode, predictions=predictions, scaffold_fn=scaffold_fn)
        else:
            raise ValueError(
              "Only TRAIN and PREDICT modes are supported: %s" % (mode))

        return output_spec

    return model_fn


def input_fn_builder(input_file, seq_length, is_training, drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
      "unique_ids": tf.FixedLenFeature([], tf.int64),
      "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
      "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
    }

    if is_training:
        name_to_features["start_positions"] = tf.FixedLenFeature([], tf.int64)
        name_to_features["end_positions"] = tf.FixedLenFeature([], tf.int64)

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t

        return example

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder))

        return d

    return input_fn


RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "start_logits", "end_logits"])


def write_predictions(all_examples, all_features, all_results, n_best_size,
                      max_answer_length, do_lower_case, output_prediction_file,
                      output_nbest_file, output_null_log_odds_file):
    """Write final predictions to the json file and log-odds of null if needed."""
    tf.logging.info("Writing predictions to: %s" % (output_prediction_file))
    tf.logging.info("Writing nbest to: %s" % (output_nbest_file))

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
      "PrelimPrediction",
      ["feature_index", "start_index", "end_index", "start_logit", "end_logit"])

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()

    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]

        prelim_predictions = []
        # keep track of the minimum score of null start+end of position 0
        score_null = 1000000  # large and positive
        min_null_feature_index = 0  # the paragraph slice with min mull score
        null_start_logit = 0  # the start logit at the slice with min null score
        null_end_logit = 0  # the end logit at the slice with min null score
        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]
            start_indexes = _get_best_indexes(result.start_logits, n_best_size)
            end_indexes = _get_best_indexes(result.end_logits, n_best_size)
            # if we could have irrelevant answers, get the min score of irrelevant
            if _version_2_with_negative:
                feature_null_score = result.start_logits[0] + result.end_logits[0]
                if feature_null_score < score_null:
                    score_null = feature_null_score
                    min_null_feature_index = feature_index
                    null_start_logit = result.start_logits[0]
                    null_end_logit = result.end_logits[0]
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.token_to_orig_map:
                        continue
                    if end_index not in feature.token_to_orig_map:
                        continue
                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    prelim_predictions.append(
                      _PrelimPrediction(
                          feature_index=feature_index,
                          start_index=start_index,
                          end_index=end_index,
                          start_logit=result.start_logits[start_index],
                          end_logit=result.end_logits[end_index]))

        if _version_2_with_negative:
            prelim_predictions.appenqd(
              _PrelimPrediction(
                  feature_index=min_null_feature_index,
                  start_index=0,
                  end_index=0,
                  start_logit=null_start_logit,
                  end_logit=null_end_logit))
        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.start_logit + x.end_logit),
            reverse=True)

        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction", ["text", "start_logit", "end_logit"])

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]
            if pred.start_index > 0:  # this is a non-null prediction
                tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
                orig_doc_start = feature.token_to_orig_map[pred.start_index]
                orig_doc_end = feature.token_to_orig_map[pred.end_index]
                orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
                tok_text = " ".join(tok_tokens)

                # De-tokenize WordPieces that have been split off.
                tok_text = tok_text.replace(" ##", "")
                tok_text = tok_text.replace("##", "")

                # Clean whitespace
                tok_text = tok_text.strip()
                tok_text = " ".join(tok_text.split())
                orig_text = " ".join(orig_tokens)

                final_text = get_final_text(tok_text, orig_text, do_lower_case)
                if final_text in seen_predictions:
                    continue

                seen_predictions[final_text] = True
            else:
                final_text = ""
                seen_predictions[final_text] = True

            nbest.append(
              _NbestPrediction(
                  text=final_text,
                  start_logit=pred.start_logit,
                  end_logit=pred.end_logit))

        # if we didn't inlude the empty option in the n-best, inlcude it
        if _version_2_with_negative:
            if "" not in seen_predictions:
                nbest.append(
                    _NbestPrediction(
                        text="", start_logit=null_start_logit,
                        end_logit=null_end_logit))
        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(
              _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

        assert len(nbest) >= 1

        total_scores = []
        best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_logit + entry.end_logit)
            if not best_non_null_entry:
                if entry.text:
                    best_non_null_entry = entry

        probs = _compute_softmax(total_scores)

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
            nbest_json.append(output)

        assert len(nbest_json) >= 1

        if not _version_2_with_negative:
            all_predictions[example.qas_id] = nbest_json[0]["text"]
        else:
            # predict "" iff the null score - the score of best non-null > threshold
            score_diff = score_null - best_non_null_entry.start_logit - (
              best_non_null_entry.end_logit)
            scores_diff_json[example.qas_id] = score_diff
            if score_diff > _null_score_diff_threshold:
                all_predictions[example.qas_id] = ""
            else:
                all_predictions[example.qas_id] = best_non_null_entry.text

        all_nbest_json[example.qas_id] = nbest_json
        write_file=False
        if(write_file == True):
            with tf.gfile.GFile(output_prediction_file, "w") as writer:
                writer.write(json.dumps(all_predictions, indent=4, ensure_ascii=False) + "\n")

            with tf.gfile.GFile(output_nbest_file, "w") as writer:
                writer.write(json.dumps(all_nbest_json, indent=4, ensure_ascii=False) + "\n")

            if _version_2_with_negative:
                with tf.gfile.GFile(output_null_log_odds_file, "w") as writer:
                    writer.write(json.dumps(scores_diff_json, indent=4, ensure_ascii=False ) + "\n")

#     print(all_predictions)
    
    my_counter=0
    for column in all_nbest_json:
        if my_counter ==0:
            # print("column:{}".format(json_data[column]))
            max_string = "\n{}, prob:{}".format(all_nbest_json[column][0]['text'],all_nbest_json[column][0]['probability'])
            max_all_nbest_json = all_nbest_json[column][0]
            break


    # print(all_predictions, max_all_nbest_json)
    # return all_predictions, max_all_nbest_json
    print(max_all_nbest_json)
    return(max_all_nbest_json)
  

# def write_predictions2(all_examples, all_features, all_results, n_best_size,
#                       max_answer_length, do_lower_case, output_prediction_file,
#                       output_nbest_file, output_null_log_odds_file):
#     """Write final predictions to the json file and log-odds of null if needed."""
#     tf.logging.info("Writing predictions to: %s" % (output_prediction_file))
#     tf.logging.info("Writing nbest to: %s" % (output_nbest_file))

#     example_index_to_features = collections.defaultdict(list)
#     for feature in all_features:
#         example_index_to_features[feature.example_index].append(feature)

#     unique_id_to_result = {}
#     for result in all_results:
#         unique_id_to_result[result.unique_id] = result

#     _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
#       "PrelimPrediction",
#       ["feature_index", "start_index", "end_index", "start_logit", "end_logit"])

#     all_predictions = collections.OrderedDict()
#     all_nbest_json = collections.OrderedDict()
#     scores_diff_json = collections.OrderedDict()

#     for (example_index, example) in enumerate(all_examples):
#         features = example_index_to_features[example_index]

#         prelim_predictions = []
#         # keep track of the minimum score of null start+end of position 0
#         score_null = 1000000  # large and positive
#         min_null_feature_index = 0  # the paragraph slice with min mull score
#         null_start_logit = 0  # the start logit at the slice with min null score
#         null_end_logit = 0  # the end logit at the slice with min null score
#         for (feature_index, feature) in enumerate(features):
#             result = unique_id_to_result[feature.unique_id]
#             start_indexes = _get_best_indexes(result.start_logits, n_best_size)
#             end_indexes = _get_best_indexes(result.end_logits, n_best_size)
#             # if we could have irrelevant answers, get the min score of irrelevant
#             if _version_2_with_negative:
#                 feature_null_score = result.start_logits[0] + result.end_logits[0]
#                 if feature_null_score < score_null:
#                     score_null = feature_null_score
#                     min_null_feature_index = feature_index
#                     null_start_logit = result.start_logits[0]
#                     null_end_logit = result.end_logits[0]
#             for start_index in start_indexes:
#                 for end_index in end_indexes:
#                     # We could hypothetically create invalid predictions, e.g., predict
#                     # that the start of the span is in the question. We throw out all
#                     # invalid predictions.
#                     if start_index >= len(feature.tokens):
#                         continue
#                     if end_index >= len(feature.tokens):
#                         continue
#                     if start_index not in feature.token_to_orig_map:
#                         continue
#                     if end_index not in feature.token_to_orig_map:
#                         continue
#                     if not feature.token_is_max_context.get(start_index, False):
#                         continue
#                     if end_index < start_index:
#                         continue
#                     length = end_index - start_index + 1
#                     if length > max_answer_length:
#                         continue
#                     prelim_predictions.append(
#                       _PrelimPrediction(
#                           feature_index=feature_index,
#                           start_index=start_index,
#                           end_index=end_index,
#                           start_logit=result.start_logits[start_index],
#                           end_logit=result.end_logits[end_index]))

#         if _version_2_with_negative:
#             prelim_predictions.append(
#               _PrelimPrediction(
#                   feature_index=min_null_feature_index,
#                   start_index=0,
#                   end_index=0,
#                   start_logit=null_start_logit,
#                   end_logit=null_end_logit))
#         prelim_predictions = sorted(
#             prelim_predictions,
#             key=lambda x: (x.start_logit + x.end_logit),
#             reverse=True)

#         _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
#             "NbestPrediction", ["text", "start_logit", "end_logit"])

#         seen_predictions = {}
#         nbest = []
#         for pred in prelim_predictions:
#             if len(nbest) >= n_best_size:
#                 break
#             feature = features[pred.feature_index]
#             if pred.start_index > 0:  # this is a non-null prediction
#                 tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
#                 orig_doc_start = feature.token_to_orig_map[pred.start_index]
#                 orig_doc_end = feature.token_to_orig_map[pred.end_index]
#                 orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
#                 tok_text = " ".join(tok_tokens)

#                 # De-tokenize WordPieces that have been split off.
#                 tok_text = tok_text.replace(" ##", "")
#                 tok_text = tok_text.replace("##", "")

#                 # Clean whitespace
#                 tok_text = tok_text.strip()
#                 tok_text = " ".join(tok_text.split())
#                 orig_text = " ".join(orig_tokens)

#                 final_text = get_final_text(tok_text, orig_text, do_lower_case)
#                 if final_text in seen_predictions:
#                     continue

#                 seen_predictions[final_text] = True
#             else:
#                 final_text = ""
#                 seen_predictions[final_text] = True

#             nbest.append(
#               _NbestPrediction(
#                   text=final_text,
#                   start_logit=pred.start_logit,
#                   end_logit=pred.end_logit))

#         # if we didn't inlude the empty option in the n-best, inlcude it
#         if _version_2_with_negative:
#             if "" not in seen_predictions:
#                 nbest.append(
#                     _NbestPrediction(
#                         text="", start_logit=null_start_logit,
#                         end_logit=null_end_logit))
#         # In very rare edge cases we could have no valid predictions. So we
#         # just create a nonce prediction in this case to avoid failure.
#         if not nbest:
#             nbest.append(
#               _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

#         assert len(nbest) >= 1

#         total_scores = []
#         best_non_null_entry = None
#         for entry in nbest:
#             total_scores.append(entry.start_logit + entry.end_logit)
#             if not best_non_null_entry:
#                 if entry.text:
#                     best_non_null_entry = entry

#         probs = _compute_softmax(total_scores)

#         nbest_json = []
#         for (i, entry) in enumerate(nbest):
#             output = collections.OrderedDict()
#             output["text"] = entry.text
#             output["probability"] = probs[i]
#             output["start_logit"] = entry.start_logit
#             output["end_logit"] = entry.end_logit
#             nbest_json.append(output)

#         assert len(nbest_json) >= 1

#         if not _version_2_with_negative:
#             all_predictions[example.qas_id] = nbest_json[0]["text"]
#         else:
#             # predict "" iff the null score - the score of best non-null > threshold
#             score_diff = score_null - best_non_null_entry.start_logit - (
#               best_non_null_entry.end_logit)
#             scores_diff_json[example.qas_id] = score_diff
#             if score_diff > _null_score_diff_threshold:
#                 all_predictions[example.qas_id] = ""
#             else:
#                 all_predictions[example.qas_id] = best_non_null_entry.text

#         all_nbest_json[example.qas_id] = nbest_json
#         write_file=False
#         if(write_file == True):
#             with tf.gfile.GFile(output_prediction_file, "w") as writer:
#                 writer.write(json.dumps(all_predictions, indent=4, ensure_ascii=False) + "\n")

#             with tf.gfile.GFile(output_nbest_file, "w") as writer:
#                 writer.write(json.dumps(all_nbest_json, indent=4, ensure_ascii=False) + "\n")

#             if _version_2_with_negative:
#                 with tf.gfile.GFile(output_null_log_odds_file, "w") as writer:
#                     writer.write(json.dumps(scores_diff_json, indent=4, ensure_ascii=False ) + "\n")

#     print(all_predictions)
#     return all_predictions


def get_final_text(pred_text, orig_text, do_lower_case):
    """Project the tokenized prediction back to the original text."""

    # When we created the data, we kept track of the alignment between original
    # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
    # now `orig_text` contains the span of our original text corresponding to the
    # span that we predicted.
    #
    # However, `orig_text` may contain extra characters that we don't want in
    # our prediction.
    #
    # For example, let's say:
    #   pred_text = steve smith
    #   orig_text = Steve Smith's
    #
    # We don't want to return `orig_text` because it contains the extra "'s".
    #
    # We don't want to return `pred_text` because it's already been normalized
    # (the SQuAD eval script also does punctuation stripping/lower casing but
    # our tokenizer does additional normalization like stripping accent
    # characters).
    #
    # What we really want to return is "Steve Smith".
    #
    # Therefore, we have to apply a semi-complicated alignment heruistic between
    # `pred_text` and `orig_text` to get a character-to-charcter alignment. This
    # can fail in certain cases in which case we just return `orig_text`.

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    # We first tokenize `orig_text`, strip whitespace from the result
    # and `pred_text`, and check if they are the same length. If they are
    # NOT the same length, the heuristic has failed. If they are the same
    # length, we assume the characters are one-to-one aligned.
    tokenizer = tokenization.BasicTokenizer(do_lower_case=do_lower_case)

    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        if _verbose_logging:
            tf.logging.info(
              "Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        if _verbose_logging:
              tf.logging.info("Length not equal after stripping spaces: '%s' vs '%s'",
                              orig_ns_text, tok_ns_text)
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in six.iteritems(tok_ns_to_s_map):
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        if _verbose_logging:
            tf.logging.info("Couldn't map start position")
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        if _verbose_logging:
            tf.logging.info("Couldn't map end position")
        return orig_text

    output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return output_text


def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs


class FeatureWriter(object):
    """Writes InputFeature to TF example file."""

    def __init__(self, filename, is_training):
        self.filename = filename
        self.is_training = is_training
        self.num_features = 0
        self._writer = tf.python_io.TFRecordWriter(filename)

    def process_feature(self, feature):
        """Write a InputFeature to the TFRecordWriter as a tf.train.Example."""
        self.num_features += 1

        def create_int_feature(values):
            feature = tf.train.Feature(
              int64_list=tf.train.Int64List(value=list(values)))
            return feature

        features = collections.OrderedDict()
        features["unique_ids"] = create_int_feature([feature.unique_id])
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)

        if self.is_training:
            features["start_positions"] = create_int_feature([feature.start_position])
            features["end_positions"] = create_int_feature([feature.end_position])
            impossible = 0
            if feature.is_impossible:
                impossible = 1
            features["is_impossible"] = create_int_feature([impossible])

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        self._writer.write(tf_example.SerializeToString())

    def close(self):
        self._writer.close()


def validate_flags_or_throw(bert_config):
    """Validate the input FLAGS or throw an exception."""
    tokenization.validate_case_matches_checkpoint(_do_lower_case,
                                                _init_checkpoint)

    if not _do_train and not _do_predict:
        raise ValueError("At least one of `do_train` or `do_predict` must be True.")

    if _do_train:
        if not _train_file:
            raise ValueError(
              "If `do_train` is True, then `train_file` must be specified.")
    if _do_predict:
        if not _predict_file:
            raise ValueError(
              "If `do_predict` is True, then `predict_file` must be specified.")

    if _max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (_max_seq_length, bert_config.max_position_embeddings))

    if _max_seq_length <= _max_query_length + 3:
        raise ValueError(
            "The max_seq_length (%d) must be greater than max_query_length "
            "(%d) + 3" % (_max_seq_length, _max_query_length))




def do_inference(_qas_id,
        _question_text,
        _doc_tokens,
     __vocab_file="model/chinese_L-12_H-768_A-12/vocab.txt", 
    __bert_config_file="model/chinese_L-12_H-768_A-12/bert_config.json", 
    __init_checkpoint="model/chinese_L-12_H-768_A-12/bert_model.ckpt", 
    __do_train=False, 
    __train_file="model/SQAD/DRCD/DRCD_training.json", 
    __do_predict=True, 
    __predict_file="model/SQAD/DRCD/my_test.json", 
    __train_batch_size=12, 
    __learning_rate=3e-5, 
    __num_train_epochs=2.0, 
    __max_seq_length=320, 
    __doc_stride=128, 
    __output_dir="model/qa_modelsize12_aptg_20200205",
    __use_tpu=False,
    __tpu_name=None,
    __tpu_zone=None,
    __version_2_with_negative=False,
    __do_lower_case =True,
    __num_tpu_cores =8, #
    __master=None,
    __predict_batch_size=8,
    __max_query_length=64,
    __iterations_per_loop=1000,
    __save_checkpoints_steps=1000,
    __n_best_size=20,
    __max_answer_length=30,
    __warmup_proportion=0.1,
    __gcp_project=None,
    __verbose_logging=False,
    __null_score_diff_threshold=0.0
                ):
    
#     global _vocab_file 
    global _vocab_file
    global _bert_config_file
    global _init_checkpoint 
    global _do_train
    global  _train_file
    global _do_predict
    global _predict_file
    global _train_batch_size
    global _learning_rate
    global _num_train_epochs 
    global _max_seq_length
    global _doc_stride
    global _output_dir
    global _use_tpu
    global _tpu_name
    global _tpu_zone
    global _version_2_with_negative
    global _do_lower_case
    global _num_tpu_cores
    global _master
    global _predict_batch_size
    global _max_query_length
    global _iterations_per_loop
    global _save_checkpoints_steps
    global _n_best_size
    global _max_answer_length
    global _warmup_proportion
    global _gcp_project
    global _verbose_logging
    global _null_score_diff_threshold
    
    _vocab_file = __vocab_file
    _bert_config_file=__bert_config_file
    _init_checkpoint = __init_checkpoint 
    _do_train = __do_train
    _train_file = __train_file 
    _do_predict = __do_predict
    _predict_file = __predict_file 
    _train_batch_size =  __train_batch_size
    _learning_rate= __learning_rate 
    _num_train_epochs = __num_train_epochs 
    _max_seq_length = __max_seq_length 
    _doc_stride = __doc_stride
    _output_dir = __output_dir
    _use_tpu = __use_tpu
#     _tpu_name = ___tpu_name
#     _tpu_zone = __tpu_zone
    _version_2_with_negative=_version_2_with_negative = __version_2_with_negative=_version_2_with_negative
    _do_lower_case = __do_lower_case
    _num_tpu_cores = __num_tpu_cores #
    _master = __master
    _predict_batch_size = __predict_batch_size
    _max_query_length = __max_query_length
    _iterations_per_loop = __iterations_per_loop
    _save_checkpoints_steps = __save_checkpoints_steps
    _n_best_size = __n_best_size
    _max_answer_length = __max_answer_length
    _warmup_proportion = __warmup_proportion
    _gcp_project = __gcp_project
    _verbose_logging = __verbose_logging
    _null_score_diff_threshold = __null_score_diff_threshold
    
    
    
    
    
    
    tf.logging.set_verbosity(tf.logging.INFO)

    bert_config = modeling.BertConfig.from_json_file(_bert_config_file)

#     validate_flags_or_throw(bert_config)

    tf.gfile.MakeDirs(_output_dir)

    tokenizer = tokenization.FullTokenizer(
      vocab_file=_vocab_file, do_lower_case=_do_lower_case)

    tpu_cluster_resolver = None
    # if FLAGS.use_tpu and FLAGS.tpu_name:
    #     tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
    #         FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)
#     print(FLAGS)
    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      master=_master,
      model_dir=_output_dir,
      save_checkpoints_steps=_save_checkpoints_steps,
      tpu_config=tf.contrib.tpu.TPUConfig(
          iterations_per_loop=_iterations_per_loop,
          num_shards=_num_tpu_cores,
          per_host_input_for_training=is_per_host))

    train_examples = None
    num_train_steps = None
    num_warmup_steps = None
    if _do_train:
        train_examples = read_squad_examples(
            input_file=_train_file, is_training=True)
        num_train_steps = int(
            len(train_examples) / _train_batch_size * _num_train_epochs)
        num_warmup_steps = int(num_train_steps * _warmup_proportion)

        # Pre-shuffle the input to avoid having to make a very large shuffle
        # buffer in in the `input_fn`.
        rng = random.Random(12345)
        rng.shuffle(train_examples)

    model_fn = model_fn_builder(
      bert_config=bert_config,
      init_checkpoint=_init_checkpoint,
      learning_rate=_learning_rate,
      num_train_steps=num_train_steps,
      num_warmup_steps=num_warmup_steps,
      use_tpu=_use_tpu,
      use_one_hot_embeddings=_use_tpu)

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = tf.contrib.tpu.TPUEstimator(
      use_tpu=_use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=_train_batch_size,
      predict_batch_size=_predict_batch_size)

    # if FLAGS.do_train:
    #     # We write to a temporary file to avoid storing very large constant tensors
    #     # in memory.
    #     train_writer = FeatureWriter(
    #         filename=os.path.join(FLAGS.output_dir, "train.tf_record"),
    #         is_training=True)
    #     convert_examples_to_features(
    #         examples=train_examples,
    #         tokenizer=tokenizer,
    #         max_seq_length=FLAGS.max_seq_length,
    #         doc_stride=FLAGS.doc_stride,
    #         max_query_length=FLAGS.max_query_length,
    #         is_training=True,
    #         output_fn=train_writer.process_feature)
    #     train_writer.close()

    #     tf.logging.info("***** Running training *****")
    #     tf.logging.info("  Num orig examples = %d", len(train_examples))
    #     tf.logging.info("  Num split examples = %d", train_writer.num_features)
    #     tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
    #     tf.logging.info("  Num steps = %d", num_train_steps)

    #     del train_examples

    #     train_input_fn = input_fn_builder(
    #         input_file=train_writer.filename,
    #         seq_length=FLAGS.max_seq_length,
    #         is_training=True,
    #         drop_remainder=True)
    #     estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)



    if _do_predict:
        eval_examples = read_squad_examples(
            input_file=_predict_file, is_training=False)
        
        #
        test_example=SquadExample(
        qas_id= _qas_id,
        question_text= _question_text,
        doc_tokens= _doc_tokens)
        eval_examples=[]
        eval_examples.append(test_example) 
        
        
        
        
        eval_writer = FeatureWriter(
            filename=os.path.join(_output_dir, "eval.tf_record"),
            is_training=False)
        eval_features = []

        def append_feature(feature):
            eval_features.append(feature)
            eval_writer.process_feature(feature)

        convert_examples_to_features(
            examples=eval_examples,
            tokenizer=tokenizer,
            max_seq_length=_max_seq_length,
            doc_stride=_doc_stride,
            max_query_length=_max_query_length,
            is_training=False,
            output_fn=append_feature)
        eval_writer.close()

        tf.logging.info("***** Running predictions *****")
        tf.logging.info("  Num orig examples = %d", len(eval_examples))
        tf.logging.info("  Num split examples = %d", len(eval_features))
        tf.logging.info("  Batch size = %d", _predict_batch_size)

        all_results = []

        predict_input_fn = input_fn_builder(
            input_file=eval_writer.filename,
            seq_length=_max_seq_length,
            is_training=False,
            drop_remainder=False)

        # If running eval on the TPU, you will need to specify the number of
        # steps.
        all_results = []
        for result in estimator.predict(
            predict_input_fn, yield_single_examples=True):
          if len(all_results) % 1000 == 0:
            tf.logging.info("Processing example: %d" % (len(all_results)))
          unique_id = int(result["unique_ids"])
          start_logits = [float(x) for x in result["start_logits"].flat]
          end_logits = [float(x) for x in result["end_logits"].flat]
          all_results.append(
              RawResult(
                  unique_id=unique_id,
                  start_logits=start_logits,
                  end_logits=end_logits))

        output_prediction_file = os.path.join(_output_dir, "predictions.json")
        output_nbest_file = os.path.join(_output_dir, "nbest_predictions.json")
        output_null_log_odds_file = os.path.join(_output_dir, "null_odds.json")

        _predict_results =write_predictions(eval_examples, eval_features, all_results,
                          _n_best_size, _max_answer_length,
                          _do_lower_case, output_prediction_file,
                          output_nbest_file, output_null_log_odds_file)
        return _predict_results




def init_inference_Engine(
     __vocab_file="model/chinese_L-12_H-768_A-12/vocab.txt", 
    __bert_config_file="model/chinese_L-12_H-768_A-12/bert_config.json", 
    __init_checkpoint="model/chinese_L-12_H-768_A-12/bert_model.ckpt", 
    __do_train=False, 
    __train_file="model/SQAD/DRCD/DRCD_training.json", 
    __do_predict=True, 
    __predict_file="model/SQAD/DRCD/my_test.json", 
    __train_batch_size=12, 
    __learning_rate=3e-5, 
    __num_train_epochs=2.0, 
    __max_seq_length=320, 
    __doc_stride=128, 
    __output_dir="model/qa_modelsize12_aptg_20200205",
    __use_tpu=False,
    __tpu_name=None,
    __tpu_zone=None,
    __version_2_with_negative=False,
    __do_lower_case =True,
    __num_tpu_cores =8, #
    __master=None,
    __predict_batch_size=8,
    __max_query_length=64,
    __iterations_per_loop=1000,
    __save_checkpoints_steps=1000,
    __n_best_size=20,
    __max_answer_length=30,
    __warmup_proportion=0.1,
    __gcp_project=None,
    __verbose_logging=False,
    __null_score_diff_threshold=0.0
                ):
    
#     global _vocab_file 
    global _vocab_file
    global _bert_config_file
    global _init_checkpoint 
    global _do_train
    global  _train_file
    global _do_predict
    global _predict_file
    global _train_batch_size
    global _learning_rate
    global _num_train_epochs 
    global _max_seq_length
    global _doc_stride
    global _output_dir
    global _use_tpu
    global _tpu_name
    global _tpu_zone
    global _version_2_with_negative
    global _do_lower_case
    global _num_tpu_cores
    global _master
    global _predict_batch_size
    global _max_query_length
    global _iterations_per_loop
    global _save_checkpoints_steps
    global _n_best_size
    global _max_answer_length
    global _warmup_proportion
    global _gcp_project
    global _verbose_logging
    global _null_score_diff_threshold
    
    _vocab_file = __vocab_file
    _bert_config_file=__bert_config_file
    _init_checkpoint = __init_checkpoint 
    _do_train = __do_train
    _train_file = __train_file 
    _do_predict = __do_predict
    _predict_file = __predict_file 
    _train_batch_size =  __train_batch_size
    _learning_rate= __learning_rate 
    _num_train_epochs = __num_train_epochs 
    _max_seq_length = __max_seq_length 
    _doc_stride = __doc_stride
    _output_dir = __output_dir
    _use_tpu = __use_tpu
#     _tpu_name = ___tpu_name
#     _tpu_zone = __tpu_zone
    _version_2_with_negative=_version_2_with_negative = __version_2_with_negative=_version_2_with_negative
    _do_lower_case = __do_lower_case
    _num_tpu_cores = __num_tpu_cores #
    _master = __master
    _predict_batch_size = __predict_batch_size
    _max_query_length = __max_query_length
    _iterations_per_loop = __iterations_per_loop
    _save_checkpoints_steps = __save_checkpoints_steps
    _n_best_size = __n_best_size
    _max_answer_length = __max_answer_length
    _warmup_proportion = __warmup_proportion
    _gcp_project = __gcp_project
    _verbose_logging = __verbose_logging
    _null_score_diff_threshold = __null_score_diff_threshold
    
    
    
    
    
    
    tf.logging.set_verbosity(tf.logging.INFO)

    bert_config = modeling.BertConfig.from_json_file(_bert_config_file)

#     validate_flags_or_throw(bert_config)

    tf.gfile.MakeDirs(_output_dir)

    tokenizer = tokenization.FullTokenizer(
      vocab_file=_vocab_file, do_lower_case=_do_lower_case)

    tpu_cluster_resolver = None
    # if FLAGS.use_tpu and FLAGS.tpu_name:
    #     tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
    #         FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)
#     print(FLAGS)
    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      master=_master,
      model_dir=_output_dir,
      save_checkpoints_steps=_save_checkpoints_steps,
      tpu_config=tf.contrib.tpu.TPUConfig(
          iterations_per_loop=_iterations_per_loop,
          num_shards=_num_tpu_cores,
          per_host_input_for_training=is_per_host))

    train_examples = None
    num_train_steps = None
    num_warmup_steps = None
    if _do_train:
        train_examples = read_squad_examples(
            input_file=_train_file, is_training=True)
        num_train_steps = int(
            len(train_examples) / _train_batch_size * _num_train_epochs)
        num_warmup_steps = int(num_train_steps * _warmup_proportion)

        # Pre-shuffle the input to avoid having to make a very large shuffle
        # buffer in in the `input_fn`.
        rng = random.Random(12345)
        rng.shuffle(train_examples)

    model_fn = model_fn_builder(
      bert_config=bert_config,
      init_checkpoint=_init_checkpoint,
      learning_rate=_learning_rate,
      num_train_steps=num_train_steps,
      num_warmup_steps=num_warmup_steps,
      use_tpu=_use_tpu,
      use_one_hot_embeddings=_use_tpu)

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = tf.contrib.tpu.TPUEstimator(
      use_tpu=_use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=_train_batch_size,
      predict_batch_size=_predict_batch_size)
    

    # estimator = tf.estimator.Estimator(
    #   use_tpu=_use_tpu,
    #   model_fn=model_fn,
    #   config=run_config,
    #   train_batch_size=_train_batch_size,
    #   predict_batch_size=_predict_batch_size,
    #   params={})

    # if FLAGS.do_train:
    #     # We write to a temporary file to avoid storing very large constant tensors
    #     # in memory.
    #     train_writer = FeatureWriter(
    #         filename=os.path.join(FLAGS.output_dir, "train.tf_record"),
    #         is_training=True)
    #     convert_examples_to_features(
    #         examples=train_examples,
    #         tokenizer=tokenizer,
    #         max_seq_length=FLAGS.max_seq_length,
    #         doc_stride=FLAGS.doc_stride,
    #         max_query_length=FLAGS.max_query_length,
    #         is_training=True,
    #         output_fn=train_writer.process_feature)
    #     train_writer.close()

    #     tf.logging.info("***** Running training *****")
    #     tf.logging.info("  Num orig examples = %d", len(train_examples))
    #     tf.logging.info("  Num split examples = %d", train_writer.num_features)
    #     tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
    #     tf.logging.info("  Num steps = %d", num_train_steps)

    #     del train_examples

    #     train_input_fn = input_fn_builder(
    #         input_file=train_writer.filename,
    #         seq_length=FLAGS.max_seq_length,
    #         is_training=True,
    #         drop_remainder=True)
    #     estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
    global _estimator
    global _tokenizer
    _estimator = estimator
    _tokenizer = tokenizer


    # estimator.export_saved_model('model/my_saved_model/123', predict_input_fn)
    # return estimator, tokenizer

def fast_do_inference(_qas_id,
        _question_text,
        _doc_tokens ):
    _do_predict =True
    global _estimator
    global _tokenizer 
    estimator = _estimator
    tokenizer = _tokenizer
    if _do_predict:
        # eval_examples = read_squad_examples(
        #     input_file=_predict_file, is_training=False)
        
        #
        test_example=SquadExample(
        qas_id= _qas_id,
        question_text= _question_text,
        doc_tokens= _doc_tokens)
        eval_examples=[]
        eval_examples.append(test_example) 
        
        
        
        
        eval_writer = FeatureWriter(
            filename=os.path.join(_output_dir, "eval.tf_record"),
            is_training=False)
        eval_features = []

        def append_feature(feature):
            eval_features.append(feature)
            eval_writer.process_feature(feature)

        convert_examples_to_features(
            examples=eval_examples,
            tokenizer=tokenizer,
            max_seq_length=_max_seq_length,
            doc_stride=_doc_stride,
            max_query_length=_max_query_length,
            is_training=False,
            output_fn=append_feature)
        eval_writer.close()

        tf.logging.info("***** Running predictions *****")
        tf.logging.info("  Num orig examples = %d", len(eval_examples))
        tf.logging.info("  Num split examples = %d", len(eval_features))
        tf.logging.info("  Batch size = %d", _predict_batch_size)

        all_results = []

        predict_input_fn = input_fn_builder(
            input_file=eval_writer.filename,
            seq_length=_max_seq_length,
            is_training=False,
            drop_remainder=False)
        


        # estimator = tf.estimator.Estimator(model_fn, 'model', params={})
        # estimator.export_saved_model('saved_model', predict_input_fn)

        # If running eval on the TPU, you will need to specify the number of
        # steps.
        
        all_results = []
        for result in estimator.predict(
            predict_input_fn, yield_single_examples=True):
          if len(all_results) % 1000 == 0:
            tf.logging.info("Processing example: %d" % (len(all_results)))
          unique_id = int(result["unique_ids"])
          start_logits = [float(x) for x in result["start_logits"].flat]
          end_logits = [float(x) for x in result["end_logits"].flat]
          all_results.append(
              RawResult(
                  unique_id=unique_id,
                  start_logits=start_logits,
                  end_logits=end_logits))

        output_prediction_file = os.path.join(_output_dir, "predictions.json")
        output_nbest_file = os.path.join(_output_dir, "nbest_predictions.json")
        output_null_log_odds_file = os.path.join(_output_dir, "null_odds.json")

        _predict_results =write_predictions(eval_examples, eval_features, all_results,
                          _n_best_size, _max_answer_length,
                          _do_lower_case, output_prediction_file,
                          output_nbest_file, output_null_log_odds_file)
        
        _estimator = estimator
        
        # estimator2 = tf.estimator.Estimator(model_fn, 'model', params={})
        # estimator.export_saved_model('saved_model', predict_input_fn)
        return _predict_results


# def save_to_model():

#     return "cc"

def main():
    paragraph="員工旅遊的目的為提倡正當休閒活動、增進同仁間情誼、培養同仁團隊精神及建立同仁與眷屬間之情感，鼓勵同仁在不影響工作進度之原則下，兼顧同仁期望與籌辦的便利性，期能平衡同仁身心健康、增進工作效率，進而凝聚同仁對公司之向心力。活動舉辦方式由職福會洽各旅行社，提供本公司員工國內、國外優惠套裝行程，由同仁自行選擇參加，費用統一由旅行社向職福會申請報支。此費用不列為個人其他所得。由職福會團康組籌劃舉辦旅遊行程，相關內容公告後員工可自由報名參加，費用統一由職福會團康組向職福會申請報支。此費用不列為個人其他所得。由員工自行組團舉辦旅遊(正職員工需達5人以上)，活動結束後統一由主辦人填寫申請單(含旅遊計劃)並檢附相關單據向職福會申請報支，另需檢附相關活動照片(背景有旅遊目的標示及活動參加人均可清晰辨識)，若有單據不符者，職福會得退回其申請。此費用不列為個人其他所得。員工個人自行規劃旅遊行程，活動結束後填寫申請單並檢附相關單據向職福會申請報支，若有單據不符者，職福會得退回其申請。此費用需列為個人其他所得。"
    question = "員工旅遊怎麼舉辦?"
    qas_id ="test"
    init_inference_Engine()

    # userText = request.args.get('msg')
    start = time.time()
    # my_result = qa_inference.do_inference(_qas_id= qas_id,_question_text = userText, _doc_tokens =paragraph )
    my_result=fast_do_inference( _qas_id= qas_id,_question_text = question, _doc_tokens =paragraph )
    end = time.time()
    print("{} prob:{}".format(my_result['text'], my_result['probability']))

    print("spend {}".format(end-start))


if __name__ == '__main__':
    import time
    main()