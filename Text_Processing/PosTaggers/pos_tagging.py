#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Kaali
Dated: 3 february, 2015
For the pos tagging of the list of sentences
"""
import os
import sys
import subprocess
import warnings
from itertools import chain
from functools import wraps
dir_name = os.path.dirname(os.path.abspath(__file__))                           

def need_word_tokenization(word_tokenize):
        def tags_decorator(func):
                @wraps(func)
                def func_wrapper(self, *args, **kwargs):
                        if word_tokenize and type(self.list_of_sentences[0]) != list : 
                                raise StandardError("This Pos tagger needs a Word tokenized list of sentences, Please try some other pos tagger\
                                        which doesnt require word tokenized sentences")
                        func(self, *args, **kwargs)
                return func_wrapper
        return tags_decorator






class PosTaggers:
        #os.environ["JAVA_HOME"] = "{0}/ForStanford/jdk1.8.0_31/jre/bin/".format(stanford_file_path)
        #stanford_jar_file = "{0}/ForStanford/stanford-postagger.jar".format(stanford_file_path) 
        #stanford_tagger = "{0}/ForStanford/models/english-bidirectional-distsim.tagger".format(stanford_file_path) 
        def __init__(self, list_of_sentences, default_pos_tagger=None, list_of_sentences_type=None):
                return 


        @need_word_tokenization(True)
        def hunpos_pos_tagger(self):
                return

        @need_word_tokenization(True)
        def stan_pos_tagger(self):
                return

        @need_word_tokenization(False)
        def textblob_pos_tagger(self):
                return 

        @need_word_tokenization(True)
        def nltk_pos_tagger(self):
                return


