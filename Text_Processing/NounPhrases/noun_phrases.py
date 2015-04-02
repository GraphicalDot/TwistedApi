#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
Author: Kaali
Dated: 31 january, 2015
This file lists the sentences and the noun phrasesthat hould be extracted
and test several noun phrases extraction algorithms whether they are providing desired output

"""

import os
import sys
import inspect
import re
from functools import wraps


def need_pos_tagged(pos_tagged):  
        def tags_decorator(func):
                @wraps(func)
                def func_wrapper(self, *args, **kwargs):
                        if pos_tagged and type(self.list_of_sentences[0]) != list :
                                raise StandardError("The pos tagger you are trying run needs pos tagged list of sentences\
                                        Please try some other pos tagger which doesnt require word tokenized sentences")
                        func(self, *args, **kwargs)
                return func_wrapper 
        return tags_decorator  



class NounPhrases:
        def __init__(self, list_of_sentences, default_np_extractor=None, regexp_grammer=None):
                self.noun_phrases = list()
                self.list_of_sentences = list_of_sentences
                self.np_extractor = ("textblob_np_conll", default_np_extractor)[default_np_extractor != None]
                if not regexp_grammer:
                        self.regexp_grammer = r"CustomNounP:{<JJ|VB|FW|VBN>?<NN.*>*<NN.*>}"

                eval("self.{0}()".format(self.np_extractor)) 
               
                self.noun_phrases = {self.np_extractor: self.noun_phrases}
                
                return 

        @need_pos_tagged(True)
        def regex_np_extractor(self):
                return



        @need_pos_tagged(False)
	def textblob_np_conll(self):
                return
        
        @need_pos_tagged(False)
        def textblob_np_base(self):
                return


        @need_pos_tagged(True)
        def regex_textblob_conll_np(self):
                return

