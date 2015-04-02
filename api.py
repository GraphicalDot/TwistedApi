#!/usr/bin/env python
#-*- coding: utf-8 -*-

"""
Author:Kaali
Dated: 17 January, 2015
Day: Saturday
Description: This file has been written for the android developer, This will be used by minimum viable product implementation
            on android 

Comment: None
"""


from __future__ import absolute_import
import copy
import re
import csv
import codecs
from flask import Flask
from flask import request, jsonify
from flask.ext import restful
from flask.ext.restful import reqparse
from flask import make_response, request, current_app
from functools import update_wrapper
from flask import jsonify
import hashlib
import subprocess
import shutil
import json
import os
import StringIO
import difflib
from Text_Processing import NounPhrases, bcolors, NERs, NpClustering 

from compiler.ast import flatten
##TODO run check_if_hunpos and check_if stanford fruntions for postagging and NERs and postagging


from Text_Processing import PosTaggers, NounPhrases


import time
from datetime import timedelta
from collections import Counter
from functools import wraps
import itertools
import random
from multiprocessing import Pool
import base64
import inspect


app = Flask(__name__)                                                                                                                                       
app.config['DEBUG'] = True
api = restful.Api(app,) 



def to_unicode_or_bust(obj, encoding='utf-8'):
	if isinstance(obj, basestring):
		if not isinstance(obj, unicode):
			obj = unicode(obj, encoding)
	return obj




def pos_tagging_algorithm(algorithm_name):
        members = [member[0] for member in inspect.getmembers(PosTaggers, predicate=inspect.ismethod) if member[0] 
                                            not in ["__init__", "to_unicode_or_bust"]]

        if algorithm_name not in members:
                raise StandardError("The algorithm you are trying to use for Pos Tagging  doesnt exists yet,\
                                    please try from these algorithms {0}".format(members))
        return algorithm_name

def noun_phrases_algorithm(algorithm_name):
        members = [member[0] for member in inspect.getmembers(NounPhrases, predicate=inspect.ismethod) if member[0] 
                                            not in ["__init__", "to_unicode_or_bust"]]

        if algorithm_name not in members:
                raise StandardError("The algorithm you are trying to use for noun phrases doesnt exists yet,\
                                    please try from these algorithms {0}".format(members))
        return algorithm_name





def np_clustering_algorithm(algorithm_name):
        members = [member[0] for member in inspect.getmembers(NpClustering, predicate=inspect.ismethod) if member[0] 
                                            not in ["__init__",]]

        if algorithm_name not in members:
                raise StandardError("The algorithm you are trying to use for noun phrase clustering doesnt exists yet,\
                                    please try from these algorithms {0}".format(members))
        return algorithm_name


def ner_algorithm(algorithm_name):
        members = [member[0] for member in inspect.getmembers(NERs, predicate=inspect.ismethod) if member[0] 
                                            not in ["__init__",]]

        if algorithm_name not in members:
                raise StandardError("The algorithm you are trying to use for ner extraction doesnt exists yet,\
                                    please try from these algorithms {0}".format(members))
        return algorithm_name


def custom_string(__str):
        return __str.encode("utf-8")



##GetWordCloud
get_word_cloud_parser = reqparse.RequestParser()
get_word_cloud_parser.add_argument('noun_phrases_algorithm', type=noun_phrases_algorithm,  required=False, location="form")
get_word_cloud_parser.add_argument('pos_tagging_algorithm', type=pos_tagging_algorithm,  required=False, location="form")
get_word_cloud_parser.add_argument('np_clustering_algorithm', type=np_clustering_algorithm,  required=False, location="form")
get_word_cloud_parser.add_argument('ner_algorithm', type=ner_algorithm,  required=False, location="form")



def cors(func, allow_origin=None, allow_headers=None, max_age=None):
	if not allow_origin:
                allow_origin = "*"
                		
	if not allow_headers:
		allow_headers = "content-type, accept"
		
	if not max_age:
		max_age = 60

	@wraps(func)
	def wrapper(*args, **kwargs):
		response = func(*args, **kwargs)
		cors_headers = {
				"Access-Control-Allow-Origin": allow_origin,
				"Access-Control-Allow-Methods": func.__name__.upper(),
				"Access-Control-Allow-Headers": allow_headers,
				"Access-Control-Max-Age": max_age,
				}
		if isinstance(response, tuple):
			if len(response) == 3:
				headers = response[-1]
			else:
				headers = {}
			headers.update(cors_headers)
			return (response[0], response[1], headers)
		else:
			return response, 200, cors_headers
	return wrapper


                
class EateriesList(restful.Resource):
	@cors
	def get(self):
		
                result = [{u'eatery_id': u'4114', u'eatery_name': u'Choko La'}, {u'eatery_id': u'820', u'eatery_name': u'Market Cafe'},
                    {u'eatery_id': u'304628', u'eatery_name': u'Prabhu Chaat Bhandar'}, {u'eatery_id': u'300656', u'eatery_name': u'Smoke House Deli'},
                    {u'eatery_id': u'4624', u'eatery_name': u'Le Marche - Sugar & Spice'}, {u'eatery_id': u'310000', u'eatery_name': u'Au Bon Pain'},]
                return {"success": True,
			"error": False,
			"result": result,
			}



class GetWordCloud(restful.Resource):
	@cors
        def post(self):
                time.sleep(30)
                return {"success": True,
				"error": False,
				"result": "some result",
                    }



api.add_resource(EateriesList, '/eateries_list')
api.add_resource(GetWordCloud, '/get_word_cloud')


if __name__ == '__main__':
        app.run(port=8000, debug=True)
