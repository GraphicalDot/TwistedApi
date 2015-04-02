# TwistedApi
The api has been written in flask, and need to be written in twisted
There are only two api's for testing purposes present in this module.
The module which was used to handle incoming params in the api's calls is flask-restful 

This module should be run in virtual env.
Instructions:
		sudo pip install virtualenv
		or 
		sudo apt-get install python-virtualenv for system wide installation, advisable


		$ virtualenv twistedapi
		$ cd twistedapi
		$ source bin/activate
		$ git clone https://github.com/kaali-python/TwistedApi.git
		$ cd TwistedApi
		$ pip install -r requirements.txt
		$ ./api.py

All these commands, if run in the sequence mentioned above, will start a localhost on port 8000


1. /eateries_list : Get Request
			Args: None
			result: A list menioned in the api

2. /get_word_cloud: Post Request:
			the dafault type allowed in flask-restful is str, int, float etc, but we required to specify
			the algorithms that hs been implemented, so we need to have custom params handler like we did
			in the below mentioned.

			for example:
				the first arg is noun_phrases_algorithm, its type is noun_phrases_algorithm which 
				is a function which checks the class noun_phrases_algorithm in Text_Processing mdule
				for all the algortihms that has already been implemented.
			Args:
				'noun_phrases_algorithm', type=noun_phrases_algorithm,  required=False, location="form"                                 
~                              	'pos_tagging_algorithm', type=pos_tagging_algorithm,  required=False, location="form"                                
~                             	'np_clustering_algorithm', type=np_clustering_algorithm,  required=False, location="form"                               
~                            	'ner_algorithm, type=ner_algorithm,  required=False, location="form" 

			This api generally takes more than 30s to respond in development phase as the cloud modules havent
			been implemented yet.so these api needs to be written in twisted so that other users dont have to wait
			for the response of  this api.



To test these api's
	 payload = {"pos_tagging_algorithm": "hunpos_pos_tagger", "ner_algorithm": "stanford_ner"}
	 r = requests.post("http://localhost:8000/get_word_cloud", data=payload)
	
	will return a result {u'error': False, u'result': u'some result', u'success': True}
	after a wait of 30s









