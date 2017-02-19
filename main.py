#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 15:36:21 2017

@author: benjaminpujol
"""
import nltk
import preprocessing as PP
def main():
    
    #NLP settings
    nltk.download('stopwords')
    stpwds = set(nltk.corpus.stopwords.words("english"))
    
    #Construct child and parent graphs
    PP.improve_info
    