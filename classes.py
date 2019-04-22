# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 23:37:38 2019

@author: Guilherme
"""

class Character:
    
    def __init__(self, name, alias_list):
        self.name = name
        self.relations = []
        self.alias = alias_list
        self.context = None
        
    def add_alias(self,alias_list):
        self.alias = self.alias + alias_list
    
    def set_relation(self,target,type_):
        self.relations.append(Relation(self,type_,target))
    
    def set_context(self,context):
        self.context = context
        
class Relation:
    
    def __init__(self, char1, type_, char2):
        self.char1 = char1
        self.type_ = type_
        self.char2 = char2
        
    def print_(self):
        string = self.char2.name + ' is ' + self.char1.name +"'s " + self.type_
        print(string)
        return string

    def tuple_(self):
        return (self.char1.name,self.char2.name,self.type_)