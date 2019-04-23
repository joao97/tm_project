import pandas as pd
import numpy as np
import json
import re
import Levenshtein as lev
import csv


#loading the characters dataset to a dictionary
with open("data_extraction/characters.json") as f:
    data = json.load(f)

#loading the characters gender dataset to a dictionary
genders = pd.read_csv('data_extraction/genders.csv')[['name','male']]

#extracting all the possible keys in the dictionary
keys = []
for character in data['characters']:
    [keys.append(key) for key in list(character.keys())]

#removing unwanted keys    
keys = list(set(keys) - set(['actorLink','actorName','actors','characterImageFull','characterImageThumb', 'characterLink', 'characterName', 'houseName', 'royal', 'nickname']))

#parsing the data to a dataframe
relationships = []
nicknames = []
for character in data['characters']:
    char1 = character['characterName']
    for key in keys:
        try:
            for char2 in character[key]:
                relationships.append([char1.lower(),char2.lower(),key.lower()])
        except:
            pass
    try:
        nicknames.append([char1.lower(), character['nickname'].lower()])
    except:
        pass

relationships = pd.DataFrame(relationships, columns = ['character1','character2','type'])
nicknames = pd.DataFrame(nicknames, columns = ['character','nickname'])

#cleaning some inconsistency
relationships.loc[relationships['type']=='killed',['character1','character2']] = relationships.loc[relationships['type']=='killed',['character2','character1']].values
relationships.loc[relationships['type']=='killed', 'type'] = 'killedby'

relationships.loc[relationships['type']=='parents',['character1','character2']] = relationships.loc[relationships['type']=='parents',['character2','character1']].values
relationships.loc[relationships['type']=='parents', 'type'] = 'parentof'

relationships.loc[relationships['type']=='serves',['character1','character2']] = relationships.loc[relationships['type']=='serves',['character2','character1']].values
relationships.loc[relationships['type']=='serves', 'type'] = 'servedby'

relationships.loc[relationships['type']=='guardianof',['character1','character2']] = relationships.loc[relationships['type']=='guardianof',['character2','character1']].values
relationships.loc[relationships['type']=='guardianof', 'type'] = 'guardedby'

relationships.loc[relationships['type']=='abducted',['character1','character2']] = relationships.loc[relationships['type']=='abducted',['character2','character1']].values
relationships.loc[relationships['type']=='abducted', 'type'] = 'abductedby'

relationships.loc[relationships['type']=='sibling',['character1','character2']] = relationships.loc[relationships['type']=='sibling',['character2','character1']].values
relationships.loc[relationships['type']=='sibling', 'type'] = 'siblings'

relationships.drop_duplicates(inplace=True)


characters_names = list(set(np.concatenate([relationships['character1'].unique(), relationships['character2'].unique()])))

relationships = relationships.append(relationships[relationships['type']=='siblings'][['character2','character1','type']].rename(columns={'character2':'character1', 'character1':'character2'}), ignore_index=True) 

#finding grandparents
new_relations = []
for name in characters_names:
    grandparents = pd.merge(relationships[(relationships['character2']==name) & (relationships['type']=='parentof')], relationships[relationships['type']=='parentof'], left_on='character1', right_on='character2', how='inner' )[['character1_y', 'character2_x']].drop_duplicates()
    grandparents['type'] = 'grandparentof'
    uncles = pd.merge(relationships[(relationships['character2']==name) & (relationships['type']=='parentof')], relationships[relationships['type']=='siblings'], left_on='character1', right_on='character2', how='inner' )[['character1_y', 'character2_x']].drop_duplicates()
    uncles['type'] = 'uncleof'
    cousins = pd.merge(uncles, relationships[relationships['type']=='parentof'], left_on='character1_y', right_on='character1', how='inner')[['character2_x','character2']].drop_duplicates()
    cousins['type'] = 'cousinof'
    for relation in grandparents.values:
        new_relations.append(relation)
    for relation in uncles.values:
        new_relations.append(relation)
    for relation in cousins.values:
        new_relations.append(relation)

relationships = relationships.append(pd.DataFrame(new_relations, columns = ['character1','character2','type']))
relationships = relationships.append(relationships[relationships['type']=='cousinof'][['character2','character1','type']].rename(columns={'character2':'character1', 'character1':'character2'}), ignore_index=True) 
relationships = relationships.append(relationships[relationships['type']=='marriedengaged'][['character2','character1','type']].rename(columns={'character2':'character1', 'character1':'character2'}), ignore_index=True) 

relationships['char1_gender'] = [genders.loc[genders['name']==name,'male'].values[0] for name in relationships['character1']]
relationships['char2_gender'] = [genders.loc[genders['name']==name,'male'].values[0] for name in relationships['character2']]

relationships['type'].unique()

#siblings
relationships.loc[(relationships['type']=='siblings') & (relationships['char1_gender']==1) ,'type'] = 'brother_of'
relationships.loc[(relationships['type']=='siblings') & (relationships['char1_gender']==0) ,'type'] = 'sister_of'

#parents
relationships.loc[(relationships['type']=='parentof') & (relationships['char1_gender']==1) ,'type'] = 'father_of'
relationships.loc[(relationships['type']=='parentof') & (relationships['char1_gender']==0) ,'type'] = 'mother_of'

#grandparents
relationships.loc[(relationships['type']=='grandparentof') & (relationships['char1_gender']==1) ,'type'] = 'grandfather_of'
relationships.loc[(relationships['type']=='grandparentof') & (relationships['char1_gender']==0) ,'type'] = 'grandmother_of'

#uncles
relationships.loc[(relationships['type']=='uncleof') & (relationships['char1_gender']==1) ,'type'] = 'uncle_of'
relationships.loc[(relationships['type']=='uncleof') & (relationships['char1_gender']==0) ,'type'] = 'aunt_of'

#married
relationships.loc[(relationships['type']=='marriedengaged') & (relationships['char1_gender']==1) ,'type'] = 'husband_of'
relationships.loc[(relationships['type']=='marriedengaged') & (relationships['char1_gender']==0) ,'type'] = 'wife_of'

#childrens
childrens = relationships[(relationships['type']=='mother_of') | (relationships['type']=='father_of') ][['character2','character1','type','char2_gender','char1_gender']]
childrens.loc[childrens['char2_gender']==1,'type'] = 'son_of'
childrens.loc[childrens['char2_gender']==0,'type'] = 'daughter_of'
childrens.columns = ['character1','character2','type','char1_gender','char2_gender']
relationships = relationships.append(childrens)

#loading the data to a csv
relationships[['character1','character2','type']].to_csv('model/relationships.csv')
nicknames.to_csv('scripts_processing/nicknames.csv')