# -*- coding: utf-8 -*-
import xml.etree.ElementTree as ET
import pickle
import os 
#from string import unicode


#The following 3 varibles are written to the output

# super_sentence contains the list of sentences. where sentences is a list of words
super_sentence = []

# super_natural contains:
#
# 1 if it is a natural event
# 0 if it is a man made event
# None if it not an event
super_natural = []

# super_trigger contains:
# the trigger string like FLOODS,LAND_SLIDES etc or None accordingly
super_trigger = []




sentence = []
natural = []
trigger = []

def listappend(text,event_tag,event_type):
    global sentence
    global super_sentence
    global super_trigger
    global trigger
    global super_natural
    global natural
    
    EOS = False
    '''
    if text!=None:
        try: 
            text=text.decode('utf8')
        except AttributeError:
            print(text)
            pass
    	#text=text.decode('utf8')
        #text=unicode(text, 'utf-8')
    '''
    if text == None and sentence == [] :
        return
    if text != None:
       
        if '।' in text:
            EOS = True
        
        sentence.append(text)
        trigger.append(event_type)
        natural.append(event_tag)
    else :
        EOS = True
    if EOS:
        #print(sentence)
        super_sentence.append(sentence)
        sentence =[]
        super_trigger.append(trigger)
        trigger =[]
        super_natural.append(natural)
        natural =[]


def parsedoc(document) :
    c=0
    for para in document :
        listappend(None,None,None)
        for tag in para :
            if tag.tag == 'w' :
                listappend(tag.text,None,None)
            else :
                
                if tag.tag == 'natural_event' :
                    event_tag = 1
                    c=c+1
                    event_type = tag.attrib['type']
                elif tag.tag == 'man_made_event' :
                    event_tag = 0
                    c=c+1
                    event_type = tag.attrib['type']
                else :
                    event_tag =None
                    event_type = None
                
                for word in tag.iter('w') :
                    listappend(word.text,event_tag,event_type)
    print(c)

#CODE TO PROCESS ONE FILE

# tree = ET.parse('/media/bhargav/DATA/NLP Term Project/event_extraction_nlp_project/hindi_anntated/Train/indore-patna-express-train-derails-in-kanpur-help-line-number-20161120000419.txt.xml')
# print(tree.getroot())
# parsedoc(tree.getroot())
#      
#        
        
#CODE TO PROCESS ALL XML FILES IN THE DIRECTORY

#path = '/home/sarath/Desktop/ARNAB'
path = '/media/shubham/1A2A3CBF2A3C99A9/Academics/Sem_5/Speech and Natural Language Processing/CS60057/event_extraction_nlp_project/hindi_anntated/Train'
c=0

for filename in os.listdir(path):
    c=c+1
    super_sentence = []
    super_natural = []
    super_trigger = []
    sentence = []
    natural = []
    trigger = []
    if not filename.endswith('.xml'): continue
    fullname = os.path.join(path, filename)
    print(fullname)
    tree = ET.parse(fullname)
    parsedoc(tree.getroot())
    file = open('result.pkl','wb')
    pickle.dump(super_sentence,file)
    pickle.dump(super_natural,file)
    pickle.dump(super_trigger,file)
    file.close()
              
print(c)

#Writes the varibles to File
#
file = open('result.pkl','wb')
print(super_sentence)
pickle.dump(super_sentence,file)
print(super_natural)
pickle.dump(super_natural,file)
print(super_trigger)
pickle.dump(super_trigger,file)
file.close()
                    
