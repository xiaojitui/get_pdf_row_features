#!/usr/bin/env python
# coding: utf-8

# ## Import 

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import os
import re

import pandas as pd

import pdfplumber
import camelot
import PyPDF2

# import cv2

from scipy.stats import mode

import tempfile

import copy


# In[ ]:





# ## All functions

# #### get text, words, and lines

# In[2]:


# use pdfplumber + pypdf2 to get text

def get_text_word(pdffile, page_select = None):
    alltext = [] # format of x0, x1, top, bottom, text
    # can be 'SCHEDULE OF INVESTMENT ASSETS' or "SCHEDULE OF ASSETS"
    allwords = []
    
    alllines = []
    allrects = []
    
    with pdfplumber.open(pdffile) as pdf:
        
        pages = pdf.pages
        page_n = len(pages)
        
        if page_select is None: 
            for i in range(page_n):

                text = pages[i].extract_text()
                words = pages[i].extract_words()
                lines = pages[i].lines
                rects = pages[i].rects
                alltext.append(text)
                allwords.append(words)
                alllines.append(lines)
                allrects.append(rects)
        else:
            for i in page_select:
                text = pages[i].extract_text()
                words = pages[i].extract_words()
                lines = pages[i].lines
                rects = pages[i].rects
                alltext.append(text)
                allwords.append(words)
                alllines.append(lines)
                allrects.append(rects)

    #text = page.extract_text()
    #text.split('\n')
    
    return alltext, allwords, alllines, allrects, pages


# this is another way to get text, sometimes they have different (but overlapping) results
def get_text_word_pypdf2(pdffile, page_select = None):
    pdf_in = open(pdffile, 'rb')
    pdf_reader = PyPDF2.PdfFileReader(pdf_in)
    
    alltext = [] 
    totalpage = pdf_reader.getNumPages()
    
    if page_select == None: 
        for page_n in range(totalpage):
            page = pdf_reader.getPage(page_n)
            text = page.extractText()
            alltext.append(text)
        pdf_in.close()
    else:
        for page_n in page_select:
            page = pdf_reader.getPage(page_n)
            text = page.extractText()
            alltext.append(text)
        pdf_in.close()
        
    return alltext


# #### search keywords on pages 

# In[3]:


# search for keywords, and return page number

#pattern1 = r'.*schedule\s*of\s*investments*.*' # any character + schedule of investment(s) + any character
#pattern2 = r'.*schedule\s*of\s*assets*.*' # any character + schedule of asset(s) + any character

def trackpage(alltext, pages, patterns):
    page_tracker = []
    
    for i in range(len(pages)):
        if alltext[i] is not None:
            page_text = alltext[i].split('\n')   
            for line in page_text:
                for pattern in patterns: 
                    if re.match(pattern, str.lower(line)): #or re.match(pattern2, str.lower(line))
                        page_tracker.append(i)
                        break 
    return list(set(page_tracker))

# try to use header keywords to refine page_tracker
#header_key_words = ['description', 'shares', 'shares/par', 'shares/par Value', 'par value', 'cost', 'value', 'identity', 
                    #'issue', 'identity of issue', 'borrower', 'lessor', 'maturity', 'date', 'maturity date', 
                    #'rate of interest', 'rate', 'interest']

header_key_words = ['description', 'securitydescription', 'security description', 'shares', 'shares/par', 'shares/par value', 
                    'shares/parvalue', 'par value', 'parvalue', 'cost', 'identity', 'issue', 'identity of issue', 
                    'identityofissue', 'borrower', 'lessor', 'maturity', 'date', 'maturity date', 'maturitydate', 'rate', 
                    'rate of interest', 'of interest', 'ofinterest', 'rateofinterest', 'historical cost', 'historicalcost', 
                    'current value', 'currentvalue']

def refinepages(alltext, page_tracker, keywords, thresh = 3):
    
    page_tracker_clean = []
    
    for page in page_tracker:
        wordcounter = 0
        page_text = alltext[page].split('\n')   
        for line in page_text: 
            if wordcounter < thresh:
                for word in keywords:
                    if word in str.lower(line):
                        wordcounter += 1
                        if wordcounter >= thresh:
                            page_tracker_clean.append(page)
                            break
            else:
                break
    return page_tracker_clean
# #### find rotated pages and rotate them back

# In[4]:


# check rotation
# rotation indicator: in a few rows,  many single chars appear, 
# use char_counter method

# 0 = normal, 1 = rotated


# use to count short chars on a single page
# use in "checkrotation" function 
def findshortchars(text, threshold = 3):
    char_counter = 0
    if text is not None:
        for ele in text.split('\n'): 
            if 0<len(ele.strip()) <= threshold:
                char_counter +=1
    return char_counter


def checkrotation(alltext, pages, threshold = 3, mincount = 9, ratio = 5):
    
    rotation = {}
    for page in pages:
        rot_cur_page = 0
        if alltext[page] is not None:
            char_counter = findshortchars(alltext[page], threshold = threshold)
            row_len = len(alltext[page].split('\n'))
            if char_counter >= mincount and row_len/char_counter < ratio:
                rot_cur_page = 1
        rotation[page] = rot_cur_page

        
    # can make the assumption that, if more than half is rotated, all the tables are actually rotated?. 
    if np.sum([k ==1 for k in rotation.values()]) > 0.5*(len(rotation.values())):
        for page in rotation.keys():
            rotation[page] = 1 

    return rotation


# In[5]:


# post-processing functions: if page is rotated, correct it

def correctrotation(pdffile, allwords, alllines, allrects, rotation):
    
    pdf_writer = PyPDF2.PdfFileWriter()
    pdf_reader = PyPDF2.PdfFileReader(pdffile)
    
    pages = list(rotation.keys())
    
    # write all rotated pages into a temporary file 
    
    rotated = []
    for i in range(len(pages)):
        page_n = pages[i]
        if rotation[page_n] == 1: 
            rotated.append(i)
            page = pdf_reader.getPage(page_n).rotateClockwise(90)
            pdf_writer.addPage(page)
    
    if rotated == []:
        return allwords, alllines, allrects
    
    with tempfile.NamedTemporaryFile(delete=False) as fp:
        pdf_writer.write(fp)
        fp.seek(0)
        pdf = pdfplumber.open(fp.name)
        
        for i in rotated:
            page_n = pages[i]
            page = pdf.pages[i]
            words = page.extract_words()
            allwords[page_n] = words
            
            lines = page.lines
            alllines[page_n] = lines
            
            rects = page.rects
            allrects[page_n] = rects
        
    return allwords, alllines, allrects


# #### group close words and assign row numbers

# In[6]:


# post-processing functions: find rows and group words
# row_tol: used to consider font variance in a row
# col_tol: if two words are within col_tol, connect them. 

def groupwords(allwords, row_tol = 1.5, col_tol = 6):
    
    allwords_clean = []
    
    for words in allwords:
        
        if words == []:
            allwords_clean.append([])
            continue
        
        # find rows first
        words[0]['row'] = 0
        for i in range(1, len(words)):
            if abs(words[i]['top'] - words[i-1]['top']) <= row_tol or                 abs(words[i]['bottom'] - words[i-1]['bottom']) <= row_tol:
                words[i]['row'] = words[i-1]['row']
            else:
                words[i]['row'] = words[i-1]['row'] + 1
                
        # then group words
        words_clean = []
        i = 0
        while i < len(words):

            x0 = words[i]['x0']  # need some tolerence?
            x1 = words[i]['x1']
            text = words[i]['text']
            top = words[i]['top']
            bottom = words[i]['bottom']
            row = words[i]['row']

            j = i + 1
            
            while j < len(words) and words[j]['row'] == words[j-1]['row'] and                 abs(words[j]['x0'] - words[j-1]['x1']) <= col_tol:
                x0 = min(x0, words[j-1]['x0'])
                x1 = max(x1, words[j]['x1'])
                text = text + ' ' + words[j]['text']
                j = j + 1
            
            # do some cleaning for $ sign
            #if words[j-1]['text'].strip() == '$':
                #x1 = min(x1, words[j-2]['x1'])
                #text = text.strip().strip('$').strip()
            
            words_clean.append({'x0': int(x0), 'x1': int(x1), 'top': int(top), 'bottom': int(bottom), 
                                 'text': text, 'row': int(row)})
            i = j
          
        allwords_clean.append(words_clean)
    
    return allwords_clean


# #### a function to group close lines

# In[7]:


# a function to group close lines

def grouplines(cols, min_sep, method = 'min'):
    grouped = []
    i = 0
    while i < len(cols):
        checked = [i]
        cur_group = [cols[i]]
        j = i+1
        while j < len(cols):
            if cols[j] - cols[i] <= min_sep:
                cur_group.append(cols[j])
                checked.append(j)
                i = j
                j = j+1
            else:
                j +=1
        grouped.append(cur_group)
        i = checked[-1] + 1

    cols_clean = []
    for i in range(len(grouped)):
        #col = np.mean(grouped[i])
        #col = np.max(grouped[i])
        if method == 'min':
            col = np.min(grouped[i])
        if method == 'max':
            col = np.max(grouped[i])
        if method == 'median':
            col = np.median(grouped[i])
        if method == 'near_max':
            if len(grouped[i]) <= 1:
                col = grouped[i][0]
            else:
                col = sorted(grouped[i])[-2]
        cols_clean.append(int(col))
        
    return cols_clean


# #### find table areas on one page

# In[8]:


# find all potential row seperators on one single page

def find_table_edge(allwords_df, page, tol = 5, min_sep = 3, method = 'min'): 
    
    use_df = allwords_df.copy()
    cur_page_df = pd.DataFrame(use_df[page])
    
    # get coordinate information 
    page_positions = cur_page_df[['row', 'top', 'bottom']]
    page_positions = page_positions.drop_duplicates().sort_values(by = 'row')
    page_positions.index = np.arange(len(page_positions))
    ############################
    
    # find overlapping rows
    i = 0
    repeat_row_id = []
    while i < len(page_positions) - 1:
        if page_positions.loc[i, 'row'] != page_positions.loc[i+1, 'row']:
            i += 1
            continue
        else: 
            cur_rw_id = [i]
            while page_positions.loc[i, 'row'] == page_positions.loc[i+1, 'row']:
                i += 1
            cur_rw_id.append(i)
            repeat_row_id.append(cur_rw_id)

    ###########################
    # merge overlapping rows
    
    new_rows = []
    for row_group in repeat_row_id:
        cur_row = page_positions.loc[row_group[0], 'row']
        cur_top = []
        cur_bottom = []
        for row in row_group:
            cur_top.append(page_positions.loc[row, 'top'])
            cur_bottom.append(page_positions.loc[row, 'bottom'])
        new_rows.append({'row': cur_row, 'top': max(cur_top), 'bottom': min(cur_bottom)})
    new_rows = pd.DataFrame(new_rows, columns = ['row', 'top', 'bottom'])    

    ##########################
    # drop and merge
    
    allid = []
    for k in repeat_row_id:
        allid += k

    page_positions= page_positions.drop(allid, axis = 0)
    page_positions = pd.concat([page_positions, new_rows], axis = 0)
    page_positions = page_positions.sort_values(by = 'row')
    page_positions.index = np.arange(len(page_positions))

    ##########################
   
    page_positions['linechange'] = 0
    for i in range(1, len(page_positions)):
        page_positions.loc[i, 'linechange'] = page_positions.loc[i, 'top'] - page_positions.loc[i-1, 'bottom']

    breaklines = []
    # do not count in the last row
    for i in range(len(page_positions) - 1):
        if page_positions.loc[i, 'linechange'] >= tol:
            breaklines.append(page_positions.loc[i, 'row'])
   
    ##########################
    # clean row seperators
    breaklines_clean = grouplines(breaklines, min_sep, method)
    
    ##########################
    # find possible table boundaries
    table_edges = []
    if breaklines_clean[0] >= 3:
        table_edges.append([0, breaklines_clean[0]])
    for i in range(len(breaklines_clean) - 1):
        table_edges.append([breaklines_clean[i], breaklines_clean[i+1]])
    if breaklines_clean[-1] <= page_positions['row'].values[-1] - 3:
        table_edges.append([breaklines_clean[-1], page_positions['row'].values[-1]])
            
    return table_edges, breaklines_clean, page_positions


# #### get words and lines information inside each table_areas

# In[9]:


def get_section_words(allwords, alllines, allrects, page, table_edges, top_tol = 3, bottom_tol = 0):
    
    section_words = []
    section_lines = []
    section_rects = []
    
    for table_edge in table_edges:
        top_row = table_edge[0] - top_tol
        bottom_row = table_edge[1] + bottom_tol
        
        cur_words = []
        cur_lines = []
        cur_rects = []
        
        for word in allwords[page]:
            if top_row <= int(word['row']) < bottom_row:
                cur_words.append(word)
            if int(word['row']) >= bottom_row:
                break
                
        if alllines[page] is not None: 
            for ele in alllines[page]:
                if ele['top'] >= top_row and ele['bottom'] < bottom_row:
                    cur_lines.append(ele)
                if ele['bottom'] >= bottom_row:
                    break
        
        if allrects[page] is not None: 
            for ele in allrects[page]:
                if ele['top'] >= top_row and ele['bottom'] < bottom_row:
                    cur_rects.append(ele)
                if ele['bottom'] >= bottom_row:
                    break
                    
        section_words.append(cur_words)
        section_lines.append(cur_lines)
        section_rects.append(cur_rects)
        
    return section_words, section_lines, section_rects


# #### find table columns for one area on one page 

# In[10]:


## function to find edge candidates

# col_max = 10: pick top 10 candidates
# col_sep = 10: if two col-lines are within 30, merge them as 1 line

def find_cols(allwords, pages, page, col_max=10, col_sep=30):
    
 
    page_w = int(pages[page].width)
    page_h = int(pages[page].height)
    
    left_x = []
    right_x = []
    mid_x = []
    for word in allwords:
        left_x.append(int(word['x0']))
        right_x.append(int(word['x1']))
        mid_x.append( (int(word['x1']) + int(word['x0']))//2  )
        #top_y.append(int(word['bottom']))
    
    a, b = np.histogram(left_x, bins=np.arange(0, page_w, 5)) # col
    #c, d = np.histogram(top_y, bins=np.arange(0, top_y[-1]+10, 5)) # rows
    c, d = np.histogram(right_x, bins=np.arange(0, page_w, 5)) # right_col 
    e, f = np.histogram(mid_x, bins=np.arange(0, page_w, 5)) # mid_col 
    
    table_indicator = 0
    count_val_a = []
    count_a = pd.value_counts(a)
    
    count_val_c = []
    count_c = pd.value_counts(c)
    
    count_val_e = []
    count_e = pd.value_counts(e)
    
    for i, j in count_a.items():
        if 2<= i <= 10:
            count_val_a.append(j)
    
    for i, j in count_c.items():
        if 2<= i <= 10:
            count_val_c.append(j)
            
    for i, j in count_e.items():
        if 2<= i <= 10:
            count_val_e.append(j)
                
    #if count_val !=[] and np.all([2<=k<=10 for k in count_val]):
    if count_val_a !=[] and np.sum([2<=k<=10 for k in count_val_a]) > 0.5*(len(count_val_a)):
        table_indicator = 1
        
    if count_val_c !=[] and np.sum([2<=k<=10 for k in count_val_c]) >= 0.5*(len(count_val_c)):
        table_indicator = 1
        
    if count_val_e !=[] and np.sum([2<=k<=10 for k in count_val_e]) >= 0.5*(len(count_val_e)):
        table_indicator = 1
    
    if table_indicator == 1:
        pick_col_indx = np.argsort(a)[::-1][:col_max]
        cols = sorted(b[pick_col_indx])    
    
        # clean cols, group them, then find mean/max
        cols_clean = grouplines(cols, col_sep)
    else:
        cols_clean = []
   

    return cols_clean, table_indicator


# In[ ]:





# #### refine columns for one area on one page 

# In[11]:


def refine_cols(alllines, allrects, col_use, pages, page, thresh = 2, edge_l = 0, col_tol = 30):
    
    col_use_refine = []
    
    x = {}
    if alllines is not None: 
        for ele in alllines:
            if ele['x0'] not in x:
                x[ele['x0']] = 1
            else:
                x[ele['x0']] += 1
            if ele['x1'] not in x:
                x[ele['x1']] = 1
            else:
                x[ele['x1']] += 1
        
    if allrects is not None:           
        for ele in allrects:
            if ele['x0'] not in x:
                x[ele['x0']] = 1
            else:
                x[ele['x0']] += 1
            if ele['x1'] not in x:
                x[ele['x1']] = 1
            else:
                x[ele['x1']] += 1   

    cols = []
    if x != {}:
        for ele in x.items():
            if ele[1] >= thresh:
                cols.append(int(ele[0]))
        cols.sort()
       
    
####
    if cols != []:
        col_use_1 = [edge_l] + col_use + [int(pages[page].width)]
            
        # clean cols first 
        cols_clean = []
        for col_add in cols:
            can_add = 1
            for ele in col_use_1:
                if abs(col_add - ele) <= 15:
                    can_add = 0
                    break
            if can_add and col_add not in cols_clean:
                cols_clean.append(col_add)
            
        col_need_add = []
        for i in range(len(col_use_1)-1):
            col_left = []
            for col_add in cols_clean:
                if col_use_1[i] <= col_add <= col_use_1[i+1]:
                    col_left.append(col_add)
                if col_add > col_use_1[i+1]:
                    break
            if col_left != []:
                if len(col_left) == 1:
                    if (col_left[0] - col_use_1[i]) > col_tol or (col_use_1[i+1] - col_left[0]) > col_tol:
                        col_need_add.append(col_left[0])
                if len(col_left) > 1:
                    if (col_left[0] - col_use_1[i]) > col_tol and (col_use_1[i+1] - col_left[-1]) > col_tol:
                        col_need_add.append(col_left[0])
                    if (col_left[0] - col_use_1[i]) <= col_tol and (col_left[-1] - col_use_1[i]) > col_tol:
                        col_need_add.append(col_left[-1])
                
        cur_col_use = col_use + col_need_add
        cur_col_use.sort()
           
    else: 
        cur_col_use = col_use
            
        
        
    col_use_refine.append(cur_col_use)
    
    
    for k in range(len(col_use_refine)):  
        if col_use_refine[k][0] < col_use[0]:
            col_use_refine[k] = col_use_refine[k][1:]

   
    #print(col_use_refine)
    ## now find the common pattern
    
    new_col_vals = []
    for i in range(len(col_use)-1):
        cur_add_vals = []
        for cols in col_use_refine:
            for col in cols[1:]:
                if col_use[i] < col < col_use[i+1]:
                    cur_add_vals.append(col)
                if col > col_use[i+1]:
                    break
        cur_add_vals = np.unique(cur_add_vals)
        if len(cur_add_vals) > 1:
            new_col_vals.append(sorted(cur_add_vals)[1]) ## pick the second small one
            #new_col_vals.append(min(cur_add_vals)+10)
        if len(cur_add_vals) == 1:
            new_col_vals.append(cur_add_vals[0])
   
    #print(new_col_vals)
    ## deal with the last element
    last_add = []
    for cols in col_use_refine:
        for col in cols[1:]:
            if col > col_use[-1]:
                last_add.append(col)
    
    last_add = np.unique(last_add)
    if len(last_add) > 1:
        new_col_vals.append(sorted(last_add)[1]) ## pick the second small one
    if len(last_add) == 1:
        new_col_vals.append(last_add[0])
    
        
    final_cols = col_use + new_col_vals
    final_cols.sort()
    #print(final_cols)
   
    return final_cols, col_use_refine


# #### clean the column seperators

# In[12]:


def refine_cols_clean(final_cols, col_use, col_tol = 40):
    final_col_use_clean = col_use.copy()
    for ele in final_cols[1:]:
        for i in range(len(col_use)-1):
            if col_use[i] < ele < col_use[i+1]:
                if (ele - col_use[i]) > col_tol and (col_use[i+1] - ele) > col_tol:
                    final_col_use_clean.append(ele)       
            if col_use[i] > ele:
                break
        if ele - col_use[-1] > col_tol:
            final_col_use_clean.append(ele) 
    #final_col_use_clean = list(set(final_col_use_clean))
    final_col_use_clean.sort()
    return final_col_use_clean


# #### track keywords row numbers

# In[1]:


def trackwords(allwords_df, pages, patterns):
    row_tracker = {}
    checked = 0
    for page in pages: 
        for word in allwords_df[page]:
            if checked == 0:
                for pattern in patterns: 
                    if re.match(pattern, str.lower(word['text'])): #or re.match(pattern2, str.lower(line))
                        row_tracker[page] = word['row']
                        checked = 1
                        break 
            else:
                break
        #if page not in row_tracker:
            #row_tracker[page] = 0
    return row_tracker


# In[4]:


def trackname(allwords_df, page, start_pattern, end_pattern):
    start_tracker = {}
    end_tracker = {}
    start_checked = 0
    end_checked = 0
    for word in allwords_df[page]:
        if start_checked == 0 and end_checked == 0:
            for pattern in start_pattern: 
                if re.match(pattern, str.lower(word['text'])): #or re.match(pattern2, str.lower(line))
                    start_tracker[page] = word['row']
                    start_checked = 1
                    break 
        elif start_checked == 1 and end_checked == 0:
            for pattern in end_pattern: 
                if re.match(pattern, str.lower(word['text'])): #or re.match(pattern2, str.lower(line))
                    end_tracker[page] = word['row']
                    end_checked = 1
                    break
        else: 
            break
        #if page not in row_tracker:
            #row_tracker[page] = 0
    return start_tracker, end_tracker


# #### show projection

# In[14]:


### visualize col and row projection: optional
def showprojection(allwords, pages, page):
    
    page_w = int(pages[page].width)
    page_h = int(pages[page].height)
    
    left_x = []
    top_y = []
    for word in allwords:
        left_x.append(int(word['x0']))
        top_y.append(int(word['bottom']))
    
    fig, ax = plt.subplots(1, 2)
    # ax0 is used to find cols, ax1 is used to find tabel edges
    
    c, d, _ = ax[1].hist(top_y, bins = range(0, top_y[-1]+10, 5), orientation = 'horizontal') #rows
    ax[1].invert_yaxis()
    a, b, _ = ax[0].hist(left_x, bins = range(0, page_w, 5)) #cols

    
    ax[0].set_xlabel('column projections')
    ax[1].set_xlabel('row projections')
    
    return a, b


# In[ ]:





# #### generate dataframe on one area in one page 

# In[15]:


## post-processing: get dataframe 

def preparedf(allwords, pages, page, col_seperators, table_edge):
    
    alltables = []
    
    #print('Processing table in this area:  ', table_edge, '...', end = ' ')
        
    cur_table_edge = [0, table_edge[0], int(pages[page].width), table_edge[1]]
    col_use = [0] + col_seperators + [cur_table_edge[2]]
        
        
    # find word in this row and col 
    #picked = [word for word in allwords]
    
    picked = []
    for word in allwords:
        for i in range(len(col_use)-1):
            if col_use[i] + 2  <= 0.5*(word['x0'] + word['x1']) <= col_use[i+1] + 2:
                word['col'] = i              
                            
        picked.append(word)

        
    # col correction:
    for i in range(1, len(picked)):
            
        if picked[i]['row'] == picked[i-1]['row'] and picked[i]['col'] <= picked[i-1]['col']:
    
            if picked[i]['col'] == 0 or picked[i]['col'] == len(col_seperators):
                picked[i]['text'] = picked[i-1]['text'] + ' ' + picked[i]['text']
                
            if 2<=i<= len(picked)-2 and picked[i-1]['col'] == picked[i-2]['col'] + 1                 and picked[i]['col']+1 == picked[i+1]['col']:
                picked[i]['text'] = picked[i-1]['text'] + ' ' + picked[i]['text']
                    
            if 2<=i<= len(picked)-2 and picked[i-1]['col'] == picked[i-2]['col'] + 1                 and picked[i]['col']+1 < picked[i+1]['col']:
                picked[i]['col'] += 1
                 
            # add this: 
            if 2<=i<= len(picked)-2 and picked[i-1]['col'] == picked[i-2]['col'] + 1                 and picked[i]['col']+1 > picked[i+1]['col']:
                picked[i]['text'] = picked[i-1]['text'] + ' ' + picked[i]['text']
            # add end
                
            if 2<=i<= len(picked)-2 and picked[i-1]['col'] > picked[i-2]['col'] + 1                 and picked[i]['col']+1 == picked[i+1]['col']:
                picked[i-1]['col'] -= 1
                    
            if 2<=i<= len(picked)-2 and picked[i-1]['col'] > picked[i-2]['col'] + 1                 and picked[i]['col']+1 < picked[i+1]['col']:
                picked[i]['col'] += 1
                    
            # add this: 
            if 2<=i<= len(picked)-2 and picked[i-1]['col'] < picked[i-2]['col'] + 1                 and picked[i]['col']+1 == picked[i+1]['col']:
                picked[i]['text'] = picked[i-1]['text'] + ' ' + picked[i]['text']
            # add end
             
    # find row and col numbers
    delta = picked[0]['row']

    for ele in picked:
        ele['row'] = ele['row']- delta

    maxrow = 0
    maxcol = 0
    for ele in picked:
        maxrow = max(maxrow, ele['row'])
        maxcol = max(maxcol, ele['col'])
            
        
    # fill the dataframe
    table_df = pd.DataFrame('', index=np.arange(maxrow+1), columns=np.arange(maxcol+1))
    for ele in picked: 
        i = ele['row']
        j = ele['col']
        table_df.iloc[i, j] = ele['text']
            
        
        
    # merge cols
    for i in range(table_df.shape[0]):
        for j in range(1, table_df.shape[1]-1):
            cur_ele = []
            next_ele = []
            for ele in picked:
                if ele['row'] == i and ele['col'] == j:
                    cur_ele = ele
                if ele['row'] == i and ele['col'] == j+1:
                    next_ele = ele
                if cur_ele !=[] and next_ele !=[]:
                    break
                        
            if cur_ele !=[] and next_ele !=[]:
                cur_left, cur_right = cur_ele['x0'], cur_ele['x1']
                next_left, next_right = next_ele['x0'], next_ele['x1']
                    
                if cur_right > next_left:
                    table_df.iloc[i, j] = table_df.iloc[i, j] + ' ' + table_df.iloc[i, j+1]
                    table_df.iloc[i, j+1] = ''
        
        
    # find each's edges and median line, for fine tuning 
    col_data = []
    for i in range(table_df.shape[1]):
        col_p_left = []
        col_p_right = []
        col_p_median = []
        for ele in picked:
            if ele['col'] == i:
                col_p_left.append(ele['x0'])
                col_p_right.append(ele['x1'])
                col_p_median.append((ele['x0']+ele['x1'])/2)
        if col_p_left != [] and col_p_right !=[]:
            ##### line 11122
            col_p = [np.median(col_p_left), np.median(col_p_right), np.median(col_p_median)]
            ##### col_p = [np.min(col_p_left), np.median(col_p_right), np.median(col_p_median)]
            #col_m = np.median(col_p_median)
        else: 
            col_p = []
            #col_m = []
        col_data.append(col_p)
            
    #print(col_data[0])
        
        
    # merge close cols and drop empty cols
    first_drop = []
    
    for i in range(table_df.shape[1]-1):
        if col_data[i] != [] and col_data[i+1] != []:
            if col_data[i+1][0] < col_data[i][2] < col_data[i+1][1]:
            #if col_data[i+1][0] < col_data[i][0] and col_data[i+1][1] > col_data[i][1]  or \
                    #col_data[i+1][0] > col_data[i][0] and col_data[i+1][1] < col_data[i][1]:
                first_drop.append(i)
                table_df.iloc[:, i+1] = table_df.iloc[:, i] + ' ' + table_df.iloc[:, i+1]
                
    #print(first_drop)
    table_df.drop(first_drop, axis = 1, inplace = True)    
    table_df.columns = range(table_df.shape[1])    
        
        
    # further clean the df
    second_drop = []
        
    for i in range(table_df.shape[1]):
        #table_df[i] = [' '.join(table_df[i][k].split('\n')) for k in range(table_df.shape[0])]
        if table_df[i].isnull().all() or np.all( table_df[i] == ''):
            second_drop.append(i)
    table_df.drop(second_drop, axis = 1, inplace = True)
    table_df.columns = range(table_df.shape[1])
        
       
    
    # clean rows
    row_drop = []
    for i in range(table_df.shape[0]):
        if table_df.iloc[i, :].isnull().all() or np.all(table_df.iloc[i, :] == ''):
            row_drop.append(i)
        
    table_df.drop(row_drop, axis = 0, inplace = True)
    table_df.index = range(table_df.shape[0])
    
    # save the results
    alltables.append(table_df)
        
    #print('Done')
        
    return alltables


# #### post-processing: remove non-table-like for one section on a single pages

# In[16]:


# post-processing: remove non-table-like pagse and convert number-like-string to int 

def finalclean(curtable, col_use, tol = 1):
    
    is_table = 0 
    
    alltables_clean = []
    page_tracker_clean = []
    
    if curtable[0].shape[1] > 1 and curtable[0].shape[1] >= len(col_use) - tol:
        is_table = 1
 
    if is_table:
        return curtable[0]
    else:
        return None
        


# In[ ]:





# #### post-processing: column correction for one single table

# In[17]:


def colcorrection(alltable, header_tracker = [0], ratio = 2):
    
    alltables_clean_1 = []
    no_header_col = []
    
    table = alltable[0].copy()
    for j in range(table.shape[1]):
        no_header = True
        # more than 3/4 is empty
    
        if np.sum(table[j] != '') < len(table)/ratio:
            for h_row in header_tracker:
                if pd.notnull(table.iloc[h_row, j]) and table.iloc[h_row, j] != '' and table.iloc[h_row, j] != ' ':
                    no_header = False
                if no_header:
                    no_header_col.append(j)
        
    not_remove_col = []
    if no_header_col != []:
        for k in no_header_col:
            # table[k-1] = table[k-1] + ' ' + table[k]
            for m in range(table.shape[0]):
                if type(table.iloc[m, k-1]) == type(table.iloc[m, k]):
                    try:
                        new_col_val = table.iloc[m, k-1] + ' ' + table.iloc[m, k]
                    except:
                        #print('error when trying to combine columns')
                        #print('table id:', i, '\trow number:', m, '\tcol id:', k, '\n')
                        not_remove_col.append(k)
                        break #continue
                    else:
                        table.iloc[m, k-1] = new_col_val
                else:
                    if type(table.iloc[m, k-1]) == str:
                        if table.iloc[m, k-1] == '' or table.iloc[m, k-1] == ' ':
                            table.iloc[m, k-1] = table.iloc[m, k]
                        else: 
                            table.iloc[m, k-1] = table.iloc[m, k-1] + ' ' + str(table.iloc[m, k])
                    elif type(table.iloc[m, k]) == str:
                        if table.iloc[m, k] == '' or table.iloc[m, k] == ' ':
                            table.iloc[m, k-1] = table.iloc[m, k-1]
                        else:
                            table.iloc[m, k-1] = str(table.iloc[m, k-1]) + ' ' + table.iloc[m, k]
                    else:
                        #print('table id:', i, '\trow number:', m, '\tcol id:', k, '\n')
                        #print('error when trying to combine columns')
                        continue
            
        final_drop_col = [k for k in no_header_col if k not in not_remove_col]
        table.drop(final_drop_col, axis = 1, inplace = True)
        table.columns = range(table.shape[1])
        
    alltables_clean_1.append(table)
             
    return alltables_clean_1
            


# #### post-processing: clean header and footnote

# In[5]:


def clean_h_f(table):

    # some header cleanning
    first_row = table.iloc[0, :].values
    first_row_ele = []
    for ele in first_row:
        if ele != '' and ele != ' ':
            first_row_ele.append(ele)
    if len(first_row_ele) == 1 and len(first_row_ele[0]) > 80:
        table = table.iloc[1:, :]
        table.index = np.arange(len(table))
            
            
    # and footnote cleanning: 
    if len(table) > 5:
        footnote_stop = 5
    else:
        footnote_stop = 2
    cur_footnote = []
    for m in range(len(table)-1, len(table)-footnote_stop, -1):
        cur_col_n = np.sum(table.iloc[m, :] != '') - np.sum(table.iloc[m, :] == ' ')
        if cur_col_n == 1:
            cur_footnote.append(m)
    if cur_footnote != []:
        table = table.iloc[:cur_footnote[-1], :]
        table.index = np.arange(len(table))
        
    
    return table


# #### post-processing: drop empty row and col

# In[6]:


def cleanempty(table):

# clean empty col
    drop_col = []
    if table is not None:
        for l in range(table.shape[1]):
            if np.sum(table.iloc[:, l] == '') + np.sum(table.iloc[:, l] == ' ') == table.shape[0]:
                drop_col.append(l)
    if drop_col != []:
        table = table.drop(drop_col, axis = 1)
        table.columns = np.arange(table.shape[1])
                
                
    # clean empty row
    drop_row = []
    if table is not None:
        for l in range(table.shape[0]):
            if np.sum(table.iloc[l, :] == '') + np.sum(table.iloc[l, :] == ' ') == table.shape[1]:
                drop_row.append(l)
    if drop_row != []:
        table = table.drop(drop_row, axis = 0)
        table.index = np.arange(table.shape[0])
        
    return table 


# #### post-processing: drop sparse row and col

# In[7]:


def cleansparse(table, sparse_thre = 0.6, ratio = 0.5):

    # clean again: col, check sparse cols
    col_sparse = 0
    if table is not None:
        for n in range(table.shape[1]):
            if (np.sum(table.iloc[:, n] == '') + np.sum(table.iloc[:, n] == ' ')) >= sparse_thre*table.shape[0]:
                col_sparse += 1
        if col_sparse >= ratio*table.shape[1]:
            table = None    
            
    # clean again: row, check sparse rows
    row_sparse = 0
    if table is not None:
        #if len(curtable[0]) <= 10:
            #start_row = 1 #2
        #else:
            #start_row = 1 #4
        start_row = 1
        for n in range(start_row, table.shape[0]):
            if (np.sum(table.iloc[n, :] == '') + np.sum(table.iloc[n, :] == ' ')) >= sparse_thre*table.shape[1]:
                row_sparse += 1
        if row_sparse >= ratio*table.shape[0]:
            table = None
            
    return table


# In[ ]:





# In[1]:



punc = '`~!@#$%^&*()_+{}|:"<>?-=[]\;\'\/,”“.'  # do not include '.'??

def get_headers(table_edges, allwords, page, word_thresh = 20, text_thresh = 40):
    
    header_rows = [k[0]-1 if k[0]!=0 else k[0] for k in table_edges]
    row_add = [max(0, k-1) for k in header_rows]
    header_rows.extend(row_add)
    header_rows.extend([0])
    header_rows = np.unique(header_rows)
    header_rows.sort()
    
    allwords_df = pd.DataFrame(allwords[page])
    row_counts = allwords_df['row'].value_counts()
    row_ids = list(row_counts.keys())
    single_row = [k for k in row_ids if row_counts[k] == 1]

    header_rows = [k for k in header_rows if k in single_row]
    
    header_text = {}
    
    for i in range(len(allwords_df)):
        if allwords_df.loc[i, 'row'] in header_rows:
            row_id = allwords_df.loc[i, 'row']
            row_text = allwords_df.loc[i, 'text']
            if len(row_text.split()) <= word_thresh and len(row_text) <= text_thresh and                 row_text.strip()[-1] not in punc:
                text_h = int(allwords_df.loc[i, 'bottom'] - allwords_df.loc[i, 'top'])
                is_upper = row_text.isupper()
                header_text[row_id] = [row_text, text_h, is_upper]

    return header_text


# In[ ]:




