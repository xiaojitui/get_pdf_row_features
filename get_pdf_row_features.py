#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import os
import copy
#import pdfplumber
#import camelot
#import PyPDF2

import tablefinder


# In[ ]:


def get_lines(alllines, allrects, page):

    y = []
    if alllines[page] is not None: 
        for ele in alllines[page]:
            if ele['top'] not in y:
                y.append(int(ele['top']))
            if ele['bottom'] not in y:
                y.append(int(ele['bottom']))
        
    if allrects[page] is not None:           
        for ele in allrects[page]:
            if ele['top'] not in y:
                y.append(int(ele['top']))
            if ele['bottom'] not in y:
                 y.append(int(ele['bottom']))
    if y != []:
        y = np.unique(y)
    return y


# In[ ]:


def get_row_features(allwords_df, alllines_df, allrects_df, pages, page): 
    
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
            while (i < len(page_positions) - 1) and page_positions.loc[i, 'row'] == page_positions.loc[i+1, 'row']:
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
   
    page_positions['to_pre'] = 0
    for i in range(1, len(page_positions)):
        page_positions.loc[i, 'to_pre'] = abs(page_positions.loc[i, 'top'] - page_positions.loc[i-1, 'bottom'])
        
    page_positions['to_next'] = 0
    for i in range(len(page_positions) - 1):
        page_positions.loc[i, 'to_next'] = abs(page_positions.loc[i, 'bottom'] - page_positions.loc[i+1, 'top'])
        
        
    page_positions['row_ele'] = 0
    page_positions['row_min'] = 0
    page_positions['row_max'] = 0
    row_x = {}
    for i in range(len(page_positions)):
        cur_row_ele = 0
        cur_row_min = np.inf
        cur_row_max = 0
        row_x[i] = []
        #cur_row_match = 0
        #cur_top_line = 0
        #cur_bottom_line = 0
        for j in range(len(cur_page_df)):
            if cur_page_df.loc[j, 'row'] > page_positions.loc[i, 'row']:
                break
            if cur_page_df.loc[j, 'row'] == page_positions.loc[i, 'row']:
                cur_row_ele += 1
                cur_row_min = min(cur_row_min, len(cur_page_df.loc[j, 'text']))
                cur_row_max = max(cur_row_max, len(cur_page_df.loc[j, 'text']))
                row_x[i].append(cur_page_df.loc[j, 'x0'])
                row_x[i].append(cur_page_df.loc[j, 'x1'])
        page_positions.loc[i, 'row_ele'] = cur_row_ele
        page_positions.loc[i, 'row_min'] = cur_row_min
        page_positions.loc[i, 'row_max'] = cur_row_max
    
    ### match previous row
    page_positions['match_pre'] = 0 
    for i in range(1, len(page_positions)):
        math_pre = 0
        for j in range(len(cur_page_df)):
            if cur_page_df.loc[j, 'row'] > page_positions.loc[i, 'row']:
                break
            if cur_page_df.loc[j, 'row'] == page_positions.loc[i, 'row']:
                if cur_page_df.loc[j, 'x0'] in row_x[i-1] or cur_page_df.loc[j, 'x1'] in row_x[i-1]:
                    math_pre += 1
        page_positions.loc[i, 'match_pre'] = math_pre
    
    ### match next row
    page_positions['match_next'] = 0 
    for i in range(len(page_positions) - 1):
        math_next = 0
        for j in range(len(cur_page_df)):
            if cur_page_df.loc[j, 'row'] > page_positions.loc[i, 'row']:
                break
            if cur_page_df.loc[j, 'row'] == page_positions.loc[i, 'row']:
                if cur_page_df.loc[j, 'x0'] in row_x[i+1] or cur_page_df.loc[j, 'x1'] in row_x[i+1]:
                    math_next += 1
        page_positions.loc[i, 'match_next'] = math_next
        
        
    rowlines = get_lines(alllines_df, allrects_df, page)
    
    page_positions['top_to_line'] = 0 
    if rowlines != []:
        for i in range(len(page_positions)):
            for rowline in rowlines:
                if abs(page_positions.loc[i, 'top'] - rowline <= 5):
                    page_positions.loc[i, 'top_to_line'] = 1
                    break
                    
    page_positions['bottom_to_line'] = 0 
    if rowlines != []:
        for i in range(len(page_positions)):
            for rowline in rowlines:
                if abs(page_positions.loc[i, 'bottom'] - rowline) <= 5:
                    page_positions.loc[i, 'bottom_to_line'] = 1
                    break
    
    # get space ratio
    page_positions['spaceratio'] = 0 
    for i in range(len(page_positions) - 1):
        nonspacelen = 0
        for j in range(len(cur_page_df)):
            if cur_page_df.loc[j, 'row'] > page_positions.loc[i, 'row']:
                break
            if cur_page_df.loc[j, 'row'] == page_positions.loc[i, 'row']:
                nonspacelen += int(cur_page_df.loc[j, 'x1'] - cur_page_df.loc[j, 'x0'])
        page_positions.loc[i, 'spaceratio'] = (int(pages[page].width) - nonspacelen)/int(pages[page].width)
    
    
    # normalize 
    #page_positions['top'] = page_positions['top']/int(pages[page].height)
    #page_positions['bottom'] = page_positions['bottom']/int(pages[page].height)
    
    
    return page_positions


# In[1]:


def get_table_feature(table_gt, doc_folder, pdfname):
    data = []
    pdfrecords = table_gt[table_gt['filename'] == pdfname]
    pdffile = os.path.join(doc_folder, pdfname + '.pdf')
    
    ################## parse text info
    alltext, allwords, alllines, allrects, pages = tablefinder.get_text_word(pdffile)
    #alltext_1 = tablefinder.get_text_word_pypdf2(pdffile)
    
    ################## clean pages
    page_tracker = np.arange(len(pages))
    page_tracker_clean = []
    image_pages = []
    for k in page_tracker:
        if alltext[k] is not None and np.sum([k != '' and k !=' ' for k in alltext[k].split('\n')]) > 4:
            page_tracker_clean.append(k)
        elif alltext[k] is None:
            image_pages.append(k)
        else: 
            continue
            
    ################## corect rotation 
    rotation = tablefinder.checkrotation(alltext, page_tracker_clean)

    allwords_df = copy.deepcopy(allwords) # just be safe... use a copy 
    alllines_df = copy.deepcopy(alllines)
    allrects_df = copy.deepcopy(allrects)

    allwords_df, alllines_df, allrects_df = tablefinder.correctrotation(pdffile, allwords_df, alllines_df, allrects_df, rotation)
    allwords_df = tablefinder.groupwords(allwords_df)
    
    ################# get page features
    for page in page_tracker_clean: 
        cur_feature = get_row_features(allwords_df, alllines_df, allrects_df, pages, page)
        cur_feature['label'] = 0

        if page in [int(k) for k in pdfrecords['page_n']]:

            cur_records = pdfrecords[pdfrecords['page_n'] == str(page)]

            for i in range(len(cur_records)):
                top = cur_records.loc[:, 'cor_y1'].values[0]
                bottom = cur_records.loc[:, 'cor_y2'].values[0]

                for j in range(len(cur_feature)):
                    if cur_feature.loc[j, 'top'] >= top - 2 and cur_feature.loc[j, 'bottom'] <= bottom + 2:
                        cur_feature.loc[j, 'label'] = 1

        cur_feature.loc[:, 'top'] = cur_feature.loc[:, 'top']/int(pages[page].height)
        cur_feature.loc[:, 'bottom'] = cur_feature.loc[:, 'bottom']/int(pages[page].height)

        data.append(cur_feature)

    data = pd.concat(data, axis = 0)
    
    return data


# In[ ]:


## test 



if __name__ == '__main__':
	gt_path = './data/gt.csv'
	doc_folder = '.data/docs/'

	table_gt = pd.readcsv(gt_path)
	pdfnames = table_gt['filename'].unique()
	# table_gt is the grouna truth table
	# cols: filename, region_id, table_id, page_n, x1, y1, x2, y2, cor_x1, cor_y1, cor_x2, cor_y2

	data = {}
	for i in range(len(pdfnames)):
		pdfname = pdfnames[i]
		data[pdfname] = get_table_feature(table_gt, doc_folder, pdfname)
	# the format of data[pdfname] is a dataframe, with cols of:
	# [row,top,bottom,to_pre,to_next,row_ele,row_min,row_max,match_pre,match_next,top_to_line,bottom_to_line,spaceratio,label]




# In[ ]:

'''
## to build models on these feature, just do:
X = data[pdfname].iloc[:, 1:-1]
y = data[pdfname].iloc[:, -1]
# then build model


'''

#



