# get_pdf_row_features

The script is used to generate features of each row in a PDF page. 
<br><br>
PDF files should be saved in the './data/docs' folder. 

<br><br>
For one example, the row features can be used to train a classification model to detect if the row is inside a table or not, which helps recognize table areas in a PDF. To do so, a ground truth csv need to be prepared first and saved as './data/gt.csv'. The columns include:  
- filename, region_id, table_id, page_n, x1, y1, x2, y2, cor_x1, cor_y1, cor_x2, cor_y2

<br><br>
To generate row features, just run: 
- python get_pdf_row_features.py

<br><br>
Specifically, the generated row features are saved in a dataframe format, with columns of:
- row: row id, 
- top: row top position y1 (starting from top of the page),
- bottom: row bottom position y2 (starting from top of the page)
- to_pre: distance to previous row
- to_next: distance to next row
- row_ele: number of words in the row
- row_min: minimum string length of a word in the row
- row_max: maximum string length of a word in the row
- match_pre: number of words which are aligned with words in the previous row (have same x coordinates)
- match_next: number of words which are aligned with words in the next row (have same x coordinates)
- top_to_line: 1 or 0, indicate if the top of the row is close to a line on the page 
- bottom_to_line: 1 or 0, indicate if the bottom of the row is close to a line on the page 
- spaceratio: ratio of total empty space length to the row width
- (if have 'gt.csv') label: 1 or 0, indicate if the row is inside a table area or not, based on groundtruth csv. 

<br><br>
For example, to classify if if a row is inside a table area or not , just do:
- X = output[pdffilename].iloc[:, 1:-1] # all features
- y = output[pdffilename].iloc[:, -1] # target
then build any classification model to train. 


