This dataset contains:
1. 8928 cartoon faces of 100 public figures
2. 1000 real faces of the same public figures
3. Annotation of various attributes, e.g., face bounding
box, age group, with or without glass, with or without beard,
facial expression, and gender.
4. Two .mat files: (i) IIITCFWdata.mat (for cartoon face
classification), (ii) queryImgInfo.mat (query images
and list of relavent cartoon faces).
5. Two .m file: to demonstrate data division and illustration
 
If you are interested in this database, please download IIITCFW1.0.zip,
and unzip it.
 
This dataset can be used for various tasks. In this release
we have provided train-test splits and usage instructions for
following two tasks: (Please use *exactly same splits* for consistency
in comparisions)
 
--------------------------------------------------------------------
        Task 1: Cartoon face classification
-------------------------------------------------------------------
Problem statement: Given a cartoon face belonging to one of 100 public
figures, recognize it.  
 
Usage instruction:
 
Step 1 (i): Open Matlab
 
Step 2: Load IIITCFWdata.mat. A structure IIITCFWdata
will be loaded. This structure contains following 8 fields:
(a)  imgId: shows the id of cartoon image.
(b)  imgName: the cartoon image file name
(c)  gender: male or female
(d)  age: old or young
(e)  glass: yes or no
(f)  beard: yes or no
(g)  **set: 1 (train), 2 (validation), 3 (test)**
(h)  class: a number from 1 to 100 showing class id.
 
Example I: 
>> load IIITCFWdata.mat
 
>> IIITCFWdata.imgId{1}
 
ans =
C1
 
>> IIITCFWdata.imgName{1}
 
ans =
cartoonFaces/AamirKhan0001.jpeg
 
>> I=imread(IIITCFWdata.imgName{1});
 
>> IIITCFWdata.set{1}   %% set=1 (train), set=2 (val), set=3(test)
 
ans =
 
     3  
 
** Use train and val set to training your classifier or feature computation,
use test set to report percentage accuracy of your cartoon face classification **
 
Example II: If you wish to read all the images and their annotation, then 
Run demoIITCFWdata.m. This demo will read each and every image in the
database, crop the face, display it, tell the class id of the image and
the set (train, test, or val) it belongs to. You can use this code as
wrapper and write your own training/testing modules.
 
--------------------------------------------------------------------
        Task 2: Photo2Cartoon
-------------------------------------------------------------------
Problem statement: Given a real face as a query retrive relavent cartoon
faces (face of the same person) from the database of cartoon faces.   
 
Usage instruction:
 
Step 1:  Open Matlab
 
Step 2: Load queryInfo.mat. A structure with 1000 cells
will be loaded (each cell contain information about one
query image). Each cell contain following six fields
(a)  queryImgName: shows the real face (query face) file name
(a)  queryImgId: shows the id of query face.
(b)  releventImgIds: Contains all relavent cartoon faces ids
(c)  releventImgNames: Contains all relavent cartoon face file names
(d)  set1: 1 or 0 (1 means use it for training)
(e)  set2: 1 or 0 (1 means use it for training)
 
(NOTE: set1 and set2 are two splits of database. Training should be
done on these splits (approximately 50% of data), while retriving
show results on complete database).
 
Example:
>> load queryInfo.mat
 
queryInfo{1}
 
ans = 
 
        queryImgName: 'realFaces/AamirKhan0001.jpg'
          queryImgId: 'R1'
      releventImgIds: {1x42 cell}
    releventImgNames: {1x42 cell}
                set1: 0
                set2: 1
 
>> queryInfo{1}.releventImgIds
 
ans = 
 
  Columns 1 through 10
 
    'C1'    'C2'    'C3'    'C4'    'C5'    'C6'    'C7'    'C8'    'C9'    'C10'
 
  Columns 11 through 19
 
    'C11'    'C12'    'C13'    'C14'    'C15'    'C16'    'C17'    'C18'    'C19'
 
  Columns 20 through 28
 
    'C20'    'C21'    'C22'    'C23'    'C24'    'C25'    'C26'    'C27'    'C28'
 
  Columns 29 through 37
 
    'C29'    'C30'    'C31'    'C32'    'C33'    'C34'    'C35'    'C36'    'C37'
 
  Columns 38 through 42
 
    'C38'    'C39'    'C40'    'C41'    'C42'
>> queryInfo{1}.releventImgNames
 
ans = 
 
  Columns 1 through 5
 
    [1x31 char]    [1x31 char]    [1x31 char]    [1x31 char]    [1x31 char]
 
  Columns 6 through 10
 
    [1x31 char]    [1x31 char]    [1x31 char]    [1x31 char]    [1x31 char]
 
  Columns 11 through 15
 
    [1x31 char]    [1x31 char]    [1x31 char]    [1x31 char]    [1x31 char]
 
  Columns 16 through 20
 
    [1x31 char]    [1x31 char]    [1x31 char]    [1x31 char]    [1x31 char]
 
  Columns 21 through 25
 
    [1x31 char]    [1x31 char]    [1x31 char]    [1x31 char]    [1x31 char]
 
  Columns 26 through 30
 
    [1x31 char]    [1x31 char]    [1x31 char]    [1x31 char]    [1x31 char]
 
  Columns 31 through 35
 
    [1x31 char]    [1x31 char]    [1x31 char]    [1x31 char]    [1x31 char]
 
  Columns 36 through 40
 
    [1x31 char]    [1x31 char]    [1x31 char]    [1x31 char]    [1x31 char]
 
  Columns 41 through 42
 
    [1x31 char]    [1x31 char]
 
>> queryInfo{1}.releventImgNames{1}
 
ans =
 
cartoonFaces/AamirKhan0001.jpeg
 
>> queryInfo{1}.releventImgNames{42} %% there are total 42 relevent images for this pair
 
ans =
 
cartoonFaces/AamirKhan0042.jpeg
 
>> queryInfo{1}.set1
ans =
 
     0
%% DO NOT USE it for training in split 1
 
>> queryInfo{1}.set2
ans =
 
     1
%% USE it for training in split 2
 
[Note: The training set should be used for learning parameters or
fine tuning. However, please report reterival performance on the 
"whole set". If train your reterival system then please train in
two splits and report the average reterival performance.]
 
------------------------------------------------------------------------------
If you use this dataset, please cite the following paper.
@InProceedings{MishraECCV16,
  author    = "Mishra, A., Nandan Rai, S., Mishra, A. and Jawahar, C.~V.",
  title     = "IIIT-CFW: A Benchmark Database of Cartoon Faces in the Wild",
  booktitle = "VASE ECCVW",
  year      = "2016",
}
 
For any queries about the dataset contact: anand.mishra@research.iiit.ac.in
 


