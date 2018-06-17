# Hybrid-Text-Mining
Report on Hybrid Text Mining 




HYBRID TEXT MINING
Independent Study Report
ISM6905







By: Charu Mathur(U74078511)
Program: Business Analytics and Information Systems
Professor: Dr. Don Berndt



Contents
Introduction	3
Background and Related Work	4
Analysis Performed	5
Corpus description	6
Removal of Stopwords	6
Stemming	7
Lemmatization and Tokenization	7
Data frame construction	8
Term frequency- Inverse document frequency (tf- idf)	8
Cosine similarity	10
K means clustering	10
Future Work	12
Multidimensional Scaling	12
Principal Coordinates Analysis	12
Visualizing document clusters	12
Topic Modelling	13
Conclusion	14
References	14








Introduction

As the volume of medical information grows steadily, the need for developing dedicated computational tools for information organization and mining becomes more pronounced. The growth in the size and variety of unstructured information poses both challenges and opportunities. The challenge is related to efficient information search and retrieval when dealing with a large volume of heterogeneous and unstructured information. Traditional search methods, such as keyword search, with their limited semantic capabilities, can no longer meet the information retrieval and organization needs of the cyber era. More advanced computational tools and techniques are needed that can facilitate search, organization, and summarization of large bodies of text more effectively. At the same time, the unstructured text available on the Internet contains valuable information that can be extracted and transformed into business intelligence to support knowledge-based systems.
Fall-related injuries are an important healthcare issue, especially among aging populations. A recent study found approximately 40% of people age 70 years and older reported falling during a 1-year period.1 Injuries due to falls are a leading cause of death and disability among older adults .A history of a previous fall is one of the most important clinical indicators that identify an elderly patient as being at high risk of future falls.6 However, falls have been found to be under-coded in administrative databases,7 making it difficult to identify at-risk patients and thus take steps to help prevent falls. An alternative source of fall-related information may be found in the clinical text associated with a patient’s electronic health record (EHR). This study explores how well falls, associated with an ambulatory encounter for the treatment of an injury, can be identified in clinical text. The study revolves around an approach to investigates a hybrid text mining strategy based for facilitating search and organization of electronic health record documents using document clustering by adopting methods using K-means.





Background and Related Work

The Veterans Health Administration’s (VHA) EHR contains almost two billion clinical and provides a rich repository to assess the effectiveness of automated text-based surveillance systems. Given this resource, we explored how well statistical text mining (STM), a machine learning approach that represents documents as a ‘bag of words’, could classify individual clinical documents (progress notes, reports, etc.) as being fall related or not. Within the VHA’s EHR, clinical documents are assigned a title that reflects either the place of service or clinical author. Example titles include ‘Emergency Department’ progress notes, ‘Nursing Triage’ progress notes, or ‘Orthopedic Surgery Consult’ progress notes. As patients may receive care for a fall from multiple sources (e.g. ED, outpatient clinic), fall-related information is likely to be found in a variety of document titles. We therefore selected a heterogeneous collection of documents, representing a wide variety of document titles, each with varying clinical sublanguages,8 to help maximize the discovery of fall-related documents and assess model performance across a variety of document types. Finally, we also explored how well our models generalized by building models using a single site and then applying those models to three other sites. This method mimics situations in which a system is built and used at a single facility and then later rolled out to other facilities.
This study demonstrated that STM could reliably identify falls in clinical text related to ambulatory events. STM is an inductive approach based on machine learning that uncovers patterns from textual documents such as clinical progress notes. STM induces structure on unstructured text by representing each document within a term-by-document matrix, in which each term from the document collection is represented in the matrix. Using a dataset of over 26 000 documents, the STM models were able to obtain AUC scores above 0.95 across four sites and over 600 distinct document titles. The results of this study suggest that STM-based fall models have the potential to improve surveillance of falls. For instance, patients could be flagged in real time as being at a higher risk of a fall given a history of falls documented in their EHR. In addition, results can be rolled up to the patient level of analysis to obtain regional or national statistics of fall prevalence for safety reporting measures. Finally, STM may also be useful in identifying other fall-related issues that may be buried within text, such as the place the fall occurred (for reimbursement purposes) or the type of injury sustained. Finally, the encouraging evidence shown here that STM is a robust technique for mining clinical documents bodes well for other surveillance-related topics.
Analysis Performed

Traditionally, the clustering of documents was carried out manually. But, with the advent of machine learning and various text classification algorithms it was possible for the computers to take upon this task. These algorithms use training datasets for leaning. Hence, the results of these algorithms would be strongly based on the input datasets provided to it and is not always highly reliable as there are many new terms and concepts which are born every day. It would be very difficult to keep a check on them and have experts identified training examples for each text class thus generated and to learn a classifier for it in the above manner. Various clustering techniques have been employed to make this process automatic. The appealing characteristic of cluster analysis is that we can find clusters directly from the given data without relying on any pre-determined information.
The standard method for information search and retrieval from the electronic medical records is the keyword-based method. For example, in a patient search scenario, a doctor from the medical industry who is looking for patients with similar medical conditions as their current patient injured due to falls in old age, can simply use fatal falls in old age and falls as the search keywords in a generic search engine. But the sheer size of the returned set would undermine the usefulness of the search result. One way to make the results more useful is to present them to the user as chunks or clusters of similar documents and then characterize each cluster using a set of features or themes. In the above example, a cluster characterized by features such as fatal falls, old age, fall injuries and disabilities by falls would be of interest for the user if inspection and assembly were the secondary services that the user is looking for. This work proposes a hybrid text mining technique, which facilitates automatic clustering and characterization of the documents available in a large medical records corpus.

 
The preliminary stages involve removing the stop words, followed by stemming and tokenizing the document. The next stages consist of constructing a tf-idf matrix, calculating the cosine similarities and cosine distances between the documents. The final level of execution involves clustering and multidimensional scaling to reduce the dimensionality within the corpus. The last stage is the visualization of the generated document clusters. 

Corpus description 
 
The first step is to create a corpus of medical documents to be used as the test data. Within the VHA’s EHR, clinical documents are assigned a title that reflects either the place of service or clinical author. Example titles include ‘Emergency Department’ progress notes, ‘Nursing Triage’ progress notes, or ‘Orthopedic Surgery Consult’ progress notes. As patients may receive care for a fall from multiple sources (e.g. ED, outpatient clinic), fall-related information is likely to be found in a variety of document titles. We therefore selected a heterogeneous collection of documents, representing a wide variety of document titles, each with varying clinical sublanguages,8 to help maximize the discovery of fall-related documents and assess model performance across a variety of document types.
 
Removal of Stopwords 
 
Large number of words make up a document but only a few of them make a significant contribution to it. Words like IT, AND, TO, ARE, THE can be found in almost every sentence of the English language. These words are called as stop words and make up a large fraction of the text in most of the documents. they have a very less significance in terms of Information Retrieval (IR) and are therefore called as stopwords, noise words or the negative dictionary. So, it is usually worthwhile to remove or ignore all stopwords when performing analysis or processing queries over text documents. In the first phase of the execution, all such stop words in the English language are removed. NLTK’s list of stop words are used to identify and separate them from the corpus. 
 
 
Stemming 
 
Morphology is the identification, analysis and description of the structure of a given language's morphemes and other linguistic units, such as root words, affixes, parts of speech, intonations and stresses, or implied context. it is one of the characteristics of text mining that must be taken into account while performing text analysis. Considering the set of words, democracy, democratic and democratization, it is observed that the words, democratization, democratic are generated by adding a suffix for the word ‘democra’ which is called the stem. Hence, all such words deriving from the same stem ‘democra’ can be represented as democra*, where the symbol * denotes a variable length don’t-care match. This has led to the development of conflation techniques which permit the matching of different forms of the same word. In this project, one of the most reliable and highly effective stemmer called the snowball stemmer is used to perform stemming. Snowball is a small string processing language designed for creating stemming algorithms for use in Information Retrieval. The English snowball stemmer breaks down any word from the English vocabulary to its root word or the stem. 
 
Lemmatization and Tokenization 
 
The goal of both stemming and lemmatization is to reduce the declensional forms and derivationally limited forms. Lemmatization uses vocabulary and morphological analysis of words to remove inflectional endings only and return the base or the dictionary form of a word called as lemma. For example, when referring to a word saw, stemming might just return s, whereas lemmatization would return either see or saw depending on whether it was used as a verb or noun in the context. In this work a function is defined which tokenizes the whole document and stems each token thus generated while the other function only tokenizes the document.
 
 
Data frame construction
 
Data Frame is a two-dimensional labeled data structure with columns of potentially different types. In this structure, each column contains measurements of one variable and each row contains one case. It accepts many kinds of inputs like lists, dictionaries, series, 2D NumPy. ND array, structured or record ND array, a series, or any other data frame. In this module the pandas data frame data structure is implemented. It is a two-dimensional, size-mutable, potentially heterogeneous tabular data structure with labeled axes (rows and columns). Arithmetic operations align on both row and column labels. It can be considered as a dictionary-like container for series objects. A Data Frame with stemmed vocabulary as index and tokenized words as the column is created. The advantage of it is that any stem word can be looked up to return a full token.

Term frequency- Inverse document frequency (tf- idf) 
 
Tf- idf or term frequency-inverse document frequency, in text mining and information retrieval, has much importance. The tf-idf weight tells us how important a word is to a document in a collection or corpus. This importance is directly proportional to the number of times a word appears in a document. Inherently, if a word appears frequently in a document, it’s important, so the word is given a high score. But, if the same word appears in many documents, it’s not a unique identifier. So, the word is given a low score. The tf-idf weight is made of two terms, Term Frequency (TF) and Inverse Document Frequency (IDF). Term frequency can be calculated by dividing the number of times a word appears in a document by the total number of words in that document. Whereas, the Inverse Document Frequency can be computed as the logarithm of the number of the documents in the corpus, divided by the number of documents where the specific term appears. 
 
 
 
With N: total number of documents in the corpus N=|D| and   is the number of documents where the term t appears (i.e., tf (t, D) ). 
A high weight in tf–idf is reached by a high term frequency (in the given document) and a low document frequency of the term in the whole collection of documents; the weights hence tend to filter out common terms. Since the ratio inside the idf's log function is always greater than or equal to 1, the value of idf (and tf-idf) is greater than or equal to zero. As a term appears in more documents, the ratio inside the logarithm approaches 1, bringing the idf and tf-idf closer to 0. In this method, the term frequency-inverse document frequency (tf-idf) vectorizer parameters are defined and then the document content list is converted into a tf-idf matrix. To get this matrix, the word occurrences by document are counted and transformed into a document-term matrix (DTM). It’s also called a term frequency matrix. The last preprocessing step is to generate the Document-Term Matrix (DTM) for the manufacturing corpus. DTM is a matrix containing the frequency of the terms in the manufacturing documents. In the DTM, documents are denoted by rows and the terms are represented by columns. If a term is repeated n times in a specific document, the value of its corresponding cell in the matrix is n. The DTM represents the vector model of the corpus and is used as the input to the next step. The term frequency inverse document frequency weighting is computed, and three parameters are defined. They are df_max, idf_min and ngram_range. df_max is the maximum frequency within the documents a given feature can have, which is to be used in the tf-idf matrix. Idf_min is an integer then the term would have to be in at least the integer number of documents specified to be considered. ngram_range is the appropriate range of ngrams which is user defined depending on the corpus. 
 

Cosine similarity 
 
Cosine similarity is a measure of similarity between two vectors of an inner product space that measures the cosine of the angle between them. This metric can be considered as a measurement of orientation and not magnitude.  It is a comparison between documents on a normalized space because, not only the magnitude of each word count (tf-idf) of each document is considered but also the angle between the documents. 
The cosine of two vectors can be derived by using the Euclidean dot product formula: 
 

Given two vectors of attributes A and B, the cosine similarity, cos), is represented using a dot product and magnitude as, 
  
 
where, Ai and Bi are components of vector A and B respectively. The resulting similarity ranges from -1 meaning exactly opposite, to 1 meaning the same, with 0 indicating orthogonality, and in between values indicating intermediate similarity or dissimilarity. In this section the cosine similarity against the tf-idf matrix is measured. Cosine distance is calculated as one minus the cosine similarity of each document. 
 
K means clustering 
 
K-means clustering is a method used to partition a data set into K groups automatically. Initially K clusters are selected, and they are iteratively refined through the process in the following manner. 
1)	Firstly, the closest cluster center for a point is identified and this point is allocated to it. 
2)	Each cluster center Cj is updated to be the mean of its constituent points . 
 
This algorithm generates clusters by optimizing a criterion function, the Sum of Squared Error (SSE) given by: 
 
                          
Where ||∙||2 , denotes the Euclidian (ℒ2) norm and 
  
This is the centroid of cluster whose cardinality is | |. The optimization of SSE is often referred to as the minimum SSE clustering (MSSC) problem. appropriate integer number of predetermined clusters are initialized. Then, each observation is assigned to a cluster to minimize the minimum sum of squares error. The mean of the clustered observations is calculated and used as the new cluster centroid. These observations are reassigned to clusters and centroids are recalculated in an iterative process until the algorithm reaches the convergence. It needs to be run several times for the algorithm to converge to a global optimum as K means is susceptible to reaching local optima. A dictionary of ‘titles’ is created which contains the title of the document, ‘synopses’, which contains the actual content of the document and ‘clusters’ which is presently empty, would later be filled with the number of the cluster to which the document belongs. A data frame with a list named ‘clusters’ as index and ‘titles’, ‘synopses’ as the columns is created. After this, the top n terms for each synopsis that are nearest to the cluster centroid are identified. This gives a good sense of the main topic of the cluster. 


 



Future Work
 
Multidimensional Scaling 

Multidimensional Scaling is a means to visualize the level of similarity of individual cases in the dataset. It is particularly used to display the information contained in a distance matrix. It uses a set of related ordination techniques popular in information visualization. The MDS algorithm places each object in N-dimensional space such that the distances between the objects are preserved as much as possible. Each object is then assigned coordinates in each of the N dimensions. There may be more than 2 dimensions in an MDS plot and it is specified by the priori. Choosing N=2 optimizes the object locations for a two-dimensional scatter plot.  
 	 
Principal Coordinates Analysis 
 
It is also known as Torgerson Scaling or Torgerson–Gower scaling. A matrix containing the dissimilarities between pairs of items is taken as an input and the coordinate matrix whose configuration minimizes a loss function, also referred to as strain, is given as an output. The distance matrix is converted into a two-dimensional array using multidimensional scaling. Principal component analysis can also be used to achieve this. 

Visualizing document clusters 
 
Matplotlib is a python 2D plotting library. It can be used in python scripts, the python and ipython shell (alaMATLAB or Mathematica), web application servers, and six graphical user interface toolkits. The process of visualizing the clusters with matplotlib can be achieved by implementing the below process. Firstly, a dataframe that comprises of the result of the data frame which was produced in the previous module with the cluster members and also the titles are created. A plot is set up, the necessary margins are added, and scale is set. Now, iteration is done through groups to layer the plot. The cluster category is matched with the document name giving it a specific color and a position in the plot. 

Topic Modelling

Document clustering results in partitioning a heterogeneous dataset into multiple clusters with more similar members. However, it doesn’t provide any description or characterization for the generated clusters. Topic Modeling is a text mining technique for analyzing large volumes of unlabeled text. Latent Dirichlet Allocation (LDA) is used as the underlying algorithm for topic modeling. LDA technique can be used for automatically discovering abstract topics in a group of unlabeled documents. A topic is a recurring pattern of words that frequently appear together. For example, in a collection of documents that are related to banking, the terms such as interest, credit, saving, checking, statement, and APR define a topic as they co-occur frequently in the documents. LDA technique assumes that each document in the dataset is randomly composed of a combination of all available topics with different probabilities for each.










Conclusion 

This report presents a novel approach to extract a medical concept from a document and cluster such set of documents depending on the concept extracted from each of them. We transform the corpus into vector space by using term frequency–inverse document frequency then calculates the cosine distance between each document, followed by clustering them using K means algorithm. We also use multidimensional scaling to reduce the dimensionality within the corpus. It results in the grouping of documents which are most similar to each other with respect to their content and the genre. The performance of the proposed method can be further improved through multiple iterations and subsequent elimination of less informative words under each topic to evaluate the performance of different topic modeling algorithms that can be used in the proposed framework. When highly informative terms are clustered together under a topic, the likelihood of discovering useful patterns in data increases. This research would be extremely helpful in analysis of large number of Electronic Health Records in Medical Research by optimizing the research time of different cases.

References

1.	Berndt_AMIA2010MOE.pdf
2.	McCart_JAMIA2013FallsSTM.pdf
3.	Berndt_TMIS2015TMQuaity.pdf
4.	https://datawarrior.wordpress.com/2018/01/22/document-term-matrix-text-mining-in-r-and-python/
5.	http://sapir.psych.wisc.edu/programming_for_psychologists/cheat_sheets/Text-Analysis-with-NLTK-Cheatsheet.pdf
6.	https://pdfs.semanticscholar.org/9f06/23edd72cb84a3cc7e37bd368d8e442a993db.pdf

