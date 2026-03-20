

'''

# Three phases

1. Performance 

Generate 5 queries of varying complexity based on Prompt (human written prompt, based on likely use cases) 
Run 5 queries and record for each step: latency, tokens used, memory used, and cost. 
Print in a table. 
Add to a log with date and version used. 
(Keep in db ? same db or different one?)

2. Quality 

Functional correctness 
Generate 5 queries of varying complexity based on Prompt (human written prompt, based on likely use cases) 
Are citations accurate? use LLM to check if the answer is supported by the cited sources.
Are paper numbers correct? use LLM to check if the cited page numbers match the content cited.


Search accuracy ( using bootstrapped ground truth data)
How can we test the accuracy of the search, reranking and returned embeddings?

Make ground truth data for a set of queries, with associated relevant documents and chunks. ( Golden data set at start - stays consistent))
(We can then compare the results of the vector search and reranking against the ground truth.) 

Then run query against the system and evaluate: 
Context Precision (ranking relevant chunks higher),
Context Recall (finding all necessary information),
Groundedness
Give scores and keep log.  

For later: 
When using approximate nearest Neighbour - compare with vector brute force search.

3. Regression 

When changes are made to the codebase: 
does modification break the code?  
- automated suite of tests 
	- function tests 
	- unit tests ( individual functions / scripts etc.)
- compare results benchmarks of safe/stable version  


'''