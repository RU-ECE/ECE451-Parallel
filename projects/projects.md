# Project Ideas

1. Process project Gutenberg books
  Build a dictionary of all words in a large set of project Gutenberg public domain books. To build the dictionary, you will have to count words.

        struct word_info {
	  uint32_t id; // unique id of the word
	  uint32_t frequency;
	  uint32_t num_books; // number of books containing this word

        }
  tasks: stripping punctuation, building distributed dictionary
         merging the dictionaries to construct one

  further: you can categorize books and build multiple dictionaries,
        There are old books in Shakespearean English, modern English


2. Ray Tracing.
   I have simple code for ray tracing based on Ray tracing in one weekend.
   Define a way to generate a movie in parallel across multiple computers (MPI)
   using Cuda on each Frame

3. Define a hardware architecture that is better than AVX for streaming algorithms. For example, we should be able to do a lot better for sorting. Currently we have to read and write memory a lot.

   Verilog, computer architecture, research
             r0
   LOAD V1  # 1, 3, 6, 8, 9 12, 15, 19

             r1
   LOAD V2  # 1, 4, 19, 20, 26, 31, 45

              r2
   Load V3  # 2, 3, 8, 10, 12, 15, 20
   LOAD V4  # -10, -9, -7, -6, -6, -6, -5
   MERGE V16, V1, v2, v3, v4

   Load V5
   Load V6
   Load V7
   Load V8
   Merge V17

   SENDTO v16 parentcpu
   SENDTO v17 parentcpu


4. 