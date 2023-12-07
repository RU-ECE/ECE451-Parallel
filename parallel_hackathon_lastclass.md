# Parallel Hackathon

## Instructions

Each group is to examine the following set of problems, sit together and strategize, split up and do as many of the problems as possible. You may create a repo if you know how, or a single google document and paste it all together if you don't. In the hierarchy, from best to worst results:

* Efficient, working, parallel code using great algorithms
* Working code for as many problems as possible
* Code that may not work, but is on the way and shows the structure
* Pseudocode that shows the idea your group has for this problem, even if you do not have the time to complete
* Notes on how the problem might be attacked

I will be going around helping groups. After 2 hours, I will select
each group to present something. You will submit the document in
canvas after the fact. As long as you make an attempt, you will not be
penalized for failure, but you must try for those two hours.

## Problems

1. Pythagorean Triplets. Write a program that accepts a number n on the command line. Find and count all pythagorean triplets up to n,n,n. For example, with n=10 your code should find (3,4,5), (6,8,10) count = 2. The idea is to come up with the most efficient parallel code to compute this for large $n=10^9$.

2. Perfect numbers. A perfect number n is one whose factors (excluding n) sum to the same as the number. For example $6 = 1 + 2 + 3$. $28 = 1 + 2 + 4 + 7 + 14$. Write a program that accepts a number n on the command line and prints the perfect numbers, and the count up to n. For example, for n=100, the output should be 6 28 count=2.

3. Finding common text sections. In project Gutenberg, text for open source books have been created and shared with the world. Each contains a lengthy and annoying header and trailer. The trailer is bigger and contains the legal license information. There is not a single one. Both header and trailer have varied over time. Write a program that finds common blocks of text at the beginning and end of a set of n documents, strips them out and creates a link. For example:

```text
This is standard header 1 of my project Gutenberg document
text goes here.
text text text
yada yada yada
This is the standard boring license v1. Why they do this I do not know.
Sometimes it is 5%
```

```text
This is standard header 1 of my project Gutenberg document
my 2nd text goes here.
text2 text2 text2
yada2 yada2 yada2
This is the standard boring license v1. Why they do this I do not know.
Sometimes it is 5% of the doc. I hate that
```

```text
This is standard header v2 of my project Gutenberg document
my 3rd text goes here.
text3 text3 text3
yada3 yada3 yada3
This is the standard boring license v2. Why they do this I do not know.
Sometimes it is 5% of the entire document. What a waste!
```

Write a program that creates a single header and footer file (you may create them all and then just choose one to use). Each file should have the standard header and footer stripped off

```html
<a href="header.html">Gutenberg header and license</a>

main body of text
```

Example Gutenberg books have been placed in the data section.


4. Partitioning into 2 Equal Sums

Given n distinct integers in a file (input.txt), read the numbers and divide them into two sets with the same sum. This requires that the sum of the numbers be even. Note the input set will be in random order and will not be consecutive numbers, the only guarantee is that each number is distinct. For example, given:

```cpp
1 2 3 4 5 6 7 8 9 10 11 
```
$sum=10(11)/2 = 132/2= 66$

you could split the set into two parts $a=2,10,4,8,9$ and $b=1,3,5,6,7,11$

5. Partitioning into two sets of equal size.
Given 2n numbers, find the two sets of equal size that sum to the same value.

For example, the set (1, 10, 2, 9, 3, 8, 4,7) has a sum of 44 which is even, so it can theoretically be split. The question then becomes is there a subset which has a sum of 22? $10 + 8 + 3 + 1 = 22$, and the other is $2 + 4 + 7 + 9$ so it works. It doesn't have to work. For example, the set (1,100,5,80, 15,60, 35,50) also has 8 members summing to $101+85+85+85=356$ but $178>100+50$ the remaining numbers are too big or too small. I picked numbers that are too big and too small to come up with the right sum. In this case, trying every permutation is going to be the only way to find out if there is an answer.
This algorithm will probably require backtracking and then just try to come up with a parallel implementation for speed. I suspect that 30 to 50 numbers will be hard. You can try with 1 to n for odd n, for which we know there is an answer. Then test with a different set.

1,31, 2, 30, 3, 29, 4, 28, 5, 27, 6, 26, ...