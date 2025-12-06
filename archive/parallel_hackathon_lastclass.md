# Parallel Hackathon

## Instructions

Each group should:

1. Read the set of problems below.
2. Sit together and **strategize**.
3. Split up work and try to solve **as many problems as possible**.

You may:

- Create a **Git repo** if you’re comfortable with that, **or**
- Use a **single Google Doc** and paste everyone’s work into it.

In order of best to worst results:

1. Efficient, **working, parallel code** using good algorithms.
2. **Working code** for as many problems as possible.
3. Code that may not work yet, but **clearly shows the structure/approach**.
4. **Pseudocode** that shows your idea for the problem, even if you don’t have time to implement it.
5. **Notes** on how you might attack the problem.

I will be going around helping groups. After about 2 hours, each group will present something.  
You will submit your document in Canvas afterwards.

As long as you make a **good-faith attempt** for those two hours, you will **not be penalized for failure**—but you must
actually try.

---

## Problems

### 1. Pythagorean Triplets

Write a program that accepts a number `n` on the command line and finds all **Pythagorean triplets** $(a,b,c)$ with:

- $a^2 + b^2 = c^2$
- $1 \le a, b, c \le n$

For example, with `n = 10`, your code should find:

- $(3, 4, 5)$
- $(6, 8, 10)$

So the count is `2`.

Your task is to come up with the most **efficient parallel code** to compute this for large $n$, e.g. **$n = 10^9$**.

Think about:

- How to **partition the search space**.
- How to **avoid redundant work**.
- Whether you can **avoid checking obviously impossible combinations**.

---

### 2. Perfect Numbers

A **perfect number** $n$ is a positive integer whose proper divisors (factors excluding $n$ itself) **sum to $n$**.

Examples:

- $6 = 1 + 2 + 3$
- $28 = 1 + 2 + 4 + 7 + 14$

Write a program that:

- Accepts a number `n` on the command line.
- Prints all perfect numbers $\le n$ and their **count**.

Example:

- Input: `n = 100`
- Output: `6 28 count = 2`

Goal: design this to scale as well as possible in **parallel**.

---

### 3. Finding Common Text Sections (Gutenberg Headers/Footers)

Project Gutenberg books are open-text files with **annoying headers and trailers** (license blocks). These:

- Are **not identical across all books**.
- Have **changed over time**.
- Can be quite **large** (e.g., 5% of the file).

Example documents:

```text
This is standard header 1 of my project Gutenberg document
text goes here.
text text text
yada yada yada
This is the standard boring license v1. Why they do this I do not know.
Sometimes it is 5%
````

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

Write a program that:

1. Finds **common blocks of text** at:
	* The **beginning** (headers), and
	* The **end** (footers) of a set of `n` documents.
2. Strips those common blocks out.
3. Creates a **link** to a separate header/footer file.

For each processed book, the result should look something like:

```html
<a href="header.html">Gutenberg header and license</a>
main body of text
```

Guidelines:

* You may generate **multiple candidate** header/footer files and then choose one to use.
* Example Gutenberg books are provided in the **data** section.

Think about parallelism:

* How to process many documents at once.
* How to detect common prefix/suffix strings efficiently.

---

### 4. Partitioning into Two Equal Sums

Given `n` **distinct integers** in a file (e.g., `input.txt`), read the numbers and divide them into **two sets with the
same sum**.

Requirements:

* The **total sum** of the numbers must be **even**; otherwise, it’s impossible.
* The input:
	* Is in **random order**.
	* Numbers are **distinct**.
	* They are **not guaranteed** to be consecutive.

Example input:

```text
1 2 3 4 5 6 7 8 9 10 11
```

The sum is:

* $\text{sum} = 11 \cdot 12 / 2 = 132 / 2 = 66$

One valid partition:

* Set $A = {2, 4, 8, 9, 10}$
* Set $B = {1, 3, 5, 6, 7, 11}$

Both sets sum to 33? (You should verify and find a correct partition.)
Your task is to **find any** valid partition (or report that none exists).

Consider:

* This is related to the **subset sum / partition problem**.
* It can be **hard** in the worst case (exponential), but small cases may be manageable.
* Try to design a **parallel** approach (e.g., divide search space across threads).

---

### 5. Partitioning into Two Sets of Equal Size and Equal Sum

Now we add an additional constraint: given **$2n$ numbers**, find **two sets** of:

* **Equal size** (each has $n$ elements),
* With the **same sum**.

Example:

* The set $(1, 10, 2, 9, 3, 8, 4, 7)$ has:
	* 8 numbers (so $n = 4$),
	* Total sum $44$, which is **even**.

We ask: is there a **subset of 4 numbers** that sums to $22$?

Yes:

* $10 + 8 + 3 + 1 = 22$
* The remaining 4 numbers are $2 + 4 + 7 + 9 = 22$

So this set **can be split** into two equal-size subsets with equal sum.

It **does not have to work** in general. For example:

* The set $(1, 100, 5, 80, 15, 60, 35, 50)$ also has 8 members and an even sum,
  but there may be **no** way to split it into two size-4 subsets with equal sum.
* In some constructed examples, numbers are chosen so that:

	* All subsets are either **too big** or **too small** to hit the target sum.

This problem:

* Often requires **backtracking** / exhaustive search.
* Likely needs a smart **parallel search strategy** to explore the space faster.
* For 30–50 numbers, this can get very hard.

A good testing strategy:

* Use inputs where you **know** there is an answer.

	* Example: `1, 31, 2, 30, 3, 29, 4, 28, 5, 27, 6, 26, ...`
	* Variants of $1, 2, \dots, n$ with pairings that obviously sum nicely.
* Then test with “hard” random or adversarial sets.
