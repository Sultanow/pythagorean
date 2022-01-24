# The Challenge and aim of this project is to find four squares whose difference in each is again a square

This challenge is inspired by the question "[Four squares such that the difference of any two is a square?](https://math.stackexchange.com/questions/3286376/four-squares-such-that-the-difference-of-any-two-is-a-square)" on Math Stack Exchange.

I published my first trials, code and ideas on Math Stack Exchange using the post "[For integers x<y<z, why are these cases impossible for Mengoli's Six-Square Problem?](https://math.stackexchange.com/questions/4342283/for-integers-xyz-why-are-these-cases-impossible-for-mengolis-six-square-pr)" and got very helpful input for further investigation.

This directed me to search for 6-tuples `[s,t,u,t+u,t+u−s,t−s]` where each element is a square.

I implemented a Python script [pythagorean_gendata2_nofile_jit.py](https://github.com/Sultanow/pythagorean/blob/main/pythagorean_gendata2_nofile_jit.py) which systematically searched for those 6-tuples and it found a lot of instances. Unfortunatelly at a certain point it became slow and did not scale anymore.

With the help of [Arty](https://stackoverflow.com/users/941531/arty), who developed an enhaced version, I could manage to find a first match. This breakthrough origins from the question on Stack Overflow [Optimizing an algorithm with 3 inner loops for searching 6-tuples of squares using Python](https://stackoverflow.com/questions/70824573/optimizing-an-algorithm-with-3-inner-loops-for-searching-6-tuples-of-squares-usi?noredirect=1#comment125210150_70824573).

In order to make the final step, namely to identify these four squares, I implemented the Mathematica Script [pythagorean.nb](https://github.com/Sultanow/pythagorean/blob/main/pythagorean.nb), which outputs an "almost solution": `[w=40579, x=-58565, y=-65221]`.

From the Data Set [pythagorean_stu_Arty_.txt](https://github.com/Sultanow/pythagorean/blob/main/pythagorean_stu_Arty_.txt) I selected the corresponding row:

```
42228, 51060, 185472, 1783203984, 2607123600, 34399862784, 37006986384, 35223782400, 823919616
```

Hence `[s=42228^2,t=51060^2,u=185472^2]`. To calculate the last variable `[z]`, I need to use the equation `u+y^2=z^2` leading to `z=196605.294`. The last trick is to multiply all integers `[w,x,y,z]` with `100^2=10000`, which brings us closer to a solution:

```
w=405790000
x=585650000
y=652210000
z=1966052940
```
