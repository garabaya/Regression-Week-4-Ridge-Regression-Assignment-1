# Regression Week 4: Ridge Regression (interpretation)

In this notebook, we will run ridge regression multiple times with different L2 penalties to see which one produces the best fit. We will revisit the example of polynomial regression as a means to see the effect of L2 regularization. In particular, we will:
* Use a pre-built implementation of regression (Turi Create) to run polynomial regression
* Use matplotlib to visualize polynomial regressions
* Use a pre-built implementation of regression (Turi Create) to run polynomial regression, this time with L2 penalty
* Use matplotlib to visualize polynomial regressions under L2 regularization
* Choose best L2 penalty using cross-validation.
* Assess the final fit using test data.

We will continue to use the House data from previous notebooks.  (In the next programming assignment for this module, you will implement your own ridge regression learning algorithm using gradient descent.)

# Fire up Turi Create


```python
import turicreate
```

# Polynomial regression, revisited

We build on the material from Week 3, where we wrote the function to produce an SFrame with columns containing the powers of a given input. Copy and paste the function `polynomial_sframe` from Week 3:


```python
def polynomial_sframe(feature, degree):
    # assume that degree >= 1
    # initialize the SFrame:
    poly_sframe = turicreate.SFrame()
    # and set poly_sframe['power_1'] equal to the passed feature
    poly_sframe['power_1'] = feature
    # first check if degree > 1
    if degree > 1:
        # then loop over the remaining degrees:
        # range usually starts at 0 and stops at the endpoint-1. We want it to start at 2 and stop at degree
        for power in range(2, degree+1): 
            # first we'll give the column a name:
            name = 'power_' + str(power)
            # then assign poly_sframe[name] to the appropriate power of feature
            poly_sframe[name] = poly_sframe['power_1'].apply(lambda x: x**power)
    return poly_sframe
    
```

Let's use matplotlib to visualize what a polynomial regression looks like on the house data.


```python
import matplotlib.pyplot as plt
%matplotlib inline
```


```python
sales = turicreate.SFrame('home_data.sframe/')
```

As in Week 3, we will use the sqft_living variable. For plotting purposes (connecting the dots), you'll need to sort by the values of sqft_living. For houses with identical square footage, we break the tie by their prices.


```python
sales = sales.sort(['sqft_living','price'])
```

Let us revisit the 15th-order polynomial model using the 'sqft_living' input. Generate polynomial features up to degree 15 using `polynomial_sframe()` and fit a model with these features. When fitting the model, use an L2 penalty of `1e-5`:


```python
l2_small_penalty = 1e-5
```

Note: When we have so many features and so few data points, the solution can become highly numerically unstable, which can sometimes lead to strange unpredictable results.  Thus, rather than using no regularization, we will introduce a tiny amount of regularization (`l2_penalty=1e-5`) to make the solution numerically stable.  (In lecture, we discussed the fact that regularization can also help with numerical stability, and here we are seeing a practical example.)

With the L2 penalty specified above, fit the model and print out the learned weights.

Hint: make sure to add 'price' column to the new SFrame before calling `turicreate.linear_regression.create()`. Also, make sure Turi Create doesn't create its own validation set by using the option `validation_set=None` in this call.


```python
poly15_data = polynomial_sframe(sales['sqft_living'], 15)
my_features = poly15_data.column_names()
poly15_data['price'] = sales['price'] # add price to the data since it's the target
model15 = turicreate.linear_regression.create(poly15_data,
                                              target = 'price',
                                              features = my_features,
                                              validation_set = None,
                                              l2_penalty = l2_small_penalty)
```


<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 21613</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 1.018939     | 2662555.735422     | 245656.462162                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>



```python
plt.plot(poly15_data['power_1'],poly15_data['price'],'.',
         poly15_data['power_1'], model15.predict(poly15_data),'-')
```




    [<matplotlib.lines.Line2D at 0x7f15d05634a8>,
     <matplotlib.lines.Line2D at 0x7f15c0440780>]




![png](output_16_1.png)



```python
model15.coefficients
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">name</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">index</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">value</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">stderr</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">(intercept)</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">None</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">167924.8683106338</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">nan</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">power_1</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">None</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">103.09091982258876</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">nan</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">power_2</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">None</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.13460458520186014</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">nan</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">power_3</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">None</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">-0.00012907138088840022</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">nan</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">power_4</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">None</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5.1892899347229256e-08</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">nan</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">power_5</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">None</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">-7.771693113356376e-12</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">nan</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">power_6</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">None</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1.711447665310959e-16</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">nan</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">power_7</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">None</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4.5117800398292624e-20</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">nan</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">power_8</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">None</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">-4.788383278136755e-25</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">nan</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">power_9</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">None</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">-2.333436313252247e-28</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">nan</td>
    </tr>
</table>
[16 rows x 4 columns]<br/>Note: Only the head of the SFrame is printed.<br/>You can use print_rows(num_rows=m, num_columns=n) to print more rows and columns.
</div>



***QUIZ QUESTION:  What's the learned value for the coefficient of feature `power_1`?***
103.09091982258876

# Observe overfitting

Recall from Week 3 that the polynomial fit of degree 15 changed wildly whenever the data changed. In particular, when we split the sales data into four subsets and fit the model of degree 15, the result came out to be very different for each subset. The model had a *high variance*. We will see in a moment that ridge regression reduces such variance. But first, we must reproduce the experiment we did in Week 3.

First, split the data into split the sales data into four subsets of roughly equal size and call them `set_1`, `set_2`, `set_3`, and `set_4`. Use `.random_split` function and make sure you set `seed=0`. 


```python
(semi_split1, semi_split2) = sales.random_split(.5,seed=0)
(set_1, set_2) = semi_split1.random_split(0.5, seed=0)
(set_3, set_4) = semi_split2.random_split(0.5, seed=0)
```

Next, fit a 15th degree polynomial on `set_1`, `set_2`, `set_3`, and `set_4`, using 'sqft_living' to predict prices. Print the weights and make a plot of the resulting model.

Hint: When calling `turicreate.linear_regression.create()`, use the same L2 penalty as before (i.e. `l2_small_penalty`).  Also, make sure Turi Create doesn't create its own validation set by using the option `validation_set = None` in this call.


```python
poly15_data = polynomial_sframe(set_1['sqft_living'], 15)
my_features = poly15_data.column_names()
poly15_data['price'] = set_1['price'] # add price to the data since it's the target
model15 = turicreate.linear_regression.create(poly15_data,
                                              target = 'price',
                                              features = my_features,
                                              validation_set = None,
                                              l2_penalty = l2_small_penalty)
model15.coefficients.print_rows(num_rows=16, num_columns=4)
plt.plot(poly15_data['power_1'],poly15_data['price'],'.',
         poly15_data['power_1'], model15.predict(poly15_data),'-')
```


<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 5404</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.017008     | 2191984.900834     | 248699.117253                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>


    +-------------+-------+-------------------------+-----------------------+
    |     name    | index |          value          |         stderr        |
    +-------------+-------+-------------------------+-----------------------+
    | (intercept) |  None |    9306.460738574853    |          nan          |
    |   power_1   |  None |    585.8658227918767    |          nan          |
    |   power_2   |  None |   -0.39730589492643253  |          nan          |
    |   power_3   |  None |  0.00014147090040698705 |          nan          |
    |   power_4   |  None |  -1.529459910633883e-08 |          nan          |
    |   power_5   |  None |  -3.797562554472397e-13 |          nan          |
    |   power_6   |  None |  5.974816422113205e-17  |          nan          |
    |   power_7   |  None |  1.068885079424125e-20  | 9.827796414259983e-17 |
    |   power_8   |  None |  1.5934406685250742e-25 |          nan          |
    |   power_9   |  None |  -6.928348684589336e-29 |          nan          |
    |   power_10  |  None |  -6.83813476071358e-33  |          nan          |
    |   power_11  |  None |  -1.626860809887929e-37 | 7.525307345296993e-33 |
    |   power_12  |  None |  2.851186198304115e-41  |          nan          |
    |   power_13  |  None |  3.799982392452631e-45  |          nan          |
    |   power_14  |  None |  1.5265261843349395e-49 |          nan          |
    |   power_15  |  None | -2.3380732075473816e-53 |          nan          |
    +-------------+-------+-------------------------+-----------------------+
    [16 rows x 4 columns]
    





    [<matplotlib.lines.Line2D at 0x7f15c04006d8>,
     <matplotlib.lines.Line2D at 0x7f15c01edb00>]




![png](output_24_17.png)



```python
poly15_data = polynomial_sframe(set_2['sqft_living'], 15)
my_features = poly15_data.column_names()
poly15_data['price'] = set_2['price'] # add price to the data since it's the target
model15 = turicreate.linear_regression.create(poly15_data,
                                              target = 'price',
                                              features = my_features,
                                              validation_set = None,
                                              l2_penalty = l2_small_penalty)
model15.coefficients.print_rows(num_rows=16, num_columns=4)
plt.plot(poly15_data['power_1'],poly15_data['price'],'.',
         poly15_data['power_1'], model15.predict(poly15_data),'-')
```


<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 5398</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.022818     | 1975178.190550     | 234533.610645                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>


    +-------------+-------+-------------------------+----------------------+
    |     name    | index |          value          |        stderr        |
    +-------------+-------+-------------------------+----------------------+
    | (intercept) |  None |   -25115.912055661436   |  1066759.1384981384  |
    |   power_1   |  None |    783.4938272804999    |  5154.635347138643   |
    |   power_2   |  None |   -0.7677593406452785   |  9.753020273656633   |
    |   power_3   |  None |   0.000438766397164288  | 0.007976352435171218 |
    |   power_4   |  None | -1.1516917938560663e-07 |         nan          |
    |   power_5   |  None |   6.84281733897351e-12  |         nan          |
    |   power_6   |  None |  2.5119510955255233e-15 |         nan          |
    |   power_7   |  None | -2.0644049916397132e-19 |         nan          |
    |   power_8   |  None |  -4.596731075779906e-23 |         nan          |
    |   power_9   |  None |  -2.712776424995684e-29 |         nan          |
    |   power_10  |  None |   6.21818426987022e-31  |         nan          |
    |   power_11  |  None |  6.517414176316569e-35  |         nan          |
    |   power_12  |  None |  -9.41315223930008e-40  |         nan          |
    |   power_13  |  None | -1.0242139283793461e-42 |         nan          |
    |   power_14  |  None | -1.0039107807384084e-46 |         nan          |
    |   power_15  |  None |  1.3011336139980334e-50 |         nan          |
    +-------------+-------+-------------------------+----------------------+
    [16 rows x 4 columns]
    





    [<matplotlib.lines.Line2D at 0x7f15c013b198>,
     <matplotlib.lines.Line2D at 0x7f15c0168860>]




![png](output_25_17.png)



```python
poly15_data = polynomial_sframe(set_3['sqft_living'], 15)
my_features = poly15_data.column_names()
poly15_data['price'] = set_3['price'] # add price to the data since it's the target
model15 = turicreate.linear_regression.create(poly15_data,
                                              target = 'price',
                                              features = my_features,
                                              validation_set = None,
                                              l2_penalty = l2_small_penalty)
model15.coefficients.print_rows(num_rows=16, num_columns=4)
plt.plot(poly15_data['power_1'],poly15_data['price'],'.',
         poly15_data['power_1'], model15.predict(poly15_data),'-')
```


<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 5409</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.017799     | 2283722.685233     | 251097.728054                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>


    +-------------+-------+-------------------------+-----------------------+
    |     name    | index |          value          |         stderr        |
    +-------------+-------+-------------------------+-----------------------+
    | (intercept) |  None |    462426.5601171815    |   1310349.080391138   |
    |   power_1   |  None |    -759.251823483348    |   6357.396009618166   |
    |   power_2   |  None |    1.028670022953628    |   12.871561488676209  |
    |   power_3   |  None |  -0.0005282645136443045 |  0.014300230535834086 |
    |   power_4   |  None |  1.1542290544321842e-07 |  9.58250019181832e-06 |
    |   power_5   |  None |  -2.260959869224837e-12 | 3.920480702158251e-09 |
    |   power_6   |  None | -2.0821425559670597e-15 |  8.31747402726747e-13 |
    |   power_7   |  None |  4.087698878766004e-20  |          nan          |
    |   power_8   |  None |  2.5707916122050266e-23 |          nan          |
    |   power_9   |  None |  1.2431131050849085e-27 |          nan          |
    |   power_10  |  None | -1.7202591411366398e-31 |  6.21691609844703e-28 |
    |   power_11  |  None | -2.9676099903981554e-35 | 8.900669040558528e-32 |
    |   power_12  |  None | -1.0657497369273635e-39 | 9.023794547733603e-36 |
    |   power_13  |  None |  2.426357157130831e-43  | 7.848200421077859e-40 |
    |   power_14  |  None |  3.5559870251379337e-47 | 3.862234528305765e-44 |
    |   power_15  |  None |  -2.857774518124205e-51 | 7.744209502999505e-49 |
    +-------------+-------+-------------------------+-----------------------+
    [16 rows x 4 columns]
    





    [<matplotlib.lines.Line2D at 0x7f15c0119cc0>,
     <matplotlib.lines.Line2D at 0x7f15c00e90b8>]




![png](output_26_17.png)



```python
poly15_data = polynomial_sframe(set_4['sqft_living'], 15)
my_features = poly15_data.column_names()
poly15_data['price'] = set_4['price'] # add price to the data since it's the target
model15 = turicreate.linear_regression.create(poly15_data,
                                              target = 'price',
                                              features = my_features,
                                              validation_set = None,
                                              l2_penalty = l2_small_penalty)
model15.coefficients.print_rows(num_rows=16, num_columns=4)
plt.plot(poly15_data['power_1'],poly15_data['price'],'.',
         poly15_data['power_1'], model15.predict(poly15_data),'-')
```


<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 5402</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.013050     | 2378292.373331     | 244341.293208                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>


    +-------------+-------+-------------------------+------------------------+
    |     name    | index |          value          |         stderr         |
    +-------------+-------+-------------------------+------------------------+
    | (intercept) |  None |   -170240.04166485136   |   1554246.1261089714   |
    |   power_1   |  None |    1247.5903831275245   |   10197.07335608088    |
    |   power_2   |  None |    -1.224609183161148   |   27.919991642491652   |
    |   power_3   |  None |  0.0005552546758525209  |  0.04229553716799707   |
    |   power_4   |  None |  -6.38262585527143e-08  | 3.9580832892859916e-05 |
    |   power_5   |  None | -2.2021594562925072e-11 | 2.4003566731442154e-08 |
    |   power_6   |  None |  4.818346610997274e-15  | 9.436056395736535e-12  |
    |   power_7   |  None |  4.2146158179457224e-19 | 2.1404750695939616e-15 |
    |   power_8   |  None |  -7.998806727069852e-23 |          nan           |
    |   power_9   |  None | -1.3236591161465522e-26 |          nan           |
    |   power_10  |  None |  1.6019790435757102e-31 |          nan           |
    |   power_11  |  None |  2.399043553347965e-34  |          nan           |
    |   power_12  |  None |  2.3335443247666626e-38 | 9.874936494018852e-35  |
    |   power_13  |  None | -1.7987404588565005e-42 | 1.2390001750004404e-38 |
    |   power_14  |  None | -6.0286258702437944e-46 | 6.0927404707810355e-43 |
    |   power_15  |  None |  4.394726119172111e-50  | 1.2983652058413109e-47 |
    +-------------+-------+-------------------------+------------------------+
    [16 rows x 4 columns]
    





    [<matplotlib.lines.Line2D at 0x7f15c0080e10>,
     <matplotlib.lines.Line2D at 0x7f15c00329b0>]




![png](output_27_17.png)


The four curves should differ from one another a lot, as should the coefficients you learned.

***QUIZ QUESTION:  For the models learned in each of these training sets, what are the smallest and largest values you learned for the coefficient of feature `power_1`?***  (For the purpose of answering this question, negative numbers are considered "smaller" than positive numbers. So -5 is smaller than -3, and -3 is smaller than 5 and so forth.)

set_3: -759.251823483348

set_4: 1247.5903831275245

# Ridge regression comes to rescue

Generally, whenever we see weights change so much in response to change in data, we believe the variance of our estimate to be large. Ridge regression aims to address this issue by penalizing "large" weights. (Weights of `model15` looked quite small, but they are not that small because 'sqft_living' input is in the order of thousands.)

With the argument `l2_penalty=1e5`, fit a 15th-order polynomial model on `set_1`, `set_2`, `set_3`, and `set_4`. Other than the change in the `l2_penalty` parameter, the code should be the same as the experiment above. Also, make sure Turi Create doesn't create its own validation set by using the option `validation_set = None` in this call.


```python
l2_penalty=1e5
```


```python
poly15_data = polynomial_sframe(set_1['sqft_living'], 15)
my_features = poly15_data.column_names()
poly15_data['price'] = set_1['price'] # add price to the data since it's the target
model15 = turicreate.linear_regression.create(poly15_data,
                                              target = 'price',
                                              features = my_features,
                                              validation_set = None,
                                              l2_penalty = l2_penalty)
model15.coefficients.print_rows(num_rows=16, num_columns=4)
plt.plot(poly15_data['power_1'],poly15_data['price'],'.',
         poly15_data['power_1'], model15.predict(poly15_data),'-')
```


<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 5404</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.011180     | 5978778.434729     | 374261.720860                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>


    +-------------+-------+------------------------+------------------------+
    |     name    | index |         value          |         stderr         |
    +-------------+-------+------------------------+------------------------+
    | (intercept) |  None |   530317.0245158835    |          nan           |
    |   power_1   |  None |   2.5873887567286866   |          nan           |
    |   power_2   |  None | 0.0012741440059211371  |          nan           |
    |   power_3   |  None | 1.7493422693158899e-07 |          nan           |
    |   power_4   |  None | 1.0602211909664251e-11 |          nan           |
    |   power_5   |  None | 5.422476044821804e-16  |          nan           |
    |   power_6   |  None | 2.895638283427737e-20  |          nan           |
    |   power_7   |  None | 1.6500066635095529e-24 | 1.4789630292567112e-16 |
    |   power_8   |  None | 9.860815284092932e-29  |          nan           |
    |   power_9   |  None |  6.06589348254357e-33  |          nan           |
    |   power_10  |  None | 3.789178688696588e-37  |          nan           |
    |   power_11  |  None | 2.3822312131219896e-41 | 1.1324666159485423e-32 |
    |   power_12  |  None | 1.4984796921456947e-45 |          nan           |
    |   power_13  |  None | 9.391611902848278e-50  |          nan           |
    |   power_14  |  None |  5.84523161980618e-54  |          nan           |
    |   power_15  |  None | 3.601202072029721e-58  |          nan           |
    +-------------+-------+------------------------+------------------------+
    [16 rows x 4 columns]
    





    [<matplotlib.lines.Line2D at 0x7f15c00656d8>,
     <matplotlib.lines.Line2D at 0x7f159c5339e8>]




![png](output_32_17.png)



```python
poly15_data = polynomial_sframe(set_2['sqft_living'], 15)
my_features = poly15_data.column_names()
poly15_data['price'] = set_2['price'] # add price to the data since it's the target
model15 = turicreate.linear_regression.create(poly15_data,
                                              target = 'price',
                                              features = my_features,
                                              validation_set = None,
                                              l2_penalty = l2_penalty)
model15.coefficients.print_rows(num_rows=16, num_columns=4)
plt.plot(poly15_data['power_1'],poly15_data['price'],'.',
         poly15_data['power_1'], model15.predict(poly15_data),'-')
```


<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 5398</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.010992     | 2984894.541944     | 323238.809634                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>


    +-------------+-------+------------------------+----------------------+
    |     name    | index |         value          |        stderr        |
    +-------------+-------+------------------------+----------------------+
    | (intercept) |  None |   519216.89738342643   |  1470228.3103281604  |
    |   power_1   |  None |   2.044704741819378    |   7104.21925932691   |
    |   power_2   |  None | 0.0011314362683958127  |  13.441803308778976  |
    |   power_3   |  None | 2.930742775489716e-07  | 0.010993164942419803 |
    |   power_4   |  None | 4.4354059845325974e-11 |         nan          |
    |   power_5   |  None | 4.808491122043446e-15  |         nan          |
    |   power_6   |  None | 4.530917078263864e-19  |         nan          |
    |   power_7   |  None | 4.1604291057458376e-23 |         nan          |
    |   power_8   |  None | 3.900946351283382e-27  |         nan          |
    |   power_9   |  None | 3.7773187602026064e-31 |         nan          |
    |   power_10  |  None | 3.766503268417181e-35  |         nan          |
    |   power_11  |  None | 3.8422809475395966e-39 |         nan          |
    |   power_12  |  None | 3.985208284143722e-43  |         nan          |
    |   power_13  |  None | 4.1827276239367343e-47 |         nan          |
    |   power_14  |  None | 4.427383328777781e-51  |         nan          |
    |   power_15  |  None | 4.715182454121554e-55  |         nan          |
    +-------------+-------+------------------------+----------------------+
    [16 rows x 4 columns]
    





    [<matplotlib.lines.Line2D at 0x7f159c4ef518>,
     <matplotlib.lines.Line2D at 0x7f159c4a7438>]




![png](output_33_17.png)



```python
poly15_data = polynomial_sframe(set_3['sqft_living'], 15)
my_features = poly15_data.column_names()
poly15_data['price'] = set_3['price'] # add price to the data since it's the target
model15 = turicreate.linear_regression.create(poly15_data,
                                              target = 'price',
                                              features = my_features,
                                              validation_set = None,
                                              l2_penalty = l2_penalty)
model15.coefficients.print_rows(num_rows=16, num_columns=4)
plt.plot(poly15_data['power_1'],poly15_data['price'],'.',
         poly15_data['power_1'], model15.predict(poly15_data),'-')
```


<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 5409</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.016658     | 3695342.767093     | 350033.521294                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>


    +-------------+-------+------------------------+------------------------+
    |     name    | index |         value          |         stderr         |
    +-------------+-------+------------------------+------------------------+
    | (intercept) |  None |   522911.5180475718    |   1826643.7784532942   |
    |   power_1   |  None |   2.2689042187657877   |   8862.293294139941    |
    |   power_2   |  None | 0.0012590504184157225  |    17.9431252817388    |
    |   power_3   |  None | 2.7755291815451765e-07 |  0.019934708643388317  |
    |   power_4   |  None | 3.209330977903899e-11  | 1.3358130760230367e-05 |
    |   power_5   |  None |  2.87573572364483e-15  | 5.465201441592894e-09  |
    |   power_6   |  None | 2.5007611267119213e-19 | 1.1594667720009293e-12 |
    |   power_7   |  None | 2.2468526590627848e-23 |          nan           |
    |   power_8   |  None | 2.0934998313470215e-27 |          nan           |
    |   power_9   |  None | 2.0043538329631962e-31 |          nan           |
    |   power_10  |  None | 1.9541080024851158e-35 | 8.666462458236405e-28  |
    |   power_11  |  None | 1.9273411945583566e-39 | 1.2407649206084062e-31 |
    |   power_12  |  None |  1.91483699012907e-43  | 1.257928777554307e-35  |
    |   power_13  |  None |  1.91102277046499e-47  | 1.0940494167353595e-39 |
    |   power_14  |  None | 1.912462423017048e-51  | 5.384005512448125e-44  |
    |   power_15  |  None | 1.9169955803503674e-55 | 1.0795529465682826e-48 |
    +-------------+-------+------------------------+------------------------+
    [16 rows x 4 columns]
    





    [<matplotlib.lines.Line2D at 0x7f159c45aeb8>,
     <matplotlib.lines.Line2D at 0x7f159c402c88>]




![png](output_34_17.png)



```python
poly15_data = polynomial_sframe(set_4['sqft_living'], 15)
my_features = poly15_data.column_names()
poly15_data['price'] = set_4['price'] # add price to the data since it's the target
model15 = turicreate.linear_regression.create(poly15_data,
                                              target = 'price',
                                              features = my_features,
                                              validation_set = None,
                                              l2_penalty = l2_penalty)
model15.coefficients.print_rows(num_rows=16, num_columns=4)
plt.plot(poly15_data['power_1'],poly15_data['price'],'.',
         poly15_data['power_1'], model15.predict(poly15_data),'-')
```


<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 5402</pre>



<pre>Number of features          : 15</pre>



<pre>Number of unpacked features : 15</pre>



<pre>Number of coefficients    : 16</pre>



<pre>Starting Newton Method</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| Iteration | Passes   | Elapsed Time | Training Max Error | Training Root-Mean-Square Error |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>| 1         | 2        | 0.011929     | 3601895.280124     | 323111.582889                   |</pre>



<pre>+-----------+----------+--------------+--------------------+---------------------------------+</pre>



<pre>SUCCESS: Optimal solution found.</pre>



<pre></pre>


    +-------------+-------+------------------------+------------------------+
    |     name    | index |         value          |         stderr         |
    +-------------+-------+------------------------+------------------------+
    | (intercept) |  None |   513667.0870874073    |   2055301.088950528    |
    |   power_1   |  None |   1.9104093824432022   |   13484.38681673255    |
    |   power_2   |  None | 0.0011005802917477242  |   36.92078639434192    |
    |   power_3   |  None | 3.127539878788059e-07  |  0.05593069343319507   |
    |   power_4   |  None | 5.500678868246386e-11  | 5.2340827864839856e-05 |
    |   power_5   |  None |  7.20467557824708e-15  | 3.1741791736253475e-08 |
    |   power_6   |  None | 8.249772493837897e-19  | 1.2478034630273316e-11 |
    |   power_7   |  None | 9.065032234977414e-23  | 2.830517424174993e-15  |
    |   power_8   |  None | 9.956831604526312e-27  |          nan           |
    |   power_9   |  None | 1.1083812798160367e-30 |          nan           |
    |   power_10  |  None | 1.2531522414327033e-34 |          nan           |
    |   power_11  |  None | 1.4360078140197673e-38 |          nan           |
    |   power_12  |  None | 1.6626996780013466e-42 | 1.3058400074822684e-34 |
    |   power_13  |  None | 1.9398172452969622e-46 | 1.638426736995295e-38  |
    |   power_14  |  None |  2.27541485770272e-50  | 8.056906762662404e-43  |
    |   power_15  |  None | 2.679487848971385e-54  | 1.7169297555862313e-47 |
    +-------------+-------+------------------------+------------------------+
    [16 rows x 4 columns]
    





    [<matplotlib.lines.Line2D at 0x7f159c3c6048>,
     <matplotlib.lines.Line2D at 0x7f159c3f45c0>]




![png](output_35_17.png)


These curves should vary a lot less, now that you applied a high degree of regularization.

***QUIZ QUESTION:  For the models learned with the high level of regularization in each of these training sets, what are the smallest and largest values you learned for the coefficient of feature `power_1`?*** (For the purpose of answering this question, negative numbers are considered "smaller" than positive numbers. So -5 is smaller than -3, and -3 is smaller than 5 and so forth.)

set_4: 1.9104093824432022

set_1: 2.5873887567286866

# Selecting an L2 penalty via cross-validation

Just like the polynomial degree, the L2 penalty is a "magic" parameter we need to select. We could use the validation set approach as we did in the last module, but that approach has a major disadvantage: it leaves fewer observations available for training. **Cross-validation** seeks to overcome this issue by using all of the training set in a smart way.

We will implement a kind of cross-validation called **k-fold cross-validation**. The method gets its name because it involves dividing the training set into k segments of roughtly equal size. Similar to the validation set method, we measure the validation error with one of the segments designated as the validation set. The major difference is that we repeat the process k times as follows:

Set aside segment 0 as the validation set, and fit a model on rest of data, and evalutate it on this validation set<br>
Set aside segment 1 as the validation set, and fit a model on rest of data, and evalutate it on this validation set<br>
...<br>
Set aside segment k-1 as the validation set, and fit a model on rest of data, and evalutate it on this validation set

After this process, we compute the average of the k validation errors, and use it as an estimate of the generalization error. Notice that  all observations are used for both training and validation, as we iterate over segments of data. 

To estimate the generalization error well, it is crucial to shuffle the training data before dividing them into segments. The package turicreate_cross_validation (see below) has a utility function for shuffling a given SFrame. We reserve 10% of the data as the test set and shuffle the remainder. (Make sure to use `seed=1` to get consistent answer.)

  
_Note:_ For applying cross-validation, we will import a package called `turicreate_cross_validation`. To install it, please run this command on your terminal:

`pip install -e git+https://github.com/Kagandi/turicreate-cross-validation.git#egg=turicreate_cross_validation`

You can find the documentation on this package here: https://github.com/Kagandi/turicreate-cross-validation


```python
import turicreate_cross_validation.cross_validation as tcv

(train_valid, test) = sales.random_split(.9, seed=1)
train_valid_shuffled = tcv.shuffle_sframe(train_valid, random_seed=1)
```

Once the data is shuffled, we divide it into equal segments. Each segment should receive `n/k` elements, where `n` is the number of observations in the training set and `k` is the number of segments. Since the segment 0 starts at index 0 and contains `n/k` elements, it ends at index `(n/k)-1`. The segment 1 starts where the segment 0 left off, at index `(n/k)`. With `n/k` elements, the segment 1 ends at index `(n*2/k)-1`. Continuing in this fashion, we deduce that the segment `i` starts at index `(n*i/k)` and ends at `(n*(i+1)/k)-1`.

With this pattern in mind, we write a short loop that prints the starting and ending indices of each segment, just to make sure you are getting the splits right.


```python
n = len(train_valid_shuffled)
k = 10 # 10-fold cross-validation

for i in range(k):
    start = (n*i)/k
    end = (n*(i+1))/k-1
    print(i, (start, end))
```

    0 (0.0, 1938.6)
    1 (1939.6, 3878.2)
    2 (3879.2, 5817.8)
    3 (5818.8, 7757.4)
    4 (7758.4, 9697.0)
    5 (9698.0, 11636.6)
    6 (11637.6, 13576.2)
    7 (13577.2, 15515.8)
    8 (15516.8, 17455.4)
    9 (17456.4, 19395.0)


Let us familiarize ourselves with array slicing with SFrame. To extract a continuous slice from an SFrame, use colon in square brackets. For instance, the following cell extracts rows 0 to 9 of `train_valid_shuffled`. Notice that the first index (0) is included in the slice but the last index (10) is omitted.


```python
train_valid_shuffled[0:10] # rows 0 to 9
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">id</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">date</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">price</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">bedrooms</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">bathrooms</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">sqft_living</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">sqft_lot</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">floors</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">waterfront</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">8645511350</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2014-12-01 00:00:00+00:00</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">300000.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1.75</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1810.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">21138.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">7237501370</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2014-07-17 00:00:00+00:00</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1079000.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3.25</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4800.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">12727.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">7278700100</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2015-01-21 00:00:00+00:00</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">625000.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2.5</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2740.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">9599.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1421079007</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2015-03-24 00:00:00+00:00</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">408506.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2.75</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2480.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">209199.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1.5</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4338800370</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2014-11-17 00:00:00+00:00</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">220000.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1000.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">6020.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">7511200020</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2014-08-29 00:00:00+00:00</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">509900.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1.75</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1690.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">53578.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3300701615</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2014-09-30 00:00:00+00:00</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">655000.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2.5</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2630.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4000.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">7011200260</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2014-12-19 00:00:00+00:00</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">485000.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1400.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3600.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3570000130</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2014-06-11 00:00:00+00:00</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">580379.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2.75</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2240.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">27820.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1.5</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2796100640</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2015-04-24 00:00:00+00:00</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">264900.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2.5</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2040.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">7000.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
    </tr>
</table>
<table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">view</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">condition</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">grade</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">sqft_above</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">sqft_basement</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">yr_built</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">yr_renovated</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">zipcode</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">lat</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">7.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1240.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">570.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1977.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">98058</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">47.46736904</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">10.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4800.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2011.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">98059</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">47.53108576</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">8.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1820.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">920.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1961.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">98177</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">47.77279701</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">8.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1870.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">610.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2000.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">98010</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">47.30847072</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">6.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1000.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1944.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">98166</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">47.47933643</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">8.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1690.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1984.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">98053</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">47.6545751</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">8.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2630.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2002.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">98117</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">47.69151411</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">7.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1100.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">300.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1900.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">98119</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">47.63846783</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">8.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2240.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1976.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">98075</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">47.59357299</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">7.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1250.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">790.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1979.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">98031</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">47.40555074</td>
    </tr>
</table>
<table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">long</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">sqft_living15</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">sqft_lot15</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">-122.17768631</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1850.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">12200.0</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">-122.13389261</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4750.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">13602.0</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">-122.38485302</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2660.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">8280.0</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">-121.88816296</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2040.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">219229.0</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">-122.34575463</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1300.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">8640.0</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">-122.04899568</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2290.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">52707.0</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">-122.38139901</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1640.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4000.0</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">-122.36993806</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1630.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2048.0</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">-122.05362447</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2330.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">20000.0</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">-122.17648783</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1900.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">7378.0</td>
    </tr>
</table>
[10 rows x 21 columns]<br/>
</div>



Now let us extract individual segments with array slicing. Consider the scenario where we group the houses in the `train_valid_shuffled` dataframe into k=10 segments of roughly equal size, with starting and ending indices computed as above.
Extract the fourth segment (segment 3) and assign it to a variable called `validation4`.


```python
validation4 = train_valid_shuffled[5818.8:7757.4]
```

To verify that we have the right elements extracted, run the following cell, which computes the average price of the fourth segment. When rounded to nearest whole number, the average should be $559,642.


```python
print(int(round(validation4['price'].mean(), 0)))
```

    559642


After designating one of the k segments as the validation set, we train a model using the rest of the data. To choose the remainder, we slice (0:start) and (end+1:n) of the data and paste them together. SFrame has `append()` method that pastes together two disjoint sets of rows originating from a common dataset. For instance, the following cell pastes together the first and last two rows of the `train_valid_shuffled` dataframe.


```python
n = len(train_valid_shuffled)
first_two = train_valid_shuffled[0:2]
last_two = train_valid_shuffled[n-2:n]
print(first_two.append(last_two))
```

    +------------+---------------------------+-----------+----------+-----------+
    |     id     |            date           |   price   | bedrooms | bathrooms |
    +------------+---------------------------+-----------+----------+-----------+
    | 8645511350 | 2014-12-01 00:00:00+00:00 |  300000.0 |   3.0    |    1.75   |
    | 7237501370 | 2014-07-17 00:00:00+00:00 | 1079000.0 |   4.0    |    3.25   |
    | 4077800582 | 2014-09-12 00:00:00+00:00 |  522000.0 |   3.0    |    1.0    |
    | 7853370620 | 2015-02-06 00:00:00+00:00 |  605000.0 |   5.0    |    4.0    |
    +------------+---------------------------+-----------+----------+-----------+
    +-------------+----------+--------+------------+------+-----------+-------+
    | sqft_living | sqft_lot | floors | waterfront | view | condition | grade |
    +-------------+----------+--------+------------+------+-----------+-------+
    |    1810.0   | 21138.0  |  1.0   |     0      |  0   |     4     |  7.0  |
    |    4800.0   | 12727.0  |  2.0   |     0      |  0   |     3     |  10.0 |
    |    1150.0   |  7080.0  |  1.0   |     0      |  0   |     3     |  7.0  |
    |    3040.0   |  6000.0  |  2.0   |     0      |  0   |     3     |  8.0  |
    +-------------+----------+--------+------------+------+-----------+-------+
    +------------+---------------+----------+--------------+---------+-------------+
    | sqft_above | sqft_basement | yr_built | yr_renovated | zipcode |     lat     |
    +------------+---------------+----------+--------------+---------+-------------+
    |   1240.0   |     570.0     |  1977.0  |     0.0      |  98058  | 47.46736904 |
    |   4800.0   |      0.0      |  2011.0  |     0.0      |  98059  | 47.53108576 |
    |   1150.0   |      0.0      |  1952.0  |     0.0      |  98125  | 47.71063854 |
    |   2280.0   |     760.0     |  2011.0  |     0.0      |  98065  | 47.51887717 |
    +------------+---------------+----------+--------------+---------+-------------+
    +---------------+---------------+-----+
    |      long     | sqft_living15 | ... |
    +---------------+---------------+-----+
    | -122.17768631 |     1850.0    | ... |
    | -122.13389261 |     4750.0    | ... |
    | -122.28837299 |     1490.0    | ... |
    | -121.87558112 |     3070.0    | ... |
    +---------------+---------------+-----+
    [4 rows x 21 columns]
    


Extract the remainder of the data after *excluding* fourth segment (segment 3) and assign the subset to `train4`.


```python
train4_1 = train_valid_shuffled[0:5818.8]
train4_2 = train_valid_shuffled[7757.4+1:n]
train4 = train4_1.append(train4_2)
```

To verify that we have the right elements extracted, run the following cell, which computes the average price of the data with fourth segment excluded. When rounded to nearest whole number, the average should be $536,865.


```python
print(int(round(train4['price'].mean(), 0)))
```

    536866


Now we are ready to implement k-fold cross-validation. Write a function that computes k validation errors by designating each of the k segments as the validation set. It accepts as parameters (i) `k`, (ii) `l2_penalty`, (iii) dataframe, (iv) name of output column (e.g. `price`) and (v) list of feature names. The function returns the average validation error using k segments as validation sets.

* For each i in [0, 1, ..., k-1]:
  * Compute starting and ending indices of segment i and call 'start' and 'end'
  * Form validation set by taking a slice (start:end+1) from the data.
  * Form training set by appending slice (end+1:n) to the end of slice (0:start).
  * Train a linear model using training set just formed, with a given l2_penalty
  * Compute validation error using validation set just formed


```python
import numpy as np
def k_fold_cross_validation(k, l2_penalty, data, output_name, features_list):
    val_errors = []
    n = len(data)
    for i in range(k):
        start = (n*i)/k
        end = (n*(i+1))/k-1
        validation = data[start:end]
        train_1 = data[0:start]
        train_2 = data[end+1:n]
        train = train_1.append(train_2)
        model = turicreate.linear_regression.create(data,
                                              target = output_name,
                                              features = features_list,
                                              validation_set = None,
                                              l2_penalty = l2_penalty,
                                              verbose = False)
        # First get the predictions
        predicted = model.predict(validation);
        # Then compute the residuals/errors
        errors = validation[output_name]-predicted;
        # Then square and add them up    
        val_errors.append((errors**2).sum());
    return np.mean(val_errors)  
```

Once we have a function to compute the average validation error for a model, we can write a loop to find the model that minimizes the average validation error. Write a loop that does the following:
* We will again be aiming to fit a 15th-order polynomial model using the `sqft_living` input
* For `l2_penalty` in [10^1, 10^1.5, 10^2, 10^2.5, ..., 10^7] (to get this in Python, you can use this Numpy function: `np.logspace(1, 7, num=13)`.)
    * Run 10-fold cross-validation with `l2_penalty`
* Report which L2 penalty produced the lowest average validation error.

Note: since the degree of the polynomial is now fixed to 15, to make things faster, you should generate polynomial features in advance and re-use them throughout the loop. Make sure to use `train_valid_shuffled` when generating polynomial features!


```python
k = 10
val = []
l2_penalties = np.logspace(1, 7, num=13)
output_name = 'price'
feature = 'sqft_living'
order = 15
poly_data = polynomial_sframe(train_valid_shuffled[feature], order)
poly_data[output_name] = train_valid_shuffled[output_name]
for l2_penalty in l2_penalties:
    val.append(k_fold_cross_validation(k = k,
                                       l2_penalty = l2_penalty,
                                       data = poly_data,
                                       output_name = output_name,
                                       features_list = ['power_1']))
print(val)
min_l2_penalty = l2_penalties[val.index(min(val))]
print('min: ' + str(min(val)) + ' corresponding to l2_penalty: ' + str(min_l2_penalty))
```

    [134815190674744.03, 134826762544992.8, 134936513681533.4, 135890041250087.4, 142333491208169.8, 167452241996168.44, 210349227220562.44, 243113604033400.66, 258014932599918.0, 263371413280573.8, 265138128087065.66, 265704396689880.6, 265884234809859.75]
    min: 134815190674744.03 corresponding to l2_penalty: 10.0


***QUIZ QUESTIONS:  What is the best value for the L2 penalty according to 10-fold validation?***

You may find it useful to plot the k-fold cross-validation errors you have obtained to better understand the behavior of the method.  


```python
# Plot the l2_penalty values in the x axis and the cross-validation error in the y axis.
# Using plt.xscale('log') will make your plot more intuitive.
plt.xscale('log')
plt.plot(l2_penalties,val,'-')
```




    [<matplotlib.lines.Line2D at 0x7f159c3ae710>]




![png](output_61_1.png)


Once you found the best value for the L2 penalty using cross-validation, it is important to retrain a final model on all of the training data using this value of `l2_penalty`. This way, your final model will be trained on the entire dataset.


```python
model = turicreate.linear_regression.create(train_valid,
                                              target = 'price',
                                              features = ['sqft_living'],
                                              validation_set = None,
                                              l2_penalty = min_l2_penalty,
                                              verbose = False)
```

***QUIZ QUESTION: Using the best L2 penalty found above, train a model using all training data. What is the RSS on the TEST data of the model you learn with this L2 penalty? ***


```python
test = test.sort(['sqft_living','price'])
# First get the predictions
predicted = model.predict(test)
# Then compute the residuals/errors
errors = test['price']-predicted
# Then square and add them up    
(errors**2).sum()
```




    129028453845344.28




```python
plt.plot(test['sqft_living'],test['price'],'.',
         test['sqft_living'], predicted,'-')
```




    [<matplotlib.lines.Line2D at 0x7f159c1f4240>,
     <matplotlib.lines.Line2D at 0x7f159c1f4320>]




![png](output_66_1.png)

