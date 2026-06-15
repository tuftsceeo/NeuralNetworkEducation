# Training
## Labeled Datasets
While there are many ways to train a neural network, we will train ours by using supervised learning. This means we will give it a labled data set: a list of inputs, and the correct corresponding outputs. 

So, with our wall stopping example, say we are using the equation,
```python
speed = 1/2 * reflection
```
Write a 5 point labled dataset that we could feed to our model, then look at the example below.

<details>
<summary>Continue</summary>
For example, we could train our model on a larger version of the following dataset:
| Reflection | Speed    |
| -----------| -------- |
| 0 | 0 |
| 10 | 5 |
| 20 | 10 |
| 30 | 15 |
| 40 | 20 |
| 50 | 25 |
| 60 | 30 |
| 70 | 35 |
| 80 | 40 |
| 90 | 45 |
| 100 | 50 |
In this case, the reflection is the input, and the speed is the label.
</details>

## Writing the Train Function

lets start by defining our function. It should take in the model to train, as well as the labeled dataset of sample inputs and outputs to train on.
```python
def train(model, sample_inputs, sample_outputs):
```

### Formatting the data

PyTorch needs to have its data as Tensors, which are basically just matrices. Here, they bind the input values with the labeled output values.

```python
dataset = TensorDataset(sample_inputs, sample_outputs)
```

It then needs to turn that dataset into a DataLoader, which just wraps it in an iterable. In other words, it just makes it easier for the program to access the datapoints.

```python
loader = DataLoader(dataset)
```
### The Training Loop
Imagine you are studying for a test. How would you use a practice test to study for the exam?

First, you might try one problem. Then, you'll check your work, and then you'll tweak your understanding of the concept based on how you were wrong. Then, you'll move onto the next problem and repeat.

Think about how we can apply this to training a neural network, then continue below.

<details>
<summary>Continue</summary>

The training loop for the neural network is just like this, but instead doing problems from a practice test, it calculates an outputs from inputs in the labled dataset, and compares it's outputs to the labeled outputs. And, instead of taking the test (going through the dataset) once, it will take that test over and over again, improving it's score each time. 

Now, lets define some tools to help with this process.

First, we need a function to tell the model how much it was wrong by. We can use criterion as a shorthand for CrossEntropyLoss, which is one method to calculate loss.
```python
 criterion = nn.CrossEntropyLoss()
``` 

Next, we need a way to know how to change our network to reduce that loss value.
```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) 
``` 
The optimizer uses an something called Adam to figure out how it needs to update the values in the network. Adam does gradient descent, which is a way of trying to find the minimum of a function, in this case that function is the loss function. One key thing to note is that we input the learning rate, lr, which tells the optimizer how much it should tweak the values in the network by each time it checks its work.

Try writing a function to train the model that does the following:
Sets up the dataset
Sets up the training tools.
Goes through the data set 10 times (or epochs), and in each time:
    Go through each datapoint. For each point:
        1) Pass it through the network. *(Hint: Remember we can pass something through using model(input))*
        2) Find the loss *(Hint: criterion is a function that can take in an estimated output and a true output)*
        3) Figure out what caused that loss *(Hint: use loss.backwards(), we will go over this later)*
        4) Adjust the model accordingly *(Hint: calling optimizer.step() will adjust the weights and biases as needed)*
Returns the model that we trained

When you have tried that, continue below to see the solution

<details>
<summary>Continue</summary>
Based on that outline, our function should look something like this:

```python
def train(model, sample_inputs, sample_outputs):
    # Setup the dataset
    dataset = TensorDataset(sample_inputs, sample_outputs)

    loader = DataLoader(dataset)

    # Setup the training tools
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Go through the dataset 10 times
    for epoch in range(10):

        # Review each datapoint
        for input, true_output in loader:

            # Get the model's current estimate of an input
            guess = model(input)

            # Calculate the loss by comparing it against the true output
            loss = criterion(guess, true_output)

            # Figure out what contributed to that loss using loss.backwards()
            loss.backwards()

            # Change our network accordingly
            optimizer.step()
    # Return the trained model
    return model
```

This function is almost ready, we just have to cover one last thing.

</details>
</details>



