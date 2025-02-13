## TTRL Algorithm (repeat for each question in the test set):

0. Preprocess the data provided into a format that can be used by the TTRL algorithm

[ 
    {
        "question": "integrate(1/(x**2 - x + 1), x)",
        "variants": [
            {
                "variant": "integrate(1/(x**2 - x + 1), x)",
                "reasoning": "The integral can be transformed using the substitution u = x - 1/2, which simplifies the denominator.",
                "difficulty": "easier"
            }
        ]
    },
    {
        "question": "integrate(1/(x**2 - x + 1), x)",
        "variants": [
            {
                "variant": "integrate(1/(x**2 - x + 1), x)",
                "reasoning": "The integral can be transformed using the substitution u = x - 1/2, which simplifies the denominator.",
                "difficulty": "easier"
            }
        ]
    }
]

1. Get a test question and its variants from the above file
3. Put the variants into a parquet file to be ready for the RL
-- Here tinyzero RL starts --
3. For each question, run GRPO using CustomTinyZero and the numerical_integration reward function
    a. Pass a reward signal to the RL algorithm (in reward function)
    b. Update the policy
4. Once done, use model to do test on actual question (with pass@1, pass@5 and pass@10 evaluation)
5. Roll back to the original model

## Difference between TTRL and RL training method

The training method we were using before is to generate variants and train the model on them to do integration, with the hope that lowered learning curve improves the performance of the model. TTRL finds variants for each question and then the model is trained on them specifically for that question.