#!/usr/bin/env python3
"""Part B: Parameterization Comparison Table."""
import pandas as pd

table = [
    {
        "property": "Input dimension",
        "R15_weights_head (B4)": "128",
        "R16_Probe_L (B3)": "128",
        "B1_OneLogit": "128",
        "B2_TwoLogit": "128"
    },
    {
        "property": "Output logits",
        "R15_weights_head (B4)": "8 (4 processes x 2 logits)",
        "R16_Probe_L (B3)": "8 (4 processes x 2 logits)",
        "B1_OneLogit": "1 (interception only)",
        "B2_TwoLogit": "2 (interception only)"
    },
    {
        "property": "Parameter count",
        "R15_weights_head (B4)": "128 * 8 + 8 = 1032",
        "R16_Probe_L (B3)": "128 * 8 + 8 = 1032",
        "B1_OneLogit": "128 * 1 + 1 = 129",
        "B2_TwoLogit": "128 * 2 + 2 = 258"
    },
    {
        "property": "Probability formulation",
        "R15_weights_head (B4)": "p = sigmoid(z_on - z_off)",
        "R16_Probe_L (B3)": "p = sigmoid(z_on - z_off)",
        "B1_OneLogit": "p = sigmoid(z)",
        "B2_TwoLogit": "p = sigmoid(z_on - z_off)"
    },
    {
        "property": "Weight initialization",
        "R15_weights_head (B4)": "normal(mean=0, std=0.001)",
        "R16_Probe_L (B3)": "xavier_uniform",
        "B1_OneLogit": "normal(mean=0, std=0.001)",
        "B2_TwoLogit": "normal(mean=0, std=0.001)"
    },
    {
        "property": "Optimizer",
        "R15_weights_head (B4)": "Adadelta (lr=1.0, rho=0.9)",
        "R16_Probe_L (B3)": "Adam (lr=0.01, betas=(0.9, 0.999))",
        "B1_OneLogit": "Adam (lr=0.01)",
        "B2_TwoLogit": "Adam (lr=0.01)"
    },
    {
        "property": "Batching & update steps",
        "R15_weights_head (B4)": "Minibatch (100 basins), 70 steps",
        "R16_Probe_L (B3)": "Full-batch (671 basins), 1000 steps",
        "B1_OneLogit": "Full-batch (671 basins), 1000 steps",
        "B2_TwoLogit": "Full-batch (671 basins), 1000 steps"
    },
    {
        "property": "Weight decay",
        "R15_weights_head (B4)": "0.0",
        "R16_Probe_L (B3)": "1e-4",
        "B1_OneLogit": "0.0",
        "B2_TwoLogit": "0.0"
    },
    {
        "property": "Loss scope",
        "R15_weights_head (B4)": "4 processes joint mean BCE",
        "R16_Probe_L (B3)": "4 processes joint mean BCE",
        "B1_OneLogit": "Interception-only BCE",
        "B2_TwoLogit": "Interception-only BCE"
    },
    {
        "property": "Converged BCE (w_int)",
        "R15_weights_head (B4)": "0.60940 (BCE_const=0.61185)",
        "R16_Probe_L (B3)": "0.59833 (BCE_const=0.61185)",
        "B1_OneLogit": "0.59807 (BCE_const=0.61185)",
        "B2_TwoLogit": "0.59795 (BCE_const=0.61185)"
    },
    {
        "property": "Weight norm ||W|| at stop",
        "R15_weights_head (B4)": "0.708",
        "R16_Probe_L (B3)": "5.016",
        "B1_OneLogit": "3.725",
        "B2_TwoLogit": "5.266"
    }
]

df = pd.DataFrame(table)
df.to_csv("results/reconciliation_r16_5/part_b_parameterization_table.csv", index=False)
print("Saved part_b_parameterization_table.csv")