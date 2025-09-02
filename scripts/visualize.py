import matplotlib.pyplot as plt
import seaborn as sns

# Data extraction (using the format you provided)
data = [
    {'loss': 0.1755, 'grad_norm': 1.0059195756912231, 'learning_rate': 1.9133996489174957e-05, 'epoch': 0.63},
    {'loss': 0.1097, 'grad_norm': 0.1984349489212036, 'learning_rate': 1.8987712112346402e-05, 'epoch': 0.63},
    {'loss': 0.1855, 'grad_norm': 1.3267991542816162, 'learning_rate': 1.8841427735517847e-05, 'epoch': 0.63},
    {'loss': 0.0514, 'grad_norm': 0.18024934828281403, 'learning_rate': 1.8695143358689292e-05, 'epoch': 0.64},
    {'loss': 0.0875, 'grad_norm': 0.16895420849323273, 'learning_rate': 1.854885898186074e-05, 'epoch': 0.64},
    {'loss': 0.1201, 'grad_norm': 0.15858447551727295, 'learning_rate': 1.8402574605032183e-05, 'epoch': 0.64},
    {'loss': 0.0713, 'grad_norm': 0.07128485292196274, 'learning_rate': 1.8256290228203628e-05, 'epoch': 0.65},
    {'loss': 0.0703, 'grad_norm': 1.0293654203414917, 'learning_rate': 1.8110005851375077e-05, 'epoch': 0.65},
    {'loss': 0.0678, 'grad_norm': 0.11217452585697174, 'learning_rate': 1.796372147454652e-05, 'epoch': 0.65},
    {'loss': 0.0623, 'grad_norm': 0.1317310929298401, 'learning_rate': 1.7817437097717964e-05, 'epoch': 0.65},
    {'loss': 0.0581, 'grad_norm': 0.13270703506469727, 'learning_rate': 1.767115272088941e-05, 'epoch': 0.65},
    {'loss': 0.0546, 'grad_norm': 0.17773187136650085, 'learning_rate': 1.7524868344060856e-05, 'epoch': 0.65},
    {'loss': 0.0814, 'grad_norm': 0.22764745330810547, 'learning_rate': 1.7378583967232303e-05, 'epoch': 0.65},
    {'loss': 0.0899, 'grad_norm': 0.2831631603240967, 'learning_rate': 1.7232299590403758e-05, 'epoch': 0.65},
    {'loss': 0.0945, 'grad_norm': 0.35307070684432983, 'learning_rate': 1.7086015213575205e-05, 'epoch': 0.66},
    {'loss': 0.0874, 'grad_norm': 0.4327012896537781, 'learning_rate': 1.6939730836746652e-05, 'epoch': 0.66},
    {'loss': 0.0777, 'grad_norm': 0.4357554316520691, 'learning_rate': 1.6793446459918098e-05, 'epoch': 0.66},
    {'loss': 0.0711, 'grad_norm': 0.4646820722579956, 'learning_rate': 1.6647162083089545e-05, 'epoch': 0.66},
    {'loss': 0.0682, 'grad_norm': 0.5272325873374939, 'learning_rate': 1.650087770626099e-05, 'epoch': 0.66},
    {'loss': 0.0659, 'grad_norm': 0.589508056640625, 'learning_rate': 1.6354593329432436e-05, 'epoch': 0.66},
    {'loss': 0.0631, 'grad_norm': 0.6715608839988708, 'learning_rate': 1.6208308952603883e-05, 'epoch': 0.66},
    {'loss': 0.0617, 'grad_norm': 0.7637172341346741, 'learning_rate': 1.6062024575775328e-05, 'epoch': 0.66},
    {'loss': 0.0604, 'grad_norm': 0.7984013557434082, 'learning_rate': 1.5915740198946775e-05, 'epoch': 0.66},
    {'loss': 0.0582, 'grad_norm': 0.8315630550384521, 'learning_rate': 1.576945582211822e-05, 'epoch': 0.66},
    {'loss': 0.0575, 'grad_norm': 0.9287873501777649, 'learning_rate': 1.5623171445289666e-05, 'epoch': 0.66},
    {'loss': 0.0563, 'grad_norm': 1.0424489974975586, 'learning_rate': 1.5476887068461113e-05, 'epoch': 0.66},
    {'loss': 0.0557, 'grad_norm': 1.1776325702667236, 'learning_rate': 1.5330602691632558e-05, 'epoch': 0.66},
    {'loss': 0.0545, 'grad_norm': 1.2720104455947876, 'learning_rate': 1.5184318314804005e-05, 'epoch': 0.66},
    {'loss': 0.0531, 'grad_norm': 1.3431495428085327, 'learning_rate': 1.5038033937975452e-05, 'epoch': 0.66},
    {'loss': 0.0524, 'grad_norm': 1.4201513528823853, 'learning_rate': 1.4891749561146897e-05, 'epoch': 0.66},
    {'loss': 0.0519, 'grad_norm': 1.4912285804748535, 'learning_rate': 1.4745465184318344e-05, 'epoch': 0.66},
    {'loss': 0.0512, 'grad_norm': 1.5371888875961304, 'learning_rate': 1.4599180807489789e-05, 'epoch': 0.66},
    {'loss': 0.0504, 'grad_norm': 1.6057673692703247, 'learning_rate': 1.4452896430661236e-05, 'epoch': 0.66},
    {'loss': 0.0500, 'grad_norm': 1.6311578750610352, 'learning_rate': 1.4306612053832683e-05, 'epoch': 0.66}
]


# Extract the data points for plotting
epochs = [entry['epoch'] for entry in data]
loss = [entry['loss'] for entry in data]
grad_norm = [entry['grad_norm'] for entry in data]
learning_rate = [entry['learning_rate'] for entry in data]

# Create the plot with subplots
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))

# Loss plot
ax1.plot(epochs, loss, label="Loss", color="blue", marker='o')
ax1.set_title('Loss vs Epoch')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.grid(True)
ax1.legend()

# Gradient Norm plot
ax2.plot(epochs, grad_norm, label="Gradient Norm", color="green", marker='o')
ax2.set_title('Gradient Norm vs Epoch')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Gradient Norm')
ax2.grid(True)
ax2.legend()

# Learning Rate plot
ax3.plot(epochs, learning_rate, label="Learning Rate", color="red", marker='o')
ax3.set_title('Learning Rate vs Epoch')
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Learning Rate')
ax3.grid(True)
ax3.legend()

# Adjust layout for better spacing
plt.tight_layout()

# Show the plot
plt.show()
