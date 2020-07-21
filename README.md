# neural_networks_project_2020

Repo for the project of NN 2020 based on the paper "Interpreting CNNs via Decision Trees"

## Definitions
- `object part`: associates each disentangled filter with an explicit semantic meaning. This association enables linguistic descriptions of middle-layer knowledge, for example, how many parts are memorized in the CNN and how the parts are organized.
- `rationale`: we define the rationale of a CNN prediction as the set of object parts (or filters) that are activated and contribute to the prediction. Given different input images, the

## Research Objectives
- How to explain features of a middle layer in a CNN at the semantic level. I.e. we aim to transform chaotic features of filters inside a CNN into semantically meaningful concepts, such as object parts, so as to help people to understand the knowledge in the CNN.
- How to quantitatively analyze the rationale of each CNN prediction. We need to figure out which filters/parts pass their information through the CNN and contribute to the prediction output. We also report the numerical contribution of each filter (or object part) to the output score.

## Explanation Tree 
- **nodes**: represents a decision node, *i.e.* a set of meaningful rationales
    - **root**: set of all decision nodes, *i.e.* set of all possible rationales 
- **leaves**: represent the specific decision node of a certain image, *i.e.* the rationale of a specific image  

### building the exaplanation Tree
1. learn filters to represents object parts???
2. assign each filter with a specific part name??
3. ¿¿mine?? the decision nodes to explain how the CNN use the filters and construct the tree