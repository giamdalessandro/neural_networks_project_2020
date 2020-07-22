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

### Section 3

They designed a linear equation to simplify the complex feature processing and it is used for:
1. understanding which object parts activate which CNN filters
2. how much they contribute to the final prediction

In order to do this we need to:
1. ensure that the CNN middle-layer features are meaningful
2. extract the contributions of middle-layer features to the prediction scores

The filter we want to learn (?) is activted by the same object part (bird's head) even in different images where this part can be in different location.

...parte matematica...

The filter loss ensures that given an input image x_f will match only one of the possible location candidates. For doing this, we assume that repetitive shapes on various region probably describe low-level textures rather than high-level shapes (evidences in the paper Interpretable CNN [40]); we are then interested in the top conv-layer filters because they will most likely represent object parts.


