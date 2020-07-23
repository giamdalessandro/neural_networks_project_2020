# README 40

### Algorithm of Interpretable CNN

The algorithm focuses on the learning of a single filter `f` in the target conv-layer.
For doing this, we add a loss to the `feature map x` of `f` after the ReLu operation.
So the structure is:
- in --> CONV --> RELU --> Loss for filter f --> out

### Feature map x 
`x` is an `n`x`n` matrix where each element is in __R*__.

- The feature map is the output of one filter applied to the previous  layer. A given filter is drawn across the entire previous layer, moved  one pixel at a time. Each position results in an activation of the  neuron and the output is collected in the feature map. 
- CNN look for "features" such as straight lines, or cats. Whenever  you spot those features they get reported to the feature map. Each  feature map is looking for something else. One feature map could be  looking for straight lines, the other for curves. The feature maps also  look for their features in different locations.
  - __Luca__: quindi una feature map ha un valore su ogni pixel tanto grande quanto è maggiore l'attivazione del filtro relativo (se cerco una linea retta e ho un'immagine bianca con solo un linea, in corrispodenza della linea avrò dei picchi mentre nel resto della feature map avrò zeri)

Since the the object corresponding to the filter `f` can appear in multiple locations  and in the image there are `n^2` possible locations, we design `n^2` __templates__ for `f`.

### Templates

Each template `T_i` is also an `n`x`n` matrix that describes the ideal distribution of activations for `x` when the target part mainly trigger the `i`-th unit in `x`.

- __Luca__: i template sono tutti i possibili modi di attivare il filtro f, cioè tutte le possibili zone in cui può apparire quella object part. Di queste viene poi selezionata solo quella con l'attivazione maggiore per evitare duplicati.
  - Perchè non usano una sliding window con lo stride per fare questo?