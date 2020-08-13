import tensorflow as tf
from tensorflow import keras

# TODO
#   - Disentangled CNN
#       - aggiungere loss filtri
#   - Build decision trees

# GPU check
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))



#   2. modificare filtri nel top conv-layer --> aggiungere maschere
#   3. aggiungere un nuovo conv-layer con M=512 filtri --> ogni filtro è un tensore 3x3xM
#   4. aggiungere maschere per i filtri del nuovo conv-layer
#   5. usare gli stessi FC inizializzati random