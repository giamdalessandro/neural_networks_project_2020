from sklearn.metrics import accuracy_score, mean_absolute_percentage_error, jaccard_similarity_score

#ro_i = elementwiseprod(w_v, x_i)
#Ro_i = prod(A, ro_i)


def obj_part_errors(y_true, y_pred):

    return accuracy_score(y_true, y_pred, normalize= True)
    '''
    Y_i = ?
    Ro_i_spec = diff(y_i, Y_i)
    
    num = 0
    den = 0

    for i in I:
        num += diff( Ro_i, Ro_i_spec)
        den += y_i
    
    m1 = frac(num, den)
    return m1
    '''

def contrib_fitness(y_true, y_pred):

    return jaccard_similarity_score(y_true, y_pred)


def pred_error(y_true, y_pred):

    return mean_absolute_percentage_error(y_true, y_pred)


