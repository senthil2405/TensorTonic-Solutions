import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    # pes = []
    # for pos in range(int(seq_len)):
    #     pe = []
    #     for i in range(int(d_model)):
            
    #         if i == d_model-1 and d_model%2 != 0:
    #             theta = (pos/base^((2*i)/int(d_model)))
    #             pe.append(np.sin(theta))
    #         elif i%2 == 0:
    #             theta = (pos/base^((2*i)/int(d_model)))
    #             pe.append(np.sin(theta))
    #         else:
    #             theta = (pos/base^((2*i)/int(d_model)))
    #             pe.append(np.cos(theta))
        
    #     pes.append(pe)


    even_indx = [i for i in range(d_model-1) if i%2==0]
    odd_indx = [i for i in range(d_model-1) if i%2!=0]
    if d_model%2 ==0:
        odd_indx += [int(d_model)-1]
    else:
        even_indx += [int(d_model)-1]

    thetas = np.power(([base]*(int(d_model))), (-2*np.array([int(i/2)  for i in range(d_model)])/d_model)).reshape(-1, 1) # (d_modelx1)
    
    pos = np.array([j for j in range(seq_len)]).reshape(-1, 1) # (seq_len x 1)
    pes = np.dot(pos, np.transpose(thetas))

    
    pes[:, even_indx] = np.sin(pes[:, even_indx])
    pes[:, odd_indx] = np.cos(pes[:, odd_indx])
        
    
    return pes
                