class RNA_SAECOLLABNET:

    # set inicial da classe
  def __init__(self, N_input, N_out, method):

    self.N_layers = {} # dictionary of layer information
    self.N_layers_Out = {} # dictionary of layer information with change function
    self.N_layers_Hide = {} # dictionary of layer information with change function
    self.N_layers_Outbranch = {} # dictionary of continue train branch
    self.N_layers_Hidebranch = {} # dictionary of continue train branch
    self.N_layers_Normal = {} # dictionary of normal train
    self.N_layers_Normalbranch = {}

    self.N_input = N_input
    self.N_out = N_out
    self.def_train = 'normal'
    self.Use_cv = False
    self.call_cont = 0
    self.t_loss = 'mse'
    self.lr = []
    self.t_metrics = []
    self.method = method #definition of the method to be used
    self.change_func = False




    # add new layers
  def addbranch(self, N_neurons, activation, use_bias=[None]):

    #defined te use bias in the layers
    use_bias = list(map(lambda use: 1 if use == True else 0, use_bias))
    if len(use_bias) == 1:
      use_bias.append(0)

    # check if N_neurons is int
    if isinstance(N_neurons, int):
      Nh = N_neurons
    elif len(N_neurons) == 2:
      Nh = N_neurons[0]
      Ne = N_neurons[1]
    elif len(N_neurons) == 1:
      Nh = N_neurons[0]
      Ne = 0


    #getting the number of existing layers
    layers = len(self.N_layers)


    #creating new layer
    layers += 1

    func_layer = {}

   # analyzes the existence of layers, if true, it concatenates the
   # layers woh(previous layer ) and whi (current layer )
    c = [1,0,1]
    if layers>1:
      c = [[0,1], 0, 1]

      if self.method == 'M1' or self.method == 'M3':
        Dk = 0
      if self.method == 'M2' or self.method == 'M4':
        Dk = 1

      N_input = np.size(self.N_layers[layers-1][0],0)

      Nh = Nh - self.N_out

      #generating the weights
      whi = np.random.random((Nh, N_input)) - 0.5
      woh = np.zeros((self.N_out, Nh))

      #generating the bias
      bias_hi = np.random.random((Nh,1)) - 0.5
      bias_oh = np.zeros((self.N_out,1))


      # concatenation of weights whi
      whi = np.concatenate((self.N_layers[layers-1][2], whi), axis = 0)

      # concatenation of bias
      bias_hi = np.concatenate((self.N_layers[layers-1][3], bias_hi), axis = 0)

      # concatenation of weights woh
      a = np.identity(self.N_out)
      woh = np.concatenate((a, woh), axis = 1)


      # concatenation of activation functions

      func_layer[0] = [self.N_layers[layers-1][8][1], activation[0]]
       # np.concatenate((self.N_layers[layers-1][8][1], activation[0]), axis = 0)
      func_layer[1] = activation[1]

      whi_ext = None
      woh_ext = None
      bias_hi_ext = None
      func_layer[2] = None
      if self.method == 'M3' or self.method == 'M4':

        if len(use_bias) == 2:
          use_bias.append(0)
        #    use_bias.append(0)
        #  elif len(use_bias) == 2:
        #    use_bias.append(0)

        #generating te weights of extra branch
        whi_ext = np.random.random((Ne,self.N_input)) - 0.5
        woh_ext = np.random.random((Nh,Ne)) - 0.5

        #generating the bias of extra branch
        bias_hi_ext = np.random.random((Ne,1)) - 0.5

        # defineting the activation function of extra branch
        func_layer[2] = activation[2]

    # if it's the first layer
    else:

      # generating the weights
      whi = np.random.random((Nh, self.N_input)) - 0.5
      woh = np.random.random((self.N_out, Nh)) - 0.5

      # generating the bias
      bias_hi = np.random.random((Nh,1)) - 0.5
      bias_oh = np.zeros((self.N_out,1))

      #creating string vector of activation functions
      func_layer[0] = activation[0]
      func_layer[1] = activation[1]

      whi_ext = None
      woh_ext = None
      bias_hi_ext = None
      func_layer[2] = None
      Dk = 1



    # add the informations of new branch in the dictionary
    self.N_layers[layers] = [whi, bias_hi, woh, bias_oh, whi_ext, bias_hi_ext, woh_ext, use_bias, func_layer, Dk, c]
    #self.Copy_Nlayers[layers] = [whi, bias_hi, woh, bias_oh, whi_ext, bias_hi_ext, woh_ext, use_bias, func_layer, Dk]



   # if self.Use_cv:
    #  self.Copy_Nlayers[layers] = [whi, bias_hi, woh, bias_oh, whi_ext, bias_hi_ext, woh_ext, use_bias, func_layer, Dk]
   #   self.N_layers_normal[layers] = [whi, bias_hi, woh, bias_oh, whi_ext, bias_hi_ext, woh_ext, use_bias, func_layer, Dk]






  #Step forward
  def feedforward(self, xin, set_train, set_weights, layer_i):

    # Ni_layers = N_layer
    # Nf_layers = len(self.N_layers) + 1



      #Getting te datas of layer in the dictionary
    whi = set_weights[layer_i][0]
    bias_hi = set_weights[layer_i][1]
    woh = set_weights[layer_i][2]
    bias_oh = set_weights[layer_i][3]
    whi_ext = set_weights[layer_i][4]
    bias_hi_ext = set_weights[layer_i][5]
    woh_ext = set_weights[layer_i][6]
    use_bias = set_weights[layer_i][7]
    t_func = set_weights[layer_i][8]
    set_cc = set_weights[layer_i][10]


    c_h = set_cc[0]
    c_o = set_cc[1]
    c_h_ext = set_cc[2]




    if layer_i > 1:
      x = set_train[layer_i-1][1] #getting yh of previous layer
    else:
      x = xin

    #fase feedforward
    mult = np.ones((1, np.size(x, 1)))*use_bias[0]
    net_h = np.dot(whi, x) + np.dot(bias_hi, mult)

    net_h_ext = None
    yh_ext = None

    # if there is an extra branch
    if layer_i == 1:
      yh = func_activation(net_h, t_func[0],c_h)
    else:
      if self.method == 'M3' or self.method == 'M4':

        mult = np.ones((1, np.size(xin ,1)))*use_bias[2]
        net_h_ext = np.dot(whi_ext, xin) + np.dot(bias_hi_ext, mult)

        yh_ext = func_activation(net_h_ext, t_func[2],c_h_ext)

        net_o_ext = np.dot(woh_ext, yh_ext)

        #n = self.N_out#net_h.shape[0] - net_o_ext.shape[0]
        net_h[self.N_out:,:] = net_h[self.N_out:,:] + net_o_ext

      yh = np.zeros((np.size(net_h,0),np.size(net_h,1)))
      yh[0:self.N_out,:] = func_activation(net_h[0:self.N_out,:], t_func[0][0],c_h[0])
      yh[self.N_out:,:] = func_activation(net_h[self.N_out:,:], t_func[0][1],c_h[1])


    #the output of the previous layer will be input to the new layer
    mult = np.ones((1, np.size(yh,1)))*use_bias[1]
    net_o = np.dot(woh,yh) + np.dot(bias_oh,mult)

    out = func_activation(net_o, t_func[1],c_o)


    #useful datas for the step backward
      # if train == True:
    out_all = [net_h, yh, net_o, out, net_h_ext, yh_ext]


    #return the informations of the layer
    return out_all



#step backward
  def backward(self, xin, set_train, set_weights, e, layer_i):

   # if e.ndim == 1:
   #     e = e.reshape(-1, 1)

    #getting the datas of layer
    whi = set_weights[layer_i][0]
    bias_hi = set_weights[layer_i][1]
    woh = set_weights[layer_i][2]
    bias_oh = set_weights[layer_i][3]
    whi_ext = set_weights[layer_i][4]
    bias_hi_ext = set_weights[layer_i][5]
    woh_ext = set_weights[layer_i][6]
    use_bias = set_weights[layer_i][7]
    t_func = set_weights[layer_i][8]
    Dk = set_weights[layer_i][9]
    set_cc = set_weights[layer_i][10]

    c_h = set_cc[0]
    c_o = set_cc[1]
    c_h_ext = set_cc[2]





    #getting the datas of the previous layer
    net_h, yh, net_o, out, net_h_ext, yh_ext = set_train[layer_i]

    if isinstance(self.lr, (int, float)):
      lr_hi = self.lr
      lr_oh = self.lr

    else:
      lr_hi = self.lr[0]
      lr_oh = self.lr[1]


    if layer_i > 1:
      x = set_train[layer_i-1][1]
    else:
      x = xin


    # loss calculation
    erro_dfO = np.zeros((np.size(net_o,0),np.size(net_o,1)))

    erro_dfO = e*func_activation( net_o, t_func[1],c_o, derivate=True)


    # calculation of Dw_oh
    Dw_oh = -lr_oh*np.dot(erro_dfO, yh.T)


    #calculation of backpropagated error
    Dbias_oh = -lr_oh*np.sum(erro_dfO.T, axis= 0, keepdims=True).T

    # calculation of  erro retropopagado
    eh = np.dot(woh.T, erro_dfO)

    if layer_i > 1:
      erro_dfH = np.zeros((np.size(net_h,0),np.size(net_h,1)))
      erro_dfH[0:self.N_out,:] = eh[0:self.N_out,:]*func_activation( net_h[0:self.N_out,:], t_func[0][0], c_h[0], derivate=True)
      erro_dfH[self.N_out:,:] = eh[self.N_out:,:]*func_activation( net_h[self.N_out:,:], t_func[0][1],c_h[1], derivate=True)
    else:
      erro_dfH = eh*func_activation( net_h, t_func[0],c_h, derivate=True)



    #calculation of delta Dw_hi
    Dw_hi = -lr_hi*np.dot(erro_dfH, x.T)

    #calculation of Dbias_hi
    Dbias_hi = -lr_hi*np.sum(erro_dfH.T, axis = 0, keepdims=True).T


    if (self.method =='M3' or self.method == 'M4') and layer_i>1:

      DWoh_ext = -lr_hi*np.dot(erro_dfH[self.N_out:,:], yh_ext.T)
      Eh_ext = np.dot(woh_ext.T,erro_dfH[self.N_out:,:])

      erro_dfH_ext = Eh_ext*func_activation( net_h_ext, t_func[2], c_h_ext, derivate=True)

      DWhi_ext = -lr_hi*np.dot(erro_dfH_ext, xin.T)
      Dbias_hi_ext = -lr_hi*np.sum(erro_dfH_ext.T, axis = 0, keepdims=True).T

      # update weigts and bias of the extra branch
      whi_ext = whi_ext + DWhi_ext
      bias_hi_ext = bias_hi_ext + Dbias_hi_ext
      woh_ext = woh_ext + DWoh_ext



    if layer_i > 1:

      Dw_oh[:,0:self.N_out] = np.multiply(Dw_oh[:,0:self.N_out], np.identity(self.N_out))*Dk
      Dbias_oh[:,0:self.N_out] = Dbias_oh[:,0:self.N_out]*Dk
      Dw_hi[0:self.N_out,:] = Dw_hi[0:self.N_out,:]*0
      Dbias_hi[0:self.N_out,:] = Dbias_hi[0:self.N_out,:]*0




    #update of the weights and bias of the layer
    whi = whi + Dw_hi
    bias_hi = bias_hi + Dbias_hi
    woh = woh + Dw_oh
    bias_oh = bias_oh + Dbias_oh




    #update in the dictionary

    set_weights[layer_i][0] = whi
    set_weights[layer_i][1] = bias_hi
    set_weights[layer_i][2] = woh
    set_weights[layer_i][3] = bias_oh
    set_weights[layer_i][4] = whi_ext
    set_weights[layer_i][5] = bias_hi_ext
    set_weights[layer_i][6] = woh_ext


    return set_weights
    #if layer_i < len(self.N_layers):
    #  self.N_layers[layer_i+1][0][0:self.N_out,:] = woh
    #  self.N_layers[layer_i+1][1][0:self.N_out,:] = bias_oh






  def generate_new_weights(self, set_weights):

    layer = len(set_weights)

    for layer_i in range(1,layer+1):

      whi = set_weights[layer_i][0]
      bias_hi = set_weights[layer_i][1]
      woh = set_weights[layer_i][2]
      bias_oh = set_weights[layer_i][3]
      whi_ext = set_weights[layer_i][4]
      bias_hi_ext = set_weights[layer_i][5]
      woh_ext = set_weights[layer_i][6]


      #novos pesos

      new_whi = np.random.random((whi.shape[0], whi.shape [1])) - 0.5
      new_woh = np.zeros((woh.shape[0], woh.shape [1]))

      new_bias_hi = np.random.random((bias_hi.shape[0], bias_hi.shape [1])) - 0.5
      new_bias_oh = np.zeros((bias_oh.shape[0], bias_oh.shape [1]))

      new_whi_ext = None
      new_woh_ext = None
      new_bias_hi_ext = None

      if layer_i > 1 and ( self.method == 'M3' or self.method == 'M4'):
        new_whi_ext = np.random.random((whi_ext.shape[0], whi_ext.shape [1])) - 0.5
        new_woh_ext = np.random.random((woh_ext.shape[0], woh_ext.shape [1])) - 0.5
        new_bias_hi_ext = np.random.random((bias_hi_ext.shape[0], bias_hi_ext.shape [1])) - 0.5


      set_weights[layer_i][0] = new_whi
      set_weights[layer_i][1] = new_bias_hi
      set_weights[layer_i][2] = new_woh
      set_weights[layer_i][3] = new_bias_oh
      set_weights[layer_i][4] = new_whi_ext
      set_weights[layer_i][5] = new_bias_hi_ext
      set_weights[layer_i][6] = new_woh_ext

      if layer_i > 1:

        set_weights[layer_i][0][0:self.N_out,:] = set_weights[layer_i-1][2]
        set_weights[layer_i][1][0:self.N_out,:] = set_weights[layer_i-1][3]
        set_weights[layer_i][2][:, 0:self.N_out] = np.identity(self.N_out)

    return set_weights






  def fit_train(self, history, change, train_data, validation_data, test_data, epochs_branch, verbose, step):

    set_AvalTrain = {}
    set_AvalTest = {}
    set_AvalValid = {}

    #c = {}
    #history = {}

    set_weights = deepcopy(self.N_layers)



    # history_all = {}
    #history = {metric: [] for metric in self.metrics + [self.t_loss]}
    #history[self.t_loss] = []
    history[change + '_train_' + self.t_loss] = []


    # set metrics for avaliation
    for m in self.t_metrics:
      #train
      history[change + '_train_' + m] = []

      #test
      if test_data is not None:
        history[change + '_test_' + m] = []

      #validation
      if validation_data is not None:
        history[change + '_val_' + m] = []




    #transposition of variables
    X_train, y_train = train_data
    X_train = X_train.T
    y_train = y_train.T


    #checks for data test
    if test_data is not None:
      history[change + '_test_' + self.t_loss] = []
      X_test, y_test = test_data
      X_test = X_test.T
      y_test = y_test.T


    #cheks for data validation
    if validation_data is not None:
      history[change + '_val_' + self.t_loss] = []
      X_val, y_val = validation_data
      X_val = X_val.T
      y_val = y_val.T

    epochs = epochs_branch*len(set_weights)

    for layer_i in range(1,len(set_weights)+1):
            #print(layer_i)

      #update the concatenated weights whi and bias of the new branch with
      #the woh and bias_oh of the old branch

      #set_weights[layer_i][10] = [1, 0, 1]
      if layer_i > 1: #and layer_i <= len(self.N_layers):
        #ant = c[layer_i-1][1]
        #c[layer_i] = [[ant,1], 1, 1]
        #set_weights[layer_i][10] = [[0,1], 0, 1]

        set_weights[layer_i][0][0:self.N_out,:] = set_weights[layer_i-1][2]
        set_weights[layer_i][1][0:self.N_out,:] = set_weights[layer_i-1][3]

      #  if ch_func:
       #   func = self.N_layers[layer_i-1][8][0]
        #  self.N_layers[layer_i][8][1] = self.N_layers[layer_i-1][8][1]
        #  self.N_layers[layer_i][8][2] = self.N_layers[layer_i-1][8][2]

      setc = 0
      epoch_init = (layer_i-1)*epochs_branch + 1
      epoch_finish = epochs_branch*layer_i


      for epoch_i in range(epoch_init,epoch_finish+1):


        #self.c[layer_i] = [[0,1], 0, 1] if layer_i > 1 else [1, 0, 1]
        #if epoch_i > epoch_init+10 :
        setc += 1/(0.7*(epochs_branch))

        if setc >= 0.98:
          setc = 1


        if change == 'ChangeOut':
          set_weights[layer_i][10] = [1, setc, 1]
          if layer_i > 1:
            #ant = set_weights[layer_i-1][10][1]
            set_weights[layer_i][10] = [[1,1], setc, 1]

        elif change == 'ChangeHide':
          set_weights[layer_i][10] = [1, 0, 1]
          if layer_i > 1:
            set_weights[layer_i][10] = [[setc,1], 0, 1]
            set_weights[layer_i-1][10][1] = setc

        elif change == 'Normal':
          set_weights[layer_i][10] = [1, 0, 1]
          if layer_i > 1:
            set_weights[layer_i][10] = [[0,1], 0, 1]







        set_AvalTrain[layer_i] = self.feedforward(X_train, set_AvalTrain, set_weights, layer_i)
        #print(set_train[layer_i][3].shape
        y_hat = set_AvalTrain[layer_i][3]
        #print(y_hat.shape)
        #print(trainY.shape)

        #cálculo do loss
        #print(y_hat-y_train)
        #error = loss(y_train, y_hat, self.t_loss)
        e = loss(y_train, y_hat, self.t_loss, derivate=True)

       # print(y_train.shape)
        #print(y_hat.shape)
       # print(e.shape)
        set_weights = self.backward(X_train, set_AvalTrain, set_weights, e, layer_i)


        error_train = loss(y_train, y_hat, self.t_loss)
        #outAvalTrain = self.predict(X_train, layer_i)

        history[change + '_train_' + self.t_loss].append(np.round(error_train,3))
        #print(len(history['train_'+ self.t_loss]))
        #print(history['train_'+ self.t_loss])
        #set_AvalTest[layer_i] = self.feedforward(testX, set_AvalTest, layer_i)
        #out = set_AvalTest[layer_i][3]

        if test_data is not None:
          #outAvalTest = self.predict(X_test,layer_i)
          set_AvalTest[layer_i] = self.feedforward(X_test, set_AvalTest, set_weights, layer_i)
          outAvalTest = set_AvalTest[layer_i][3]
          error_test = loss(y_test, outAvalTest, self.t_loss)
          history[change + '_test_' + self.t_loss].append(np.round(error_test, 3))

        if validation_data is not None:
          set_AvalValid[layer_i] = self.feedforward(X_val, set_AvalValid, set_weights, layer_i)
          outAvalValid = set_AvalValid[layer_i][3]
          error_val = loss(y_val, outAvalValid, self.t_loss)
          history[change + '_val_' + self.t_loss].append(np.round(error_val, 3))



        if (epoch_i % step == 0 and verbose == 1):
          #error = loss(y_train, y_hat, self.t_loss)
          print(f'Epoch: {epoch_i}/{epochs}', end='')
          print('[',end='')

          for i in range(1,31):
            time.sleep(0.001)
            print('=',end='')

          print(f']  loss: {np.round(error_train,4)} ',end='')
          print('')


        if (epoch_i % step == 0 and verbose == 2):
          # error = loss(y_train, y_hat,self.t_loss)
          print(f'Epoch: {epoch_i}/{epochs}  loss: {np.round(error_train, 4)} ',end='')
          print('')


      #treino
      #outAvalTrain = self.predict(X_train, layer_i)
      for m in history.keys():
       # set_AvalTrain[layer_i] = self.feedforward(X_train, set_AvalTrain, layer_i)
        #outAvalTrain = set_AvalTrain[layer_i][3]
        if change + '_train_' in m:
          name = m.split('_')[2]
          if self.t_loss != name:
            history[m].append( Tmetrics(y_train, y_hat, name) )

      #test
      if test_data is not None:
       # outAvalTest = self.predict(X_test, layer_i)
       # set_AvalTest[layer_i] = self.feedforward(X_test, set_AvalTest, layer_i)
       # outAvalTest = set_AvalTest[layer_i][3]
        for m in history.keys():
          if change + '_test_' in m:
            name = m.split('_')[2]
            if self.t_loss != name:
              history[m].append( Tmetrics(y_test, outAvalTest, name) )

      #validation
      if validation_data is not None:
        for m in history.keys():
          if change + '_val_' in m:
            name = m.split('_')[2]
            if self.t_loss != name:
              history[m].append( Tmetrics(y_val, outAvalValid, name) )



      #print(setc)

      # if training continuation is enabled
      #if self.train_branch:
      #  self.N_layers_change_func_branch[layer_i] = self.N_layers[layer_i]

      if change == 'Normal':
        self.N_layers_Normal = deepcopy(set_weights)

      elif change == 'ChangeHide':
        self.N_layers_Hide = deepcopy(set_weights)

      elif change == 'ChangeOut':
        self.N_layers_Out = deepcopy(set_weights)



    return history, set_AvalTrain, set_AvalValid, set_AvalTest





  def fit_train_branch(self, history, change, set_AvalTrain, set_AvalValid, set_AvalTest, train_data, validation_data,
                                test_data, epochs_branch, verbose, step):


    if change == 'Outbranch':
      name = 'ChangeOut'
      set_weigths = deepcopy(self.N_layers_Out )
    elif change == 'Hidebranch':
      name = 'ChangeHide'
      set_weigths = deepcopy(self.N_layers_Hide)
    elif change == 'Normalbranch':
      name = 'Normal'
      set_weigths = deepcopy(self.N_layers_Normal)


    # history_all = {}
    #history = {metric: [] for metric in self.metrics + [self.t_loss]}
    #history[self.t_loss] = []

    AvalTrain = {}
    AvalTest = {}
    AvalValid = {}
    set_train_branch = {}


    for layer_i in range(1, len(set_weigths)):

      if change == 'Hidebranch':
        set_weigths[layer_i][10] = [1, 0, 1]
        if layer_i > 1:
          set_weigths[layer_i][10] = [[1,1], 0, 1]


      set_train_branch[layer_i]  = set_AvalTrain[layer_i]

      AvalTrain[layer_i] = deepcopy(set_AvalTrain[layer_i])
      AvalTest[layer_i] = deepcopy(set_AvalTest[layer_i])
      AvalValid[layer_i] = deepcopy(set_AvalValid[layer_i])



      history[str(layer_i) + change + '_train_' + self.t_loss] = []
      history[str(layer_i) + change + '_train_' + self.t_loss].append( history[name + '_train_' + self.t_loss][(layer_i*epochs_branch)-1] )

      if validation_data is not None:
        history[str(layer_i) + change + '_val_' + self.t_loss] = []
        history[str(layer_i) + change + '_val_' + self.t_loss].append( history[name + '_val_' + self.t_loss][(layer_i*epochs_branch)-1] )

      if test_data is not None:
        history[str(layer_i) + change + '_test_' + self.t_loss] = []
        history[str(layer_i) + change + '_test_' + self.t_loss].append( history[name + '_test_' + self.t_loss][(layer_i*epochs_branch)-1] )


      # set metrics for avaliation
      for m in self.t_metrics:
        #train
        history[str(layer_i) + change + '_train_' + m] = []
        history[str(layer_i) + change + '_train_' + m].append( history[name + '_train_' + m][layer_i-1] )

        #test
        if test_data is not None:
          history[str(layer_i) + change + '_test_' + m] = []
          history[str(layer_i) + change + '_test_' + m].append( history[name + '_test_' + m][layer_i-1] )

        #validation
        if validation_data is not None:
          history[str(layer_i) + change + '_val_' + m] = []
          history[str(layer_i) + change + '_val_' + m].append( history[name + '_val_' + m][layer_i-1] )



    #transposition of variables
    X_train, y_train = train_data
    X_train = X_train.T
    y_train = y_train.T

    #checks for data validation
    if validation_data is not None:
      X_val, y_val = validation_data
      X_val = X_val.T
      y_val = y_val.T


    if test_data is not None:
      X_test, y_test = test_data
      X_test = X_test.T
      y_test = y_test.T



    epochs = epochs_branch*len(set_weigths)

    for layer_i in range(1, len(set_weigths)):

      if layer_i > 1:
        set_train_branch[layer_i-1] = deepcopy(set_AvalTrain[layer_i-1])

      #set_AvalTest_continue = set_AvalTest[layer_i]
      #set_AvalTrain_continue = set_AvalTrain[layer_i]

      for epoch_i in range((epochs_branch*layer_i)+1 ,epochs+1):

        set_train_branch[layer_i] = self.feedforward(X_train, set_train_branch, set_weigths, layer_i)

        y_hat = set_train_branch[layer_i][3]
        #print(y_hat)
        #print(y_train)


        e = loss(y_train, y_hat, self.t_loss, derivate = True)

       # print(y_train.shape)
        #print(y_hat.shape)
       # print(e.shape)
        set_weigths =  self.backward(X_train, set_train_branch, set_weigths, e, layer_i )


        #error_train = loss(y_train, y_hat, self.t_loss)
        #outAvalTrain = self.predict(X_train, layer_i)
        #outAvalTrain = y_hat #self.feedforward(X_train, AvalTrain, set_weigths, layer_i)[3]
        error_train = loss(y_train, y_hat, self.t_loss)
        history[str(layer_i) + change + '_train_' + self.t_loss].append(np.round(error_train,3))

        #set_AvalTest[layer_i] = self.feedforward(testX, set_AvalTest, layer_i, choose_layer = False)
        #out = set_AvalTest[layer_i][3]

        if test_data is not None:
          #outAvalTest = self.predict(X_test,layer_i)
          outAvalTest = self.feedforward(X_test, AvalTest, set_weigths, layer_i)[3]
          error_test = loss(y_test, outAvalTest, self.t_loss)
          history[str(layer_i) + change + '_test_' + self.t_loss].append(np.round(error_test, 3))

        if validation_data is not None:
          outAvalValid = self.feedforward(X_val, AvalValid, set_weigths, layer_i)[3]
          error_val = loss(y_val, outAvalValid, self.t_loss)
          history[str(layer_i) + change + '_val_' + self.t_loss].append(np.round(error_val, 3))


        if (epoch_i % step == 0 and verbose == 1):
          #error = loss(y_train, y_hat, self.t_loss)
          print(f'Epoch: {epoch_i}/{epochs}', end='')
          print('[', end = '')

          for i in range(1,31):
            time.sleep(0.001)
            print('=', end = '')

          print(f']  loss: {np.round(error_train, 4)} ',end='')
          print('')


        if (epoch_i % step == 0 and verbose == 2):
          # error = loss(y_train, y_hat,self.t_loss)
          print(f'Epoch: {epoch_i}/{epochs}  loss: {np.round(error_train,4)} ',end='')
          print('')



        if epoch_i % epochs_branch == 0:
        #  print(f'y:{y_train}')
        #  print(f'out:{outAvalTrain}')
          #treino
          #outAvalTrain = self.predict(X_train, layer_i)
          for m in history.keys():
            # set_AvalTrain[layer_i] = self.feedforward(X_train, set_AvalTrain, layer_i, choose_layer = False)
            # outAvalTrain = set_AvalTrain[layer_i][3]
            if (str(layer_i) + change + '_train_') in m:
              name = m.split('_')[2]
              if name != self.t_loss:
                history[m].append( Tmetrics(y_train, y_hat, name) )
               # print(f'{m}: {Tmetrics(y_train, outAvalTrain, name)}')

          #test
          if test_data is not None:
          # outAvalTest = self.predict(X_test, layer_i)
          # set_AvalTest[layer_i] = self.feedforward(X_test, set_AvalTest, layer_i, choose_layer = False)
          # outAvalTest = set_AvalTest[layer_i][3]
            for m in history.keys():
              if (str(layer_i) + change + '_test_') in m:
                name = m.split('_')[2]
                if name != self.t_loss:
                  history[m].append( Tmetrics(y_test, outAvalTest, name) )
                  #print(f'{m}: {Tmetrics(y_test, outAvalTest, name)}')

          #validation
          if validation_data is not None:
            for m in history.keys():
              if (str(layer_i) + change + '_val_') in m:
                name = m.split('_')[2]
                if name != self.t_loss:
                  history[m].append( Tmetrics(y_val, outAvalValid, name) )
                  #print(f'{m}: {Tmetrics(y_val, outAvalValid, name)}')


    if change == 'Outbranch':
      self.N_layers_Outbranch = deepcopy(set_weigths)

    elif change == 'Hidebranch':
      self.N_layers_Hidebranch = deepcopy(set_weigths)

    elif change == 'Normalbranch':
      self.N_layers_Normalbranch = deepcopy(set_weigths)


    return history











  def compile(self, lr, loss, metrics, change_func, use_cv=False):
        # self.optimizer = optimizer
    self.t_loss = loss
    self.lr = lr
    self.t_metrics = metrics
    self.change_func = change_func
    self.Use_cv = use_cv # Assign the use_cv value to self.Use_cv

   # if 'Normal' in self.change_func:
    #  self.N_layers_Normal = self.N_layers

   # if 'ChangeOut' in self.change_func:
    #  self.N_layers_Out = self.N_layers

    #if 'ChangeHide' in self.change_func:
   #   self.N_layers_Hide = self.N_layers







  def fit(self, train_data, validation_data = None, test_data = None, epochs_branch = 500, verbose = 0, step = 1):

    history = {}

    #['Normal', 'Change_Out', 'Change_Hide', 'Out_branch', 'Hide_branch']

    if 'Normal' in self.change_func:
      print('Type train:Normal')
      history, set_AvalTrain, set_AvalValid, set_AvalTest = self.fit_train(history, 'Normal', train_data, validation_data, test_data,
                                                                                        epochs_branch , verbose, step)
      if 'Normalbranch' in self.change_func:
        print('Type train:Normal branch')
        history = self.fit_train_branch(history, 'Normalbranch', set_AvalTrain, set_AvalValid, set_AvalTest, train_data, validation_data,
                                          test_data, epochs_branch, verbose, step)

    if 'ChangeOut' in self.change_func:
      print('Type train: Change Out')
      history, set_AvalTrain, set_AvalValid, set_AvalTest  = self.fit_train(history, 'ChangeOut', train_data, validation_data, test_data,
                                                                                        epochs_branch , verbose, step)
      if 'Outbranch' in self.change_func:
        print('Type train: Out branch')
        history = self.fit_train_branch(history, 'Outbranch', set_AvalTrain, set_AvalValid, set_AvalTest, train_data, validation_data,
                                        test_data, epochs_branch, verbose, step)

    if 'ChangeHide' in self.change_func:
      print('Type train: Change Hide')
      history, set_AvalTrain, set_AvalValid, set_AvalTest = self.fit_train(history, 'ChangeHide', train_data, validation_data, test_data,
                                                                                        epochs_branch , verbose, step)
      if 'Hidebranch' in self.change_func:
        print('Type train: Hide branch')
        history = self.fit_train_branch(history, 'Hidebranch', set_AvalTrain, set_AvalValid, set_AvalTest, train_data, validation_data,
                                        test_data, epochs_branch, verbose, step)





    if self.Use_cv:
      self.N_layers = self.generate_new_weights(self.N_layers)



    return history









  # predição da saída
  def predict(self, xin, change = 'Normal'):

    # transposição da variável
    #x = x.T
    set_pred = {}
   # pred = []

    if change == 'Normal':
      set_weights = self.N_layers_Normal
    elif change == 'ChangeOut':
      set_weights = self.N_layers_Out
    elif change == 'ChangeHide':
      set_weights = self.N_layers_Hide


    layer = len(set_weights)



    for layer_i in range(1, layer+1):
      set_pred[layer_i] = self.feedforward(xin, set_pred, set_weights, layer_i)
      pred = set_pred[layer_i][3]

    return pred



