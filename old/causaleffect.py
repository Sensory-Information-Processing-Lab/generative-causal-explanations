from __future__ import division
import numpy as np
import torch


"""
ind_uncond:
    Sample-based estimate of negative "independent, unconditional" causal effect,
                            -1/K * sum_i I(alpha_i; Yhat).
Inputs:
    - params["alpha_dim"]
    - params["Nalpha"]
    - params["Nbeta"]
    - params["decoder_net"]
    - params["gamma"] (if decoder_net == 'linGauss')
    - decoder
    - classifier
    - What (if decoder_net = 'linGauss')
Outputs:
    - negCausalEffect (sample-based estimate of - 1/K * sum_i I(alpha_i; Yhat))
    - info["Xhat"]
    - info["yhat"]
    - info["negHYhat"] : -1/K * H(p(Yhat))
    - info["EalphaHYhatgivenalpha"] : 1/K * E_alpha[H(p(Yhat|alpha)]
    (Note: negCausalEffect = negHYhat + EalphaHYhatgivenalpha
"""
def ind_uncond(params, decoder, classifier, device, What=None):
    # params["Ni"] = params["Nalpha"] # TODO - check this and replace Ni below
    # params["No"] = params["Nbeta"] # TODO - check this and replace No below
    latent_vec = np.zeros((params["alpha_dim"]*params["No"]*params["Ni"],params["z_dim"]))
    count = 0
    for kk in range(params["alpha_dim"]):
        for m in range(params["No"]):
            ind_sample_val = np.random.randn(1)
            if params["break_up_ce"] == True:
                latent_vec = np.zeros((params["Ni"],params["z_dim"]))
                count =0
            for n in range(params["Ni"]):
                latent_vec_temp  = np.random.randn(params["z_dim"])
                latent_vec_temp[kk] = ind_sample_val
                latent_vec[count,:] = latent_vec_temp                    
                count += 1
            if params["break_up_ce"] == True:   
                latent_vec_torch = torch.from_numpy(latent_vec).float().to(device)
                Xhat_single = decoder(latent_vec_torch)
                yhat_single = classifier(Xhat_single)[0]
                if kk == 0 and m == 0:
                    Xhat = Xhat_single
                    yhat = yhat_single
                else:
                    #Xhat = torch.cat((Xhat,Xhat_single),0)
                    yhat = torch.cat((yhat,yhat_single),0)
    if params["break_up_ce"] == False:                
        latent_vec_torch = torch.from_numpy(latent_vec).float().to(device)
        if params["decoder_net"] == 'linGauss':
            Xhat = decoder(latent_vec_torch, What, params["gamma"])
        elif params["decoder_net"] == 'nonLinGauss':
            Xhat,Xmu,Xstd = decoder(latent_vec_torch)
        elif params["decoder_net"] == 'VAE' or params["decoder_net"] == 'VAE_CNN': 
            Xhat = decoder(latent_vec_torch)
        # This classifier outputs the label and the hyperplane classifier weights
        yhat = classifier(Xhat)[0]
    I_sum = 0.0
    # Note that latent_vec is alpha_dim*No*Ni in length. The following
    # for loop runs through the loops over K' and N_o from our write up
    eps_add = 1e-8
    negHYhat = 0.0
    EalphaHYhatgivenalpha = 0.0
    for p in range(0, params["alpha_dim"]):
        I_sum_p = 0.0
        qo_vec = torch.zeros(yhat.shape[1]).to(device)
        for m in range(0, params["No"]):
            y_use = yhat[int((p*params["No"] + m)*params["Ni"]):int((p*params["No"]+m+1)*params["Ni"]),:]
            q_vec =  1/float(params["Ni"])*torch.sum(y_use,0)
            q_log = torch.log(q_vec+eps_add*torch.ones_like(q_vec))
            I_sum_p += 1/float(params["No"])*torch.sum(torch.mul(q_vec,q_log))
            EalphaHYhatgivenalpha -= 1/float(params["No"])*torch.sum(torch.mul(q_vec,q_log))
            qo_vec = qo_vec + 1/float(params["No"])*q_vec
        qo_log = torch.log(qo_vec+eps_add*torch.ones_like(qo_vec))
        I_sum_p -= torch.sum(torch.mul(qo_vec,qo_log))
        negHYhat += torch.sum(torch.mul(qo_vec,qo_log))
        I_sum -= I_sum_p
    negCausalEffect = 1. / params["alpha_dim"] * I_sum
    negHYhat *= 1. / params["alpha_dim"]
    EalphaHYhatgivenalpha *= 1. / params["alpha_dim"]
    info = {"Xhat" : Xhat,
            "yhat" : yhat,
            "negHYhat" : negHYhat,
            "EalphaHYhatgivenalpha" : EalphaHYhatgivenalpha}
    return negCausalEffect, info
        

"""
ind_cond:
    Sample-based estimate of negative "independent, conditional" causal effect,
                     - 1/K * sum_i I(alpha_i; Yhat | beta).
Inputs:
    - params["alpha_dim"]
    - params["Nalpha"]
    - params["Nbeta"]
    - params["decoder_net"]
    - params["gamma"] (if decoder_net == 'linGauss')
    - decoder
    - classifier
    - What (if decoder_net = 'linGauss')
Outputs:
    - negCausalEffect (sample-based estimate of - 1/K * sum_i I(alpha_i; Yhat | beta))
    - info["Xhat"]
    - info["yhat"]
    - info["H_Yhatgivenbeta"]
    - info["Ealpha_H_Yhatgivenalphabeta"]
"""
def ind_cond(params, decoder, classifier, device, What=None):
    params["Ni"] = params["Nalpha"] # TODO - check this and replace Ni below
    params["No"] = params["Nbeta"] # TODO - check this and replace No below
    latent_vec = np.zeros((params["alpha_dim"]*params["No"]*params["Ni"],params["z_dim"]))
    count = 0
    # Loop over causal dimensions
    for kk in range(0, params["alpha_dim"]):
        # Loop over the number of samples of the outer loop values
        for m in range(0, params["No"]):
            latent_vec_temp = np.random.randn(params["z_dim"])
            if params["break_up_ce"] == True:
                latent_vec = np.zeros((params["Ni"],params["z_dim"]))
                count =0
            # Loop over the number of samples of the inner loop values
            for n in range(0,params["Ni"]):
                ind_sample_val = np.random.randn(1)
                latent_vec_temp[kk] = ind_sample_val
                latent_vec[count,:] = latent_vec_temp
                count += 1
            if params["break_up_ce"] == True:   
                latent_vec_torch = torch.from_numpy(latent_vec).float().to(device)
                Xhat_single = decoder(latent_vec_torch)
                yhat_single = classifier(Xhat_single)[0]
                if kk == 0 and m == 0:
                    Xhat = Xhat_single
                    yhat = yhat_single
                else:
                    #Xhat = torch.cat((Xhat,Xhat_single),0)
                    yhat = torch.cat((yhat,yhat_single),0)
    if params["break_up_ce"] == False:  
        latent_vec_torch = torch.from_numpy(latent_vec).float().to(device)
        if params["decoder_net"] == 'linGauss':
            Xhat = decoder(latent_vec_torch, What, params["gamma"])
        elif params["decoder_net"] == 'nonLinGauss':
        	Xhat,Xmu,Xstd = decoder(latent_vec_torch)
        elif params["decoder_net"] == 'VAE' or params["decoder_net"] == 'VAE_CNN': 
            Xhat = decoder(latent_vec_torch)
        # This classifier outputs the label and the hyperplane classifier weights
        yhat = classifier(Xhat)[0]
    I_sum = 0.0
    # Note that latent_vec is alpha_dim*No*Ni in length. The following
    # for loop runs through the loops over K' and N_o from our write up
    eps_add = 1e-8
    H_Yhatgivenbeta = 0.0
    Ealpha_H_Yhatgivenalphabeta = 0.0
    for p in range(0, params["alpha_dim"]):
        I_sum_p = 0.0
        for m in range(0, params["No"]):
            y_use = yhat[int((p*params["No"] + m)*params["Ni"]):int((p*params["No"]+m+1)*params["Ni"]),:]
            y_log = torch.log(y_use+eps_add*torch.ones_like(y_use))
            I_m = 1/float(params["Ni"])*torch.sum(torch.mul(y_use,y_log))
            Ealpha_H_Yhatgivenalphabeta -= 1/float(params["No"])*1/float(params["Ni"])*torch.sum(torch.mul(y_use,y_log))
            q_vec =  1/float(params["Ni"])*torch.sum(y_use,0)
            q_log = torch.log(q_vec+eps_add*torch.ones_like(q_vec))
            I_m = I_m - torch.sum(torch.mul(q_vec,q_log))
            H_Yhatgivenbeta += 1/float(params["No"])*torch.sum(torch.mul(q_vec,q_log))
            I_sum_p = I_sum_p + 1/float(params["No"])*I_m
        I_sum = I_sum - I_sum_p
    info = {"Xhat" : Xhat, 
            "yhat" : yhat,
            "H_Yhatgivenbeta" : H_Yhatgivenbeta,
            "Ealpha_H_Yhatgivenalphabeta":Ealpha_H_Yhatgivenalphabeta}
    negCausalEffect = 1. / params["alpha_dim"] * I_sum
    return negCausalEffect, info


"""
joint_uncond:
    Sample-based estimate of "joint, unconditional" causal effect,
                        - I(alpha; Yhat).
Inputs:
    - params["alpha_dim"]
    - params["Nalpha"]
    - params["Nbeta"]
    - params["decoder_net"]
    - params["gamma"] (if decoder_net == 'linGauss')
    - decoder
    - classifier
    - What (if decoder_net = 'linGauss')
Outputs:
    - negCausalEffect (sample-based estimate of - I(alpha; Yhat))
    - info["Xhat"]
    - info["yhat"]
"""
def joint_uncond(params, decoder, classifier, device, What=None):
    params["Ni"] = params["Nalpha"] # TODO - check this and replace Ni below
    params["No"] = params["Nbeta"] # TODO - check this and replace No below
    # Generate data associated and classify the output
    latent_vec = np.zeros((params["No"]*params["Ni"], params["z_dim"]))
    count = 0
    # causal factors are in range(0,alpha_dim)
    # Loop over the number of samples of the outer loop values
    eps_add = 1e-8        
    I_sum = 0.0
    qo_vec = torch.zeros(params["y_dim"]).to(device)
    for m in range(0, params["No"]):
        alpha_sample_val = np.random.randn(params["alpha_dim"]) 
        latent_vec = np.zeros((params["Ni"],params["z_dim"]))   
        count = 0
        # Loop over the number of samples of the inner loop values
        for n in range(0, params["Ni"]):
            latent_vec_temp = np.random.randn(params["z_dim"])
            latent_vec_temp[:params["alpha_dim"]] = alpha_sample_val
            latent_vec[count,:] = latent_vec_temp
            count += 1
        latent_vec_torch = torch.from_numpy(latent_vec).float().to(device)
        if params["decoder_net"] == 'linGauss':
            Xhat = decoder(latent_vec_torch, What, params["gamma"])
        elif params["decoder_net"] == 'nonLinGauss':
        	Xhat, Xmu, Xstd = decoder(latent_vec_torch)
        elif params["decoder_net"] == 'VAE' or params["decoder_net"] == 'VAE_CNN':  
            Xhat = decoder(latent_vec_torch)
        #Xhat_single = decoder(latent_vec_torch)
        yhat = classifier(Xhat)[0]
    # Note that latent_vec is No*Ni in length. The following
    # for loop runs through the loops over K' and N_o from our write up
        q_vec =  1/float(params["Ni"])*torch.sum(yhat,0)
        q_log = torch.log(q_vec+eps_add*torch.ones_like(q_vec))
        I_sum = I_sum + 1/float(params["No"])*torch.sum(torch.mul(q_vec,q_log))
        qo_vec = qo_vec + 1/float(params["No"])*q_vec
    qo_log = torch.log(qo_vec+eps_add*torch.ones_like(qo_vec))
    I_sum = I_sum - torch.sum(torch.mul(qo_vec,qo_log))
    I_sum *= -1 # I_sum is defined as negative information
    negCausalEffect = I_sum
    info = {"Xhat" : Xhat, "yhat" : yhat}
    return negCausalEffect, info


"""
joint_cond:
    Sample-based estimate of "joint, conditional" causal effect,
                   - I(alpha; Yhat | beta).
Inputs:
    - params["alpha_dim"]
    - params["Nalpha"]
    - params["Nbeta"]
    - params["decoder_net"]
    - params["gamma"] (if decoder_net == 'linGauss')
    - decoder
    - classifier
    - What (if decoder_net = 'linGauss')
Outputs:
    - negCausalEffect (sample-based estimate of - I(alpha; Yhat | beta))
    - info["Xhat"]
    - info["yhat"]
"""
def joint_cond(params, decoder, classifier, device, What=None):
    params["Ni"] = params["Nalpha"] # TODO - check this and replace Ni below
    params["No"] = params["Nbeta"] # TODO - check this and replace No below
    latent_vec = np.zeros((params["No"]*params["Ni"], params["z_dim"]))
    count = 0
    # causal factors are in range(0,alpha_dim)
    # Loop over the number of samples of the outer loop values
    for m in range(0, params["No"]):
        latent_vec_temp = np.random.randn(params["z_dim"])
        if params["break_up_ce"] == True:
                latent_vec = np.zeros((params["Ni"],params["z_dim"]))
                count =0
        # Loop over the number of samples of the inner loop values
        for n in range(0, params["Ni"]):
            alpha_sample_val = np.random.randn(params["alpha_dim"])
            latent_vec_temp[:params["alpha_dim"]] = alpha_sample_val
            latent_vec[count,:] = latent_vec_temp
            count += 1
        if params["break_up_ce"] == True:   
            latent_vec_torch = torch.from_numpy(latent_vec).float().to(device)
            Xhat_single = decoder(latent_vec_torch)
            yhat_single = classifier(Xhat_single)[0]
            if m == 0:
                Xhat = Xhat_single
                yhat = yhat_single
            else:
                #Xhat = torch.cat((Xhat,Xhat_single),0)
                yhat = torch.cat((yhat,yhat_single),0)
    if params["break_up_ce"] == False:  
        latent_vec_torch = torch.from_numpy(latent_vec).float().to(device)
        if params["decoder_net"] == 'linGauss':
            Xhat = decoder(latent_vec_torch, What, params["gamma"])
        elif params["decoder_net"] == 'nonLinGauss':
        	Xhat, Xmu, Xstd = decoder(latent_vec_torch)
        elif params["decoder_net"] == 'VAE' or params["decoder_net"] == 'VAE_CNN': 
            Xhat = decoder(latent_vec_torch)
        	    # This classifier outputs the label and the hyperplane classifier weights
        yhat = classifier(Xhat)[0]
    I_sum = 0.0        
    # Note that latent_vec is No*Ni in length. The following
    # for loop runs through the loops over K' and N_o from our write up
    eps_add = 1e-8
    I_sum = 0.0
    for m in range(0, params["No"]):
        y_use = yhat[int(m*params["Ni"]):int((m+1)*params["Ni"]),:]
        y_log = torch.log(y_use+eps_add*torch.ones_like(y_use))
        I_m = 1/float(params["Ni"])*torch.sum(torch.mul(y_use,y_log))
        q_vec =  1/float(params["Ni"])*torch.sum(y_use,0)
        q_log = torch.log(q_vec+eps_add*torch.ones_like(q_vec))
        I_m = I_m - torch.sum(torch.mul(q_vec,q_log))
        I_sum = I_sum + 1/float(params["No"])*I_m
    I_sum *= -1 # I_sum is defined as negative information 
    negCausalEffect = I_sum
    info = {"Xhat" : Xhat, "yhat" : yhat}
    return negCausalEffect, info
