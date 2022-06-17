# confidence selection
from scipy.stats import entropy


def get_least_confident_points(model, data_loader, budget):
    '''
    based on entropy score, can chagnge, but make sure to get max or min accordingly
    '''
    uncertainty_estimates = []
    indices_all = []
    for data in data_loader:
        images, labels, expert_preds, indices, _ = data
        images, labels, expert_preds = images.to(device), labels.to(device), expert_preds.to(device)
        outputs = model(images)
        batch_size = outputs.size()[0]  
        for i in range(0, batch_size):
            output_i =  outputs.data[i].cpu().numpy()
            entropy_i = entropy(output_i)
            #entropy_i = 1 - max(output_i)
            uncertainty_estimates.append(entropy_i)
            indices_all.append(indices[i].item())
    indices_all = np.array(indices_all)
    top_budget_indices = np.argsort(uncertainty_estimates)[-budget:]
    actual_indices = indices_all[top_budget_indices]
    uncertainty_estimates = np.array(uncertainty_estimates)
    return actual_indices


error_confidence_trials = []
for trial in range(MAX_TRIALS):
    print(f'\n \n \n Trial {trial} \n \n \n ')

    all_indices = list(range(len(train_dataset.indices)))
    all_data_x = np.array(train_dataset.dataset.data)[train_dataset.indices]
    all_data_y = np.array(train_dataset.dataset.targets)[train_dataset.indices]

    indices_labeled  = Intial_random_set
    indices_unlabeled= list(set(all_indices) - set(indices_labeled))

    dataset_train_labeled = CifarExpertDataset(all_data_x[indices_labeled], all_data_y[indices_labeled], Expert.predict , [1]*len(indices_labeled), indices_labeled)
    dataset_train_unlabeled = CifarExpertDataset(all_data_x[indices_unlabeled], all_data_y[indices_unlabeled], Expert.predict , [0]*len(indices_unlabeled), indices_unlabeled)

    dataLoaderTrainLabeled = DataLoader(dataset=dataset_train_labeled, batch_size=128, shuffle=True,  num_workers=0, pin_memory=True)
    dataLoaderTrainUnlabeled = DataLoader(dataset=dataset_train_unlabeled, batch_size=128, shuffle=True,  num_workers=0, pin_memory=True)

    model_expert = NetSimple(2, 100, 100, 1000,500).to(device)
    run_expert(model_expert, EPOCH_TRAIN, dataLoaderTrainLabeled, dataLoaderTrainLabeled) 
    data_sizes = []
    error_confidence = []
    data_sizes.append(INITIAL_SIZE)
    metrics_confidence = metrics_print_2step(model, model_expert, Expert.predict, 10, dataLoaderVal)
    error_confidence.append(metrics_confidence['system accuracy'])
    for round in range(MAX_ROUNDS):
        print(f'\n \n Round {round} \n \n')
        indices_confidence = get_least_confident_points(model_expert, dataLoaderTrainUnlabeled, BATCH_SIZE_AL)
        indices_labeled  = indices_labeled + list(indices_confidence) 
        indices_unlabeled= list(set(all_indices) - set(indices_labeled))
        dataset_train_labeled = CifarExpertDataset(all_data_x[indices_labeled], all_data_y[indices_labeled], Expert.predict , [1]*len(indices_labeled))
        dataset_train_unlabeled = CifarExpertDataset(all_data_x[indices_unlabeled], all_data_y[indices_unlabeled], Expert.predict , [0]*len(indices_unlabeled))
        dataLoaderTrainLabeled = DataLoader(dataset=dataset_train_labeled, batch_size=128, shuffle=True,  num_workers=0, pin_memory=True)
        dataLoaderTrainUnlabeled = DataLoader(dataset=dataset_train_unlabeled, batch_size=128, shuffle=True,  num_workers=0, pin_memory=True)
        
        run_expert(model_expert, EPOCH_TRAIN, dataLoaderTrainLabeled, dataLoaderTrainLabeled)

        metrics_confidence = metrics_print_2step(model, model_expert, Expert.predict, 10, dataLoaderVal)
        error_confidence.append(metrics_confidence['system accuracy'])
        data_sizes.append((round+1)*BATCH_SIZE_AL + INITIAL_SIZE)
    error_confidence_trials.append(error_confidence)


def get_least_confident_points_ensemble(models, data_loader, budget):
    '''
    based on entropy score, can chagnge, but make sure to get max or min accordingly
    '''
    uncertainty_estimates = []
    indices_all = []
    for data in data_loader:
        images, labels, expert_preds, indices, _ = data
        images, labels, expert_preds = images.to(device), labels.to(device), expert_preds.to(device)
        outputs_all = []
        for model in models:
            output = model(images)
            outputs_all.append(output)
        batch_size = outputs_all[0].size()[0]  
        for i in range(0, batch_size):
            outputs_np = []
            entropies = []
            predictions = []
            for outputs in outputs_all:
                output_i =  outputs.data[i].cpu().numpy()
                outputs_np.append(output_i)
                predictions.append(np.argmax(output_i))
                entropies.append(entropy(output_i))
            majority_prediction = np.round(np.average(predictions))
            disagreement_score = sum((predictions[i] != majority_prediction)*entropies[i] for i in range(len(predictions)))

            entropy_i = np.std(entropies)
            uncertainty_estimates.append(disagreement_score)
            indices_all.append(indices[i].item())
            
    indices_all = np.array(indices_all)
    top_budget_indices = np.argsort(uncertainty_estimates)[-budget:]
    actual_indices = indices_all[top_budget_indices]
    uncertainty_estimates = np.array(uncertainty_estimates)
    return actual_indices

# confidence ensemble selection
from scipy.stats import entropy

error_confidenceensemble_trials = []
ENSEMBLE_SIZE = 10
for trial in range(MAX_TRIALS):
    print(f'\n \n \n Trial {trial} \n \n \n ')

    all_indices = list(range(len(train_dataset.indices)))
    all_data_x = np.array(train_dataset.dataset.data)[train_dataset.indices]
    all_data_y = np.array(train_dataset.dataset.targets)[train_dataset.indices]

    indices_labeled  = Intial_random_set
    indices_unlabeled= list(set(all_indices) - set(indices_labeled))

    dataset_train_labeled = CifarExpertDataset(all_data_x[indices_labeled], all_data_y[indices_labeled], Expert.predict , [1]*len(indices_labeled), indices_labeled)
    dataset_train_unlabeled = CifarExpertDataset(all_data_x[indices_unlabeled], all_data_y[indices_unlabeled], Expert.predict , [0]*len(indices_unlabeled), indices_unlabeled)

    dataLoaderTrainLabeled = DataLoader(dataset=dataset_train_labeled, batch_size=128, shuffle=True,  num_workers=0, pin_memory=True)
    dataLoaderTrainUnlabeled = DataLoader(dataset=dataset_train_unlabeled, batch_size=128, shuffle=True,  num_workers=0, pin_memory=True)

    model_expert_OG = NetSimple(2, 100, 100, 1000,500).to(device)
    run_expert(model_expert, EPOCH_TRAIN, dataLoaderTrainLabeled, dataLoaderTrainLabeled) 
    data_sizes = []
    error_confidence = []
    data_sizes.append(INITIAL_SIZE)
    metrics_confidence = metrics_print_2step(model, model_expert_OG, Expert.predict, 10, dataLoaderVal)
    error_confidence.append(metrics_confidence['system accuracy'])
    for round in range(MAX_ROUNDS):
        print(f'\n \n Round {round} \n \n')
        ensemble_experts = []
        for ens in range(ENSEMBLE_SIZE):
            model_expert = NetSimple(2, 100, 100, 1000,500).to(device)
            run_expert(model_expert, EPOCH_TRAIN, dataLoaderTrainLabeled, dataLoaderTrainLabeled)
            ensemble_experts.append(model_expert)
        indices_confidence = get_least_confident_points_ensemble(ensemble_experts, dataLoaderTrainUnlabeled, BATCH_SIZE_AL)
        indices_labeled  = indices_labeled + list(indices_confidence) 
        indices_unlabeled= list(set(all_indices) - set(indices_labeled))
        dataset_train_labeled = CifarExpertDataset(all_data_x[indices_labeled], all_data_y[indices_labeled], Expert.predict , [1]*len(indices_labeled))
        dataset_train_unlabeled = CifarExpertDataset(all_data_x[indices_unlabeled], all_data_y[indices_unlabeled], Expert.predict , [0]*len(indices_unlabeled))
        dataLoaderTrainLabeled = DataLoader(dataset=dataset_train_labeled, batch_size=128, shuffle=True,  num_workers=0, pin_memory=True)
        dataLoaderTrainUnlabeled = DataLoader(dataset=dataset_train_unlabeled, batch_size=128, shuffle=True,  num_workers=0, pin_memory=True)
        
        run_expert(model_expert_OG, EPOCH_TRAIN, dataLoaderTrainLabeled, dataLoaderTrainLabeled)
        metrics_confidence = metrics_print_2step(model, model_expert_OG, Expert.predict, 10, dataLoaderVal)
        error_confidence.append(metrics_confidence['system accuracy'])
        data_sizes.append((round+1)*BATCH_SIZE_AL + INITIAL_SIZE)
    error_confidenceensemble_trials.append(error_confidence)


# confidence of rejector selection
from scipy.stats import entropy


def get_least_confident_rejector(model, model_exp, data_loader, budget):
    '''
    based on entropy score, can chagnge, but make sure to get max or min accordingly
    '''
    uncertainty_estimates = []
    indices_all = []
    for data in data_loader:
        images, labels, expert_preds, indices, _ = data
        images, labels, expert_preds = images.to(device), labels.to(device), expert_preds.to(device)
        outputs_mod = model(images)
        outputs_exp = model_exp(images)
        batch_size = outputs_mod.size()[0]  
        _, predicted = torch.max(outputs_mod.data, 1)
        for i in range(0, batch_size):
            output_i=  outputs_mod.data[i].cpu().numpy()[predicted[i].item()]
            output_exp = outputs_exp.data[i][1].item() 
            #r_score = -abs(output_exp - output_i) +  entropy(outputs_exp.data[i].cpu().numpy())
            r_score = 1 - output_i
            r_actual = (output_exp >= output_i)
            error_score = 0
            ai_is_correct = (predicted[i].item() != labels[i].item()) * 1.0
            error_score = ai_is_correct * (1 - output_i)
            #if r_actual == 1:
            #    error_score = (expert_preds[i].item() != labels[i].item())*1.0 + entropy(output_exp)
            #else:
            #    error_score = (predicted[i].item() != labels[i].item())*1.0 + entropy(output_i)
            uncertainty_estimates.append(r_score)
            indices_all.append(indices[i].item())
    uncertainty_estimates = np.array(uncertainty_estimates)
    indices_all = np.array(indices_all)
    top_budget_indices = np.argsort(uncertainty_estimates)[-budget:]
    print(uncertainty_estimates[top_budget_indices])
    actual_indices = indices_all[top_budget_indices]
    return actual_indices

error_confidence_rejector_trials = []
for trial in range(MAX_TRIALS):
    all_indices = list(range(len(train_dataset.indices)))
    all_data_x = np.array(train_dataset.dataset.data)[train_dataset.indices]
    all_data_y = np.array(train_dataset.dataset.targets)[train_dataset.indices]

    indices_labeled  = Intial_random_set
    indices_unlabeled= list(set(all_indices) - set(indices_labeled))

    dataset_train_labeled = CifarExpertDataset(all_data_x[indices_labeled], all_data_y[indices_labeled], Expert.predict , [1]*len(indices_labeled), indices_labeled)
    dataset_train_unlabeled = CifarExpertDataset(all_data_x[indices_unlabeled], all_data_y[indices_unlabeled], Expert.predict , [0]*len(indices_unlabeled), indices_unlabeled)

    dataLoaderTrainLabeled = DataLoader(dataset=dataset_train_labeled, batch_size=128, shuffle=True,  num_workers=0, pin_memory=True)
    dataLoaderTrainUnlabeled = DataLoader(dataset=dataset_train_unlabeled, batch_size=128, shuffle=True,  num_workers=0, pin_memory=True)

    model_expert = NetSimple(2, 100, 100, 1000,500).to(device)
    run_expert(model_expert, EPOCH_TRAIN, dataLoaderTrainLabeled, dataLoaderTrainLabeled) 
    data_sizes = []
    error_confidence_rejector = []
    data_sizes.append(INITIAL_SIZE)
    metrics_confidence = metrics_print_2step(model, model_expert, Expert.predict, 10, dataLoaderVal)
    error_confidence_rejector.append(metrics_confidence['system accuracy'])
    for round in range(MAX_ROUNDS):
        print(f'\n \n Round {round} \n \n')
        #if round % 2 == 1:
        #    indices_confidence = random.sample(indices_unlabeled, BATCH_SIZE_AL)
        #else:
        indices_confidence = get_least_confident_rejector(model, model_expert, dataLoaderTrainUnlabeled, BATCH_SIZE_AL)
        indices_labeled  = indices_labeled + list(indices_confidence) 
        indices_unlabeled= list(set(all_indices) - set(indices_labeled))
        dataset_train_labeled = CifarExpertDataset(all_data_x[indices_labeled], all_data_y[indices_labeled], Expert.predict , [1]*len(indices_labeled))
        dataset_train_unlabeled = CifarExpertDataset(all_data_x[indices_unlabeled], all_data_y[indices_unlabeled], Expert.predict , [0]*len(indices_unlabeled))
        dataLoaderTrainLabeled = DataLoader(dataset=dataset_train_labeled, batch_size=128, shuffle=True,  num_workers=0, pin_memory=True)
        dataLoaderTrainUnlabeled = DataLoader(dataset=dataset_train_unlabeled, batch_size=128, shuffle=True,  num_workers=0, pin_memory=True)
        #model_expert = NetSimple(2, 100, 100, 1000,500).to(device)
        run_expert(model_expert, EPOCH_TRAIN, dataLoaderTrainLabeled, dataLoaderTrainLabeled)

        metrics_confidence = metrics_print_2step(model, model_expert, Expert.predict, 10, dataLoaderVal)
        error_confidence_rejector.append(metrics_confidence['system accuracy'])
        data_sizes.append((round+1)*BATCH_SIZE_AL + INITIAL_SIZE)
    error_confidence_rejector_trials.append(error_confidence_rejector)




# confidence of rejector selection but with L_CE loss

all_indices = list(range(len(train_dataset.indices)))
all_data_x = np.array(train_dataset.dataset.data)[train_dataset.indices]
all_data_y = np.array(train_dataset.dataset.targets)[train_dataset.indices]

intial_random_set = random.sample(all_indices, INITIAL_SIZE)
indices_labeled  = intial_random_set
indices_unlabeled= list(set(all_indices) - set(indices_labeled))

dataset_train_labeled = CifarExpertDataset(all_data_x[indices_labeled], all_data_y[indices_labeled], Expert.predict , [1]*len(indices_labeled), indices_labeled)
dataset_train_unlabeled = CifarExpertDataset(all_data_x[indices_unlabeled], all_data_y[indices_unlabeled], Expert.predict , [0]*len(indices_unlabeled), indices_unlabeled)

dataLoaderTrainLabeled = DataLoader(dataset=dataset_train_labeled, batch_size=128, shuffle=True,  num_workers=0, pin_memory=True)
dataLoaderTrainUnlabeled = DataLoader(dataset=dataset_train_unlabeled, batch_size=128, shuffle=True,  num_workers=0, pin_memory=True)

model_lce = NetSimple(n_dataset + 1, 100, 100, 1000,500).to(device)
run_reject_class(model_lce, 10, dataLoaderTrainUnlabeled, dataLoaderVal)

#run_reject(model, 10, Expert.predict, 70,0.5, dataLoaderTrain, dataLoaderVal)


def get_least_confident_rejector_uncertain(model, data_loader, budget):
    '''
    based on entropy score, can chagnge, but make sure to get max or min accordingly
    '''
    uncertainty_estimates = []
    indices_all = []
    for data in data_loader:
        images, labels, expert_preds, indices, _ = data
        images, labels, expert_preds = images.to(device), labels.to(device), expert_preds.to(device)
        outputs_mod = model(images)
        batch_size = outputs_mod.size()[0]  
        _, predicted = torch.max(outputs_mod.data, 1)
        for i in range(0, batch_size):
            output_i=  outputs_mod.data[i].cpu().numpy()[predicted[i].item()]
            output_exp = outputs_exp.data[i][1].item()
            all_output_exp =  outputs_exp.data[i].cpu().numpy()
            entropy_exp = entropy(all_output_exp)
            r_score = -abs(output_exp - output_i) + entropy_exp
            uncertainty_estimates.append(r_score)
            indices_all.append(indices[i].item())
    indices_all = np.array(indices_all)
    top_budget_indices = np.argsort(uncertainty_estimates)[-budget:]
    actual_indices = indices_all[top_budget_indices]
    return actual_indices

#model_expert = NetSimple(2, 100, 100, 1000,500).to(device)
#run_expert(model_expert, 10, dataLoaderTrainLabeled, dataLoaderTrainLabeled) 
data_sizes = []
errors_LCE = []
data_sizes.append(INITIAL_SIZE)
metrics_confidence = metrics_print(model_lce, Expert.predict, n_dataset, dataLoaderVal)
errors_LCE.append(metrics_confidence['system accuracy'])
for round in range(MAX_ROUNDS):
    print(f'\n \n Round {round} \n \n')
    #indices_confidence = get_least_confident_rejector(model, model_expert, dataLoaderTrainUnlabeled, BATCH_SIZE_AL)
    indices_confidence = random.sample(indices_unlabeled, BATCH_SIZE_AL)
    indices_labeled  = indices_labeled + list(indices_confidence) 
    indices_unlabeled= list(set(all_indices) - set(indices_labeled))
    dataset_train_labeled = CifarExpertDataset(all_data_x[indices_labeled], all_data_y[indices_labeled], Expert.predict , [1]*len(indices_labeled))
    dataset_train_unlabeled = CifarExpertDataset(all_data_x[indices_unlabeled], all_data_y[indices_unlabeled], Expert.predict , [0]*len(indices_unlabeled))
    dataLoaderTrainLabeled = DataLoader(dataset=dataset_train_labeled, batch_size=128, shuffle=True,  num_workers=0, pin_memory=True)
    dataLoaderTrainUnlabeled = DataLoader(dataset=dataset_train_unlabeled, batch_size=128, shuffle=True,  num_workers=0, pin_memory=True)
    
    #run_expert(model_expert, 20, dataLoaderTrainLabeled, dataLoaderVal)
    run_reject(model_lce, 10, Expert.predict, 1, 0.5, dataLoaderTrainLabeled, dataLoaderTrainLabeled)
    metrics_confidence = metrics_print(model_lce, Expert.predict, n_dataset, dataLoaderVal)
    errors_LCE.append(metrics_confidence['system accuracy'])
    data_sizes.append((round+1)*BATCH_SIZE_AL + INITIAL_SIZE)






# teaching baseline
import copy
from scipy.stats import entropy
all_indices = list(range(len(train_dataset.indices)))
all_data_x = np.array(train_dataset.dataset.data)[train_dataset.indices]
all_data_y = np.array(train_dataset.dataset.targets)[train_dataset.indices]

intial_random_set = random.sample(all_indices, INITIAL_SIZE)
indices_labeled  = intial_random_set
indices_unlabeled= list(set(all_indices) - set(indices_labeled))

dataset_train_labeled = CifarExpertDataset(all_data_x[indices_labeled], all_data_y[indices_labeled], Expert.predict , [1]*len(indices_labeled), indices_labeled)
dataset_train_unlabeled = CifarExpertDataset(all_data_x[indices_unlabeled], all_data_y[indices_unlabeled], Expert.predict , [0]*len(indices_unlabeled), indices_unlabeled)

dataLoaderTrainLabeled = DataLoader(dataset=dataset_train_labeled, batch_size=128, shuffle=True,  num_workers=0, pin_memory=True)
dataLoaderTrainUnlabeled = DataLoader(dataset=dataset_train_unlabeled, batch_size=128, shuffle=True,  num_workers=0, pin_memory=True)


model_expert = NetSimple(2, 100, 100, 1000,500).to(device)
run_expert(model_expert, 10, dataLoaderTrainLabeled, dataLoaderTrainLabeled) 
data_sizes = []
errors_teaching = []
data_sizes.append(INITIAL_SIZE)
metrics_confidence = metrics_print_2step(model, model_expert, Expert.predict, 10, dataLoaderVal)
errors_teaching.append(metrics_confidence['system accuracy'])
RANDOM_SEARCH_SIZE = 30
for round in range(MAX_ROUNDS):
    print(f'\n \n Round {round} \n \n')
    random_sets = []
    best_set_score = 0
    best_set = []
    saved_expert_model = copy.deepcopy(model_expert.state_dict())
    for trial_set in range(RANDOM_SEARCH_SIZE):
        model_expert.load_state_dict(saved_expert_model)
        random_set = random.sample(indices_unlabeled, BATCH_SIZE_AL)
        random_sets.append(random_set)
        indices_labeled_trial  = indices_labeled + list(random_set) 
        indices_unlabeled_trial= list(set(all_indices) - set(indices_labeled))
        dataset_train_labeled = CifarExpertDataset(all_data_x[indices_labeled], all_data_y[indices_labeled], Expert.predict , [1]*len(indices_labeled))
        dataset_train_unlabeled = CifarExpertDataset(all_data_x[indices_unlabeled], all_data_y[indices_unlabeled], Expert.predict , [0]*len(indices_unlabeled))
        dataLoaderTrainLabeled = DataLoader(dataset=dataset_train_labeled, batch_size=128, shuffle=True,  num_workers=0, pin_memory=True)
        dataLoaderTrainUnlabeled = DataLoader(dataset=dataset_train_unlabeled, batch_size=128, shuffle=True,  num_workers=0, pin_memory=True)
        run_expert(model_expert, 20, dataLoaderTrainLabeled, dataLoaderVal)
        metrics_confidence = metrics_print_2step(model, model_expert, Expert.predict, 10, dataLoaderVal)
        error_random_set = metrics_confidence['system accuracy']
        if error_random_set >= best_set_score:
            best_set_score = error_random_set
            best_set = random_set

    model_expert.load_state_dict(saved_expert_model)
    indices_labeled  = indices_labeled + list(best_set) 
    indices_unlabeled = list(set(all_indices) - set(indices_labeled))
    dataset_train_labeled = CifarExpertDataset(all_data_x[indices_labeled], all_data_y[indices_labeled], Expert.predict , [1]*len(indices_labeled))
    dataset_train_unlabeled = CifarExpertDataset(all_data_x[indices_unlabeled], all_data_y[indices_unlabeled], Expert.predict , [0]*len(indices_unlabeled))
    dataLoaderTrainLabeled = DataLoader(dataset=dataset_train_labeled, batch_size=128, shuffle=True,  num_workers=0, pin_memory=True)
    dataLoaderTrainUnlabeled = DataLoader(dataset=dataset_train_unlabeled, batch_size=128, shuffle=True,  num_workers=0, pin_memory=True)
    run_expert(model_expert, 20, dataLoaderTrainLabeled, dataLoaderVal)
    metrics_confidence = metrics_print_2step(model, model_expert, Expert.predict, 10, dataLoaderVal)
    errors_teaching.append(metrics_confidence['system accuracy'])
    data_sizes.append((round+1)*BATCH_SIZE_AL + INITIAL_SIZE)