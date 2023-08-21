from train_eval import *

save_path_base = '/home/annika/mlde/gb1_results/'

mlde_sim = MLDESim(save_path=save_path_base, 
                   encoding='one-hot', 
                   model_class='boosting', 
                   n_samples=149360, 
                   train_name='gb1_train.csv',
                   test_name='gb1_test.csv', 
                   validation_name='k-fold', 
                   first_append=True, 
                   feat_to_predict='Fitness', 
                   neural_network=True)

if 'feed' or 'cnn' in mlde_sim.model_class:
    mlde_sim.run_neural_network(learning_rate=1e-3, num_epochs=23)
else: 
    mlde_sim.train_all()
