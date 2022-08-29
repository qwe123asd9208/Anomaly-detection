# 专业：计算机科学与技术
# author: Yixian Luo

from data_processing import *
from model import *

def train_model(num_of_feature_with_window,W,num_of_feature_with_value):
    train_data_pool_complete_Xt = train_data_pool_complete
    train_data_pool_complete_yt = train_data_pool_complete_label
    train_data_pool_complete_weight = (99 * np.asarray(train_data_pool_label_vital)) + 1
    train_data_pool_complete_Xv = test_data_pool[0]
    train_data_pool_complete_yv = test_time_series_dataset_label[0][2 * max(W):]

    Xv_yv_tuple = (np.asarray(train_data_pool_complete_Xv), np.asarray(train_data_pool_complete_yv))
    # Fit the model
    assert (len(train_data_pool_complete_Xt) ==
            len(train_data_pool_complete_yt) == len(train_data_pool_complete_weight))

    print('Keras: start to train DNN!')
    start_time = time.time()

    class_weight = {0: 1., 1: 1000.}
    # sample_weight is used when the dataset is not augmented
    model.fit(np.asarray(train_data_pool_complete_Xt),
              np.ravel(train_data_pool_complete_yt),
              validation_split=0,
              validation_data=Xv_yv_tuple,
              epochs=200,
              batch_size=100,
              class_weight=class_weight,
              sample_weight=train_data_pool_complete_weight,
              verbose=1)

    end_time = time.time()
    print('It took %d seconds to train the model!' % (end_time - start_time))
    # get validation metrics
    train_data_pool_complete_Xv_check = np.ravel(model.predict(np.asarray(train_data_pool_complete_Xv)) > 0.5)
    assert (len(train_data_pool_complete_Xv_check) == len(train_data_pool_complete_yv))

    # output the results
    print('%f precision' % (precision_score(train_data_pool_complete_yv, train_data_pool_complete_Xv_check)))
    print('%f recall' % (recall_score(train_data_pool_complete_yv, train_data_pool_complete_Xv_check)))
    print('%f f1_score' % (f1_score(train_data_pool_complete_yv, train_data_pool_complete_Xv_check)))
    # get training metrics
    train_data_pool_complete_check = np.ravel(model.predict(np.asarray(train_data_pool_complete)) > 0.5)
    assert (len(train_data_pool_complete_check) == len(train_data_pool_complete_label))

    # output the results
    print('%f precision' % (precision_score(train_data_pool_complete_label, train_data_pool_complete_check)))
    print('%f recall' % (recall_score(train_data_pool_complete_label, train_data_pool_complete_check)))
    print('%f f1_score' % (f1_score(train_data_pool_complete_label, train_data_pool_complete_check)))
    # get testing metrics
    test_time_series_dataset_flaglist = np.array([])
    for i in np.arange(test_time_series_dataset_size):
        test_time_series_dataset_flag = np.concatenate((np.zeros(2 * max(W)), \
                                                        np.ravel(model.predict(np.asarray(test_data_pool[i])) > 0.5)))

        print('No. %d, %f precision, %f recall, %f f1_score'
              % (i,
                 precision_score(test_time_series_dataset_label[i], test_time_series_dataset_flag),
                 recall_score(test_time_series_dataset_label[i], test_time_series_dataset_flag),
                 f1_score(test_time_series_dataset_label[i], test_time_series_dataset_flag)))

        test_time_series_dataset_flaglist = np.concatenate((test_time_series_dataset_flaglist,
                                                            test_time_series_dataset_flag))

    # get overall results
    assert (len(phase2_test['label']) == len(test_time_series_dataset_flaglist))

    print('Overall statistics:')
    print('%f precision' % (precision_score(phase2_test['label'], test_time_series_dataset_flaglist)))
    print('%f recall' % (recall_score(phase2_test['label'], test_time_series_dataset_flaglist)))
    print('%f f1_score' % (f1_score(phase2_test['label'], test_time_series_dataset_flaglist)))

    predict = pd.DataFrame({'KPI ID': [str(item) for item in phase2_test['KPI ID']],
                            'timestamp': phase2_test['timestamp'],
                            'predict': test_time_series_dataset_flaglist})
    predict.to_csv('predict.csv', index=False)

    #!python evaluation.py '../input/phase2_ground_truth.hdf' 'predict.csv'7

def test_model():
    for index in np.arange(train_time_series_dataset_size):
        plt.subplot(3, 1, 1)
        plt.plot(train_time_series_dataset[index], 'b')
        plt.subplot(3, 1, 2)
        plt.plot(train_time_series_dataset_scaled[index], 'r')
        plt.subplot(3, 1, 3)
        plt.plot(train_time_series_dataset_label[index], 'k')
        plt.show()
    for index in np.arange(test_time_series_dataset_size):
        print(index)
        plt.subplot(3, 1, 1)
        plt.plot(test_time_series_dataset_label[index], 'b')
        plt.subplot(3, 1, 2)
        plt.plot(test_time_series_dataset_scaled[index], 'r')
        plt.subplot(3, 1, 3)
        plt.plot(np.concatenate((np.zeros(2 * max(W)), np.ravel(model.predict(np.asarray(test_data_pool[index]))))),
                 'k')
        plt.show()
    index = 11

    plt.subplot(3, 1, 1)
    plt.plot(test_time_series_dataset_label[index], 'b')
    plt.subplot(3, 1, 2)
    plt.plot(test_time_series_dataset_scaled[index], 'r')
    plt.subplot(3, 1, 3)
    plt.plot(np.concatenate((np.zeros(2 * max(W)), np.ravel(model.predict(np.asarray(test_data_pool[index]))))), 'k')
    plt.show()

    for fid in np.arange(num_of_feature_with_window * len(W) + num_of_feature_with_value):
        sequence = [point[fid] for point in test_data_pool[index]]

        print(fid + 1,
              (fid - num_of_feature_with_value) // num_of_feature_with_window + 1,
              (fid + 1 - num_of_feature_with_value) % num_of_feature_with_window)
        plt.plot(sequence)
        plt.show()

if __name__ == '__main__':
    # read datasets as pandas dataframes
    phase2_train = pd.read_csv('./input/phase2_train.csv')
    phase2_test = pd.read_hdf('./input/phase2_ground_truth.hdf')
    show_dataset(phase2_train)
    show_dataset(phase2_test)
    train_time_series_dataset, \
    train_time_series_dataset_label = get_time_series_from_file(phase2_train)
    print()
    test_time_series_dataset, \
    test_time_series_dataset_label = get_time_series_from_file(phase2_test)
    # here we use the basic one (minmax_scale()) to get the baseline performance
    train_time_series_dataset_size = len(train_time_series_dataset)
    test_time_series_dataset_size = len(test_time_series_dataset)

    train_time_series_dataset_scaled = []
    test_time_series_dataset_scaled = []

    for i in np.arange(train_time_series_dataset_size):
        train_time_series_dataset_scaled.append(minmax_scale(train_time_series_dataset[i]))

    for i in np.arange(test_time_series_dataset_size):

        test_time_series_dataset_scaled.append(minmax_scale(test_time_series_dataset[i]))

    # plot an example after scaling/normalization/transforming, random index
    print('Here is a training example of a time series after/before scaling/normalization/transforming:')
    plot_time_series_info(train_time_series_dataset,
                          train_time_series_dataset_scaled,
                          train_time_series_dataset_label,
                          random.randint(0, train_time_series_dataset_size))

    print('Here is a testing example of a time series after/before scaling/normalization/transforming:')
    plot_time_series_info(test_time_series_dataset,
                          test_time_series_dataset_scaled,
                          test_time_series_dataset_label,
                          random.randint(0, test_time_series_dataset_size))
    num_of_feature_with_value = 9
    num_of_feature_with_window = 12
    train_data_pool_complete, train_data_pool_complete_label,train_data_pool_label_vital=train_dataset_gene(train_time_series_dataset_size,test_time_series_dataset_scaled,test_time_series_dataset_label)
    test_data_pool=test_dataset_gene(test_time_series_dataset_size,test_time_series_dataset_scaled,test_time_series_dataset_label)
    model = model_dnn(num_of_feature_with_window, W, num_of_feature_with_value)
    train_model(num_of_feature_with_window, W, num_of_feature_with_value)
    test_model()